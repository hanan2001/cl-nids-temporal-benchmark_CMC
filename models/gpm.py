"""
Risk-Guided Gradient Projection Memory (RGGP).

CORRECTIONS MAJEURES vs version précédente:
1. MAPPING DE CLÉS RÉPARÉ. La mémoire est indexée par le nom du tenseur de
   poids ('network.<idx>.weight'), EXACTEMENT le nom renvoyé par
   model.named_parameters(). Avant, la mémoire était indexée par des clés
   d'activation ('linear_0') tandis que project_gradient interrogeait des
   noms de paramètres -> aucune correspondance -> projection JAMAIS appliquée.
2. PROJECTION DIMENSIONNELLEMENT CORRECTE. Pour un poids W de forme
   [out_features, in_features], la base U vit dans l'espace des ENTRÉES
   (dimension in_features = grad.size(1)). On projette donc les LIGNES de W:
       W_proj = W - (W @ U) @ U.T        avec U de forme [in_features, k]
   L'entrée de la couche est capturée par le hook (voir network.py).
3. risk_factor SUPPRIMÉ. Multiplier le gradient projeté par un facteur > 1
   détruisait l'orthogonalité et ne pouvait plus s'appeler "projection".
   Le risk-guiding subsiste UNIQUEMENT via get_weighted_samples (choix des
   échantillons servant à construire la base) et via le keep_ratio.
4. DEBUG intégré (self.debug). Mets gpm.debug = True pour vérifier au premier
   run que la projection s'active réellement.
"""

import torch
import numpy as np
from collections import OrderedDict


class GradientProjectionMemory:

    def __init__(self, k=8, sample_size=50, use_risk=True,
                 risk_threshold=0.5, risk_multiplier=2.0, device='cpu',
                 use_adaptive_risk=True):
        self.k = k
        self.sample_size = sample_size
        self.use_risk = use_risk
        self.risk_threshold = risk_threshold
        self.risk_multiplier = risk_multiplier
        self.device = device
        self.use_adaptive_risk = use_adaptive_risk

        # clé = nom du tenseur de poids, ex 'network.0.weight'
        self.memory = OrderedDict()
        self.class_importance = {}
        self.class_counts = {}
        self.total_projections = 0
        self.risk_weighted_projections = 0

        self.debug = False          # -> True pour tracer l'activation de la projection
        self._debug_printed = False

    # ------------------------------------------------------------------ #
    # Importance des classes
    # ------------------------------------------------------------------ #
    def set_class_importance(self, class_to_risk, class_counts=None):
        self.class_importance = dict(class_to_risk)
        if class_counts is not None:
            self.class_counts = class_counts
            for cls, count in class_counts.items():
                rarity_factor = 1.0 + max(0, 1.0 - count / 500) * 5.0
                if cls in self.class_importance:
                    self.class_importance[cls] = min(
                        1.0, self.class_importance[cls] * rarity_factor)

    # ------------------------------------------------------------------ #
    # Sélection pondérée des échantillons pour construire la base
    # ------------------------------------------------------------------ #
    def get_weighted_samples(self, labels, risk_mapping):
        n = len(labels)
        if n == 0:
            return np.array([], dtype=int)

        if not self.use_risk or not risk_mapping:
            size = min(self.sample_size, n)
            return np.random.choice(n, size=size, replace=False)

        weights = []
        for label in labels:
            label_val = label.item() if hasattr(label, 'item') else label
            risk = risk_mapping.get(label_val, 0.5)
            if self.use_adaptive_risk and self.class_counts:
                count = self.class_counts.get(label_val, 0)
                rarity = max(1.0, 500.0 / (count + 1))
                risk = min(1.0, risk * (1.0 + rarity * 0.2))
            weights.append(risk * self.risk_multiplier if risk > self.risk_threshold else 1.0)

        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n

        size = min(self.sample_size, n)
        return np.random.choice(n, size=size, replace=True, p=weights)

    # ------------------------------------------------------------------ #
    # Construction / mise à jour de la base par couche
    # ------------------------------------------------------------------ #
    def update_memory(self, model, samples, labels, risk_mapping=None):
        if samples is None or len(samples) == 0:
            return

        # comptage des classes du lot courant
        class_counts = {}
        for label in labels:
            v = label.item() if hasattr(label, 'item') else label
            class_counts[v] = class_counts.get(v, 0) + 1
        self.class_counts = class_counts

        if self.use_risk and risk_mapping is not None:
            self.set_class_importance(risk_mapping, class_counts)

        model.eval()
        with torch.no_grad():
            _ = model(samples.to(self.device))
            representations = model.get_representations_for_gpm()

        if not representations:
            if self.debug:
                print("[GPM DEBUG] update_memory: aucune représentation capturée")
            return

        idx = self.get_weighted_samples(labels, risk_mapping)
        if len(idx) == 0:
            return

        for act_key, reps in representations.items():
            # act_key = 'param_network.<i>'  ->  weight_key = 'network.<i>.weight'
            weight_key = act_key.replace('param_', '') + '.weight'

            if not isinstance(reps, torch.Tensor) or reps.dim() < 2:
                continue

            # sous-échantillonnage pondéré (les indices portent sur la dim batch)
            if len(idx) <= reps.size(0):
                reps = reps[idx]
            if reps.size(0) < 2:
                continue

            if reps.dim() > 2:
                reps = reps.view(reps.size(0), -1)

            # standardisation colonne
            mean = reps.mean(dim=0, keepdim=True)
            std = reps.std(dim=0, keepdim=True) + 1e-8
            reps = (reps - mean) / std
            # reps : [n_samples, in_features]

            try:
                if weight_key not in self.memory:
                    # base initiale : colonnes de U couvrant l'espace des entrées
                    # reps.T : [in_features, n_samples]
                    U, _, _ = torch.svd(reps.T)
                    k_actual = min(self.k, U.size(1))
                    self.memory[weight_key] = U[:, :k_actual].to(self.device)
                else:
                    U_old = self.memory[weight_key]          # [in_features, k_old]
                    if reps.size(1) != U_old.size(0):
                        continue
                    # composante orthogonale au sous-espace déjà mémorisé
                    projected = reps - reps @ U_old @ U_old.T
                    if projected.size(0) > 1:
                        U_new, _, _ = torch.svd(projected.T)
                        k_new = min(self.k, U_new.size(1))
                        U_cat = torch.cat([U_old, U_new[:, :k_new].to(self.device)], dim=1)
                        # plafonner : ne jamais dépasser 50% de la dimension d'entrée
                        max_rank = max(1, U_old.size(0) // 2)
                        if U_cat.size(1) > max_rank:
                            U_cat = U_cat[:, :max_rank]
                        self.memory[weight_key] = U_cat

            except Exception as e:
                if self.debug:
                    print(f"[GPM DEBUG] SVD échouée pour {weight_key}: {e}")
                continue

        if self.debug:
            print(f"[GPM DEBUG] mémoire construite: {list(self.memory.keys())}")

    # ------------------------------------------------------------------ #
    # Projection des gradients
    # ------------------------------------------------------------------ #
    def project_gradient(self, model):
        projected_gradients = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.data

            # DEBUG : vérifier au premier appel que les clés matchent
            if self.debug and not self._debug_printed and name.endswith('.weight'):
                print(f"[GPM DEBUG] param={name} | in memory: {name in self.memory} "
                      f"| mem_keys={list(self.memory.keys())[:3]}")

            if name not in self.memory:
                projected_gradients[name] = grad
                continue

            U = self.memory[name]            # [in_features, k]

            # grad d'un poids : [out_features, in_features]
            if grad.dim() != 2 or U.size(0) != grad.size(1):
                projected_gradients[name] = grad
                continue

            try:
                # projection des LIGNES de grad hors du sous-espace mémorisé
                grad_projected = grad - (grad @ U) @ U.T

                # projection douce (préserve la plasticité), pilotée par le risque moyen
                if self.use_risk and self.class_importance:
                    avg_risk = float(np.mean(list(self.class_importance.values())))
                    
                    
                    keep_ratio = 0.7 + 0.2 * avg_risk
                    grad_projected = keep_ratio * grad_projected + (1 - keep_ratio) * grad
                # NB: aucune multiplication par risk_factor -> orthogonalité préservée

                projected_gradients[name] = grad_projected.to(self.device)
                self.total_projections += 1
            except Exception as e:
                if self.debug:
                    print(f"[GPM DEBUG] projection échouée pour {name}: {e}")
                projected_gradients[name] = grad

        if self.debug and not self._debug_printed:
            print(f"[GPM DEBUG] total_projections après 1er appel = {self.total_projections}")
            self._debug_printed = True

        return projected_gradients

    def apply_projected_gradients(self, model, projected_gradients):
        for name, param in model.named_parameters():
            if name in projected_gradients and param.grad is not None:
                param.grad.data = projected_gradients[name].to(param.grad.data.device)

    def clear(self):
        self.memory = OrderedDict()
        self.class_importance = {}
        self.class_counts = {}
        self.total_projections = 0
        self.risk_weighted_projections = 0
        self._debug_printed = False

    def get_stats(self):
        return {
            'total_projections': self.total_projections,
            'risk_weighted_projections': self.risk_weighted_projections,
            'memory_size': len(self.memory),
        }