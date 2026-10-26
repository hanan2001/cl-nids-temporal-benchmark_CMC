"""
Baselines de continual learning pour la comparaison du papier IJACSA.

Fournit trois baselines standard, en plus des deux méthodes déjà couvertes par
ton ContinualLearner (Replay uniforme et Risk-Aware Replay) :

  - FineTuning    : borne basse. Entraîne séquentiellement sans aucun mécanisme
                    anti-oubli (ni replay, ni régularisation).
  - EWC           : Elastic Weight Consolidation (Kirkpatrick et al. 2017).
                    Régularise les poids importants des tâches passées via la
                    matrice de Fisher diagonale.
  - JointTraining : borne haute. Entraîne UNE fois sur l'union de toutes les
                    tâches (pas de continual learning) -> performance cible.

Toutes renvoient le même dict de résultats que run_experiment (clés
'metrics', 'forgetting_rate', 'accuracies_history', ...) pour que le tableau
comparatif soit homogène.

Ces baselines réutilisent : CyberNetwork (models), FocalLoss (models),
compute_all_metrics / compute_forgetting_rate_matrix (evaluation),
get_risk_scores (data).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from models import CyberNetwork
from evaluation.metrics import (compute_all_metrics,
                                compute_forgetting_rate_matrix,
                                compute_plasticity_stability_metrics)


# --------------------------------------------------------------------------- #
# Utilitaires communs
# --------------------------------------------------------------------------- #
def _loaders(task, batch_size):
    tr = DataLoader(task['train'], batch_size=batch_size, shuffle=True)
    va = DataLoader(task['val'], batch_size=batch_size, shuffle=False)
    return tr, va


def _evaluate(model, dataset, device, batch_size):
    model.eval()
    preds, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            p = out.argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(yb.numpy().tolist())
    return np.array(labels), np.array(preds)


def _train_epochs(model, train_loader, val_loader, device, config,
                  extra_loss_fn=None):
    """Boucle d'entraînement générique avec early stopping sur val loss."""
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    best, best_state, patience = float('inf'), None, 0
    for epoch in range(config.MAX_EPOCHS_PER_TASK):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            loss.backward()
            opt.step()
        # val loss
        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb), yb).item()
                nb += 1
        vl = vl / max(1, nb)
        if vl < best - config.EARLY_STOPPING_MIN_DELTA:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _finalize(model, tasks, class_names, device, config, risk_mapping,
              method_name, seed, accuracies_history):
    """Calcule métriques finales sur l'agrégat des TEST + forgetting."""
    all_labels, all_preds = [], []
    for t in tasks:
        yl, yp = _evaluate(model, t['test'], device, config.BATCH_SIZE)
        all_labels.extend(yl.tolist())
        all_preds.extend(yp.tolist())
    metrics = compute_all_metrics(all_labels, all_preds, risk_mapping)
    if len(accuracies_history) > 1:
        forget = compute_forgetting_rate_matrix(accuracies_history)
        ps = compute_plasticity_stability_metrics(accuracies_history)
    else:
        forget = 0.0
        ps = {'plasticity': 0.0, 'stability': 0.0, 'tradeoff': 0.0}
    return {
        'method': method_name, 'seed': seed,
        'metrics': metrics,
        'forgetting_rate': forget,
        'plasticity': ps['plasticity'], 'stability': ps['stability'],
        'accuracies_history': accuracies_history,
    }


def _eval_seen_tasks(model, tasks, upto, device, config):
    """Accuracy par tâche vue (0..upto) sur leurs TEST sets — pour le forgetting."""
    accs = []
    for pid in range(upto + 1):
        yl, yp = _evaluate(model, tasks[pid]['test'], device, config.BATCH_SIZE)
        accs.append(float((yl == yp).mean()) if len(yl) else 0.0)
    return accs


# --------------------------------------------------------------------------- #
# 1) Fine-tuning (borne basse)
# --------------------------------------------------------------------------- #
def run_finetuning(tasks, class_names, config, risk_mapping, seed=42,
                   n_classes=None):
    device = config.DEVICE
    n_classes = n_classes or len(class_names)
    in_size = tasks[0]['train'].tensors[0].shape[1]
    model = CyberNetwork(in_size, config.HIDDEN_SIZES, n_classes).to(device)

    acc_hist = []
    for tid, task in enumerate(tasks):
        tr, va = _loaders(task, config.BATCH_SIZE)
        model = _train_epochs(model, tr, va, device, config)
        acc_hist.append(_eval_seen_tasks(model, tasks, tid, device, config))
        print(f"   [FineTuning] tâche {tid+1}/{len(tasks)} done")
    return _finalize(model, tasks, class_names, device, config, risk_mapping,
                     'FineTuning', seed, acc_hist)


# --------------------------------------------------------------------------- #
# 2) EWC (Elastic Weight Consolidation)
# --------------------------------------------------------------------------- #
class _EWC:
    """Accumule Fisher diagonal + moyennes des poids des tâches passées."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.fishers = []   # liste de dict param_name -> Fisher
        self.means = []     # liste de dict param_name -> theta*

    def penalty(self):
        if not self.fishers:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        for fisher, mean in zip(self.fishers, self.means):
            for n, p in params.items():
                if n in fisher:
                    loss = loss + (fisher[n] * (p - mean[n]) ** 2).sum()
        return loss

    def consolidate(self, loader, device, n_batches=50):
        """Estime la Fisher diagonale sur la tâche qui vient d'être apprise."""
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        mean = {n: p.detach().clone() for n, p in params.items()}
        self.model.eval()
        crit = nn.CrossEntropyLoss()
        seen = 0
        for i, (xb, yb) in enumerate(loader):
            if i >= n_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            self.model.zero_grad()
            out = self.model(xb)
            loss = crit(out, yb)
            loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            seen += 1
        for n in fisher:
            fisher[n] /= max(1, seen)
        self.fishers.append(fisher)
        self.means.append(mean)


def run_ewc(tasks, class_names, config, risk_mapping, seed=42, n_classes=None):
    device = config.DEVICE
    n_classes = n_classes or len(class_names)
    in_size = tasks[0]['train'].tensors[0].shape[1]
    model = CyberNetwork(in_size, config.HIDDEN_SIZES, n_classes).to(device)
    ewc = _EWC(model, device)
    lam = getattr(config, 'EWC_LAMBDA', 1000.0)

    acc_hist = []
    for tid, task in enumerate(tasks):
        tr, va = _loaders(task, config.BATCH_SIZE)
        model = _train_epochs(model, tr, va, device, config,
                              extra_loss_fn=lambda: lam * ewc.penalty())
        ewc.consolidate(tr, device)
        acc_hist.append(_eval_seen_tasks(model, tasks, tid, device, config))
        print(f"   [EWC] tâche {tid+1}/{len(tasks)} done")
    return _finalize(model, tasks, class_names, device, config, risk_mapping,
                     'EWC', seed, acc_hist)


# --------------------------------------------------------------------------- #
# 3) Joint training (borne haute)
# --------------------------------------------------------------------------- #
def run_joint(tasks, class_names, config, risk_mapping, seed=42, n_classes=None):
    device = config.DEVICE
    n_classes = n_classes or len(class_names)
    in_size = tasks[0]['train'].tensors[0].shape[1]
    model = CyberNetwork(in_size, config.HIDDEN_SIZES, n_classes).to(device)

    # union de tous les trains / vals
    train_union = ConcatDataset([t['train'] for t in tasks])
    val_union = ConcatDataset([t['val'] for t in tasks])
    tr = DataLoader(train_union, batch_size=config.BATCH_SIZE, shuffle=True)
    va = DataLoader(val_union, batch_size=config.BATCH_SIZE, shuffle=False)
    model = _train_epochs(model, tr, va, device, config)

    # historique "plat" : le joint voit tout d'un coup -> une seule ligne
    acc = _eval_seen_tasks(model, tasks, len(tasks) - 1, device, config)
    acc_hist = [acc]  # forgetting ~ 0 par construction
    print(f"   [Joint] entraînement unique sur {len(tasks)} jours done")
    return _finalize(model, tasks, class_names, device, config, risk_mapping,
                     'Joint', seed, acc_hist)