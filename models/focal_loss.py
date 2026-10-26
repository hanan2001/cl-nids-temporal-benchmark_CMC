"""
Pertes — VERSION CORRIGÉE.

Bug corrigé dans ClassBalancedFocalLoss : les classes absentes recevaient
spc=1, donc le poids maximal ; après normalisation sur les 15 classes, les
classes réellement présentes recevaient ~1 % du poids et toute la perte était
divisée par ~100 (losses affichées 0.0000, early stopping incapable de
détecter une amélioration). Désormais seules les classes présentes ont un
poids, normalisé pour que leur moyenne vaille 1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., CVPR 2019).
    w_c = (1 - beta) / (1 - beta^n_c) pour les classes présentes (n_c > 0),
    0 pour les classes absentes ; moyenne des poids présents = 1.
    class_risk (optionnel) : poids multiplicatif (1 + risque_c) appliqué
    AVANT normalisation -> variante « risk-aware » de la perte.
    """
    def __init__(self, samples_per_class, beta=0.999, gamma=1.0,
                 class_risk=None, max_weight=10.0, device='cpu', legacy=False):
        super().__init__()
        spc = np.asarray(samples_per_class, dtype=np.float64)
        self.legacy = legacy
        if legacy:   # ANCIEN COMPORTEMENT (ablation des pièges uniquement)
            spc = np.where(spc <= 0, 1, spc)
            w = (1.0 - beta) / (1.0 - np.power(beta, spc))
            w = np.clip(w / w.sum() * len(spc), 0.0, max_weight)
            self.register_buffer('weights', torch.tensor(w, dtype=torch.float32, device=device))
            self.gamma = gamma
            return
        present = spc > 0
        w = np.zeros_like(spc)
        w[present] = (1.0 - beta) / (1.0 - np.power(beta, spc[present]))
        if class_risk is not None:
            r = np.asarray(class_risk, dtype=np.float64)
            w[present] *= (1.0 + r[present])
        w[present] = w[present] / w[present].mean()
        w = np.clip(w, 0.0, max_weight)
        self.register_buffer('weights', torch.tensor(w, dtype=torch.float32, device=device))
        self.gamma = gamma

    def forward(self, logits, targets):
        if self.legacy:  # ancien calcul : p_t tiré de la CE déjà pondérée
            ce = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
            return (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        w = self.weights[targets]
        return (w * (1.0 - pt) ** self.gamma * ce).mean()


class FocalLoss(nn.Module):
    """Conservée pour compatibilité avec l'ancien code (non utilisée par cl_methods)."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', risk_mapping=None):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.risk_mapping = risk_mapping

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        fw = (1 - torch.exp(-ce)) ** self.gamma
        if self.risk_mapping is not None:
            fw = fw * torch.tensor([1.0 + self.risk_mapping.get(int(t), 0.5) for t in targets],
                                   device=inputs.device)
        loss = fw * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()
