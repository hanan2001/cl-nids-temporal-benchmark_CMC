"""
Méthodes de continual learning — VERSION CORRIGÉE ET CONTRÔLÉE.

Toutes les méthodes (FineTuning, EWC, ReplayUniform, ReplayRisk, Joint)
passent par la MÊME fonction train_on_task :
  - même backbone, même perte (Class-Balanced Focal corrigée), même clipping ;
  - NOUVEL optimiseur Adam à chaque tâche, LR constant (plus de scheduler
    persistant qui faisait tomber le LR vers 1e-8 dès T5-T6) ;
  - early stopping RELATIF sur la cross-entropy de validation non pondérée,
    avec restauration du meilleur checkpoint.
Seule la stratégie anti-oubli diffère :
  - FineTuning : rien ;
  - EWC        : pénalité de Fisher (Kirkpatrick et al., 2017) ;
  - Replay*    : réservoir équilibré par classe + terme λ·CE(batch replay)
                 à CHAQUE batch (Eq. 3 du papier, réellement implémentée) ;
  - ReplayRisk : capacité et probabilité de tirage du réservoir ∝ (1+AIM),
                 ET poids de perte ∝ (1+AIM) — le risque agit vraiment ;
  - Joint      : borne haute, un seul entraînement sur l'union des trains.
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score)

from models.focal_loss import ClassBalancedFocalLoss

# Scores AIM de la Table I, indexés par NOM (aucune valeur par défaut silencieuse)
AIM_SCORES = {
    'BENIGN': 0.05, 'Bot': 0.82, 'DDoS': 0.67, 'DoS GoldenEye': 0.80,
    'DoS Hulk': 0.80, 'DoS Slowhttptest': 0.75, 'DoS slowloris': 0.75,
    'FTP-Patator': 0.58, 'Heartbleed': 0.58, 'Infiltration': 0.91,
    'PortScan': 0.58, 'SSH-Patator': 0.58, 'Web Attack - Brute Force': 0.58,
    'Web Attack - Sql Injection': 0.58, 'Web Attack - XSS': 0.58,
}


class CLConfig:
    HIDDEN = [512, 256, 128]
    DROPOUT = 0.3
    LR = 1e-4
    BATCH = 512
    MAX_EPOCHS = 80
    PATIENCE = 10
    REL_MIN_DELTA = 1e-3          # amélioration = baisse relative ≥ 0,1 %
    CB_BETA = 0.9999              # 0.999 sature vers ~3000 échantillons : Bot n'était pas repondéré
    FOCAL_GAMMA = 1.0
    CLIP_NORM = 1.0               # appliqué à TOUTES les méthodes
    REPLAY_PER_CLASS = 500        # quota de base par classe
    REPLAY_BATCH_FRAC = 0.5       # taille du batch replay = 0,5 × BATCH
    REPLAY_LAMBDA = 0.5           # λ de l'Eq. (3)
    RISK_ON_BUFFER = True         # ReplayRisk uniquement
    RISK_ON_LOSS = True           # ReplayRisk uniquement
    AIM_MODE = 'table'            # 'table' | 'uniform' | 'inverted' | 'permuted' (ablation de calibration)
    AIM_PERMUTE_SEED = 0
    EWC_LAMBDA = 1000.0
    EWC_FISHER_BATCHES = 300
    EWC_FISHER_BS = 32
    # --- Pièges de l'ancienne version, réactivables UN PAR UN (ablation) ---
    PITFALL_PERSISTENT_OPT = False   # Adam + ReduceLROnPlateau jamais réinitialisés
    PITFALL_CB_LEGACY = False        # poids CB maximaux pour les classes absentes
    PITFALL_REPLAY_CONCAT = False    # 1200 échantillons concaténés une fois par tâche
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VERBOSE = True

    @classmethod
    def as_dict(cls):
        return {k: (str(v) if k == 'DEVICE' else v) for k, v in vars(cls).items()
                if k.isupper()}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def risk_vector(class_names, cfg=None):
    missing = [c for c in class_names if c not in AIM_SCORES]
    if missing:
        raise KeyError(f"Classes sans score AIM : {missing}")
    r = np.array([AIM_SCORES[c] for c in class_names], dtype=np.float64)
    mode = getattr(cfg, 'AIM_MODE', 'table')
    att = np.array([c != 'BENIGN' for c in class_names])
    if mode == 'uniform':                 # toutes les attaques au même score
        r[att] = r[att].mean()
    elif mode == 'inverted':              # ordre de sévérité inversé
        r[att] = r[att].max() + r[att].min() - r[att]
    elif mode == 'permuted':              # scores permutés entre attaques
        rng = np.random.default_rng(getattr(cfg, 'AIM_PERMUTE_SEED', 0))
        r[att] = rng.permutation(r[att])
    elif mode != 'table':
        raise ValueError(mode)
    return r


class MLP(nn.Module):
    """78 → 512 → 256 → 128 → 15, ReLU, dropout 0,3 (à décrire dans le papier)."""
    def __init__(self, d_in, hidden, n_out, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Réservoir équilibré par classe
# --------------------------------------------------------------------------- #
class ClassBalancedReservoir:
    def __init__(self, per_class, class_risk=None, use_risk=False, seed=0):
        self.per_class = per_class
        self.class_risk = class_risk
        self.use_risk = use_risk and class_risk is not None
        self.rng = np.random.default_rng(seed)
        self.X, self.seen = {}, {}

    def cap(self, c):
        if self.use_risk:
            return int(round(self.per_class * (1.0 + self.class_risk[c])))
        return self.per_class

    def size(self):
        return sum(len(v) for v in self.X.values())

    def add(self, X, y):
        """Reservoir sampling par classe (Vitter, algorithme R), vectorisé."""
        for c in np.unique(y):
            Xc = X[y == c]
            cap, s = self.cap(int(c)), self.seen.get(int(c), 0)
            buf = self.X.get(int(c), np.empty((0, X.shape[1]), dtype=np.float32))
            k = max(0, min(cap - len(buf), len(Xc)))
            if k:
                buf = np.vstack([buf, Xc[:k]])
            rest = Xc[k:]
            if len(rest):
                j = self.rng.integers(0, s + k + np.arange(1, len(rest) + 1))
                m = j < cap
                buf[j[m]] = rest[m]
            self.X[int(c)], self.seen[int(c)] = buf, s + len(Xc)

    def epoch_batches(self, n_batches, batch_size):
        """Tire à l'avance tous les batches replay d'une époque."""
        classes = sorted(self.X)
        Xall = np.vstack([self.X[c] for c in classes]).astype(np.float32)
        yall = np.concatenate([np.full(len(self.X[c]), c) for c in classes])
        w_class = {c: (1.0 + self.class_risk[c]) if self.use_risk else 1.0 for c in classes}
        p = np.concatenate([np.full(len(self.X[c]), w_class[c] / len(self.X[c]))
                            for c in classes])
        p /= p.sum()
        idx = self.rng.choice(len(yall), size=(n_batches, batch_size), p=p)
        return torch.from_numpy(Xall[idx]), torch.from_numpy(yall[idx].astype(np.int64))


# --------------------------------------------------------------------------- #
# EWC
# --------------------------------------------------------------------------- #
class EWC:
    def __init__(self):
        self.terms = []

    def consolidate(self, model, train_ds, cfg):
        dev = cfg.DEVICE
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        loader = DataLoader(train_ds, batch_size=cfg.EWC_FISHER_BS, shuffle=True)
        model.eval(); nb = 0
        for xb, yb in loader:
            if nb >= cfg.EWC_FISHER_BATCHES:
                break
            model.zero_grad()
            F.cross_entropy(model(xb.to(dev)), yb.to(dev)).backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            nb += 1
        fisher = {n: f / max(1, nb) for n, f in fisher.items()}
        means = {n: p.detach().clone() for n, p in params.items()}
        self.terms.append((fisher, means))
        model.zero_grad()

    def penalty(self, model):
        loss = 0.0
        for fisher, means in self.terms:
            for n, p in model.named_parameters():
                loss = loss + (fisher[n] * (p - means[n]) ** 2).sum()
        return loss


# --------------------------------------------------------------------------- #
# Entraînement partagé
# --------------------------------------------------------------------------- #
class EarlyStopper:
    def __init__(self, patience, rel_delta):
        self.patience, self.rel = patience, rel_delta
        self.best, self.state, self.bad, self.best_epoch = float('inf'), None, 0, -1

    def step(self, val, model, epoch):
        if val < self.best * (1.0 - self.rel) - 1e-7:
            self.best, self.bad, self.best_epoch = val, 0, epoch
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)


@torch.no_grad()
def predict(model, ds, cfg, bs=8192):
    model.eval()
    out = [model(xb.to(cfg.DEVICE)).argmax(1).cpu()
           for xb, _ in DataLoader(ds, batch_size=bs)]
    return ds.tensors[1].numpy(), torch.cat(out).numpy()


@torch.no_grad()
def val_ce(model, ds, cfg, bs=8192):
    """CE de validation ÉQUILIBRÉE : moyenne, sur les classes présentes, de la CE
    moyenne de chaque classe. Évite qu'un val à 99 % BENIGN pilote l'early stopping."""
    model.eval(); sums, counts = {}, {}
    for xb, yb in DataLoader(ds, batch_size=bs):
        yb = yb.to(cfg.DEVICE)
        ce = F.cross_entropy(model(xb.to(cfg.DEVICE)), yb, reduction='none')
        for c in torch.unique(yb).tolist():
            m = yb == c
            sums[c] = sums.get(c, 0.0) + ce[m].sum().item()
            counts[c] = counts.get(c, 0) + int(m.sum().item())
    return float(np.mean([sums[c] / counts[c] for c in sums])) if sums else 0.0


def macro_recall(y, p):
    return float(recall_score(y, p, labels=np.unique(y), average='macro', zero_division=0))


def train_on_task(model, train_ds, val_ds, cfg, n_classes, class_risk=None,
                  buffer=None, ewc=None, tag='', state=None):
    dev = cfg.DEVICE
    y_tr = train_ds.tensors[1].numpy()
    spc = np.bincount(y_tr, minlength=n_classes)
    crit = ClassBalancedFocalLoss(spc, beta=cfg.CB_BETA, gamma=cfg.FOCAL_GAMMA,
                                  class_risk=class_risk, device=dev,
                                  legacy=cfg.PITFALL_CB_LEGACY)
    sched = None
    if cfg.PITFALL_PERSISTENT_OPT and state is not None:   # piège : état conservé entre tâches
        if 'opt' not in state:
            state['opt'] = torch.optim.Adam(model.parameters(), lr=cfg.LR)
            state['sched'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                state['opt'], mode='min', factor=0.5, patience=5)
        opt, sched = state['opt'], state['sched']
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)   # réinitialisé à chaque tâche
    loader = DataLoader(train_ds, batch_size=cfg.BATCH, shuffle=True)
    stopper = EarlyStopper(cfg.PATIENCE, cfg.REL_MIN_DELTA)
    rb = int(cfg.BATCH * cfg.REPLAY_BATCH_FRAC)
    use_replay = buffer is not None and buffer.size() > 0
    use_ewc = ewc is not None and len(ewc.terms) > 0

    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        if use_replay:
            Xr, yr = buffer.epoch_batches(len(loader), rb)
        tot, correct, n = 0.0, 0, 0
        for b, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            loss = crit(out, yb)
            if use_replay:   # Eq. (3) : L = L_cb-focal + λ · CE(replay)
                loss = loss + cfg.REPLAY_LAMBDA * F.cross_entropy(
                    model(Xr[b].to(dev)), yr[b].to(dev))
            if use_ewc:
                loss = loss + 0.5 * cfg.EWC_LAMBDA * ewc.penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.CLIP_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
            opt.step()
            tot += loss.item() * len(yb); n += len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        v = val_ce(model, val_ds, cfg)
        if sched is not None:
            sched.step(v)
        stop = stopper.step(v, model, epoch)
        if cfg.VERBOSE:
            print(f"   {tag} ep{epoch+1:>2} loss={tot/n:.4f} acc={correct/n:.4f} "
                  f"val_ce={v:.4f} lr={opt.param_groups[0]['lr']:.1e}")
        if stop:
            break
    stopper.restore(model)

    yv, pv = predict(model, val_ds, cfg)
    yt, pt = predict(model, train_ds, cfg)
    info = {'epochs': epoch + 1, 'best_epoch': stopper.best_epoch + 1,
            'train_macro_recall': macro_recall(yt, pt),
            'val_macro_recall': macro_recall(yv, pv) if len(yv) else None}
    if info['train_macro_recall'] < 0.5:
        print(f"   ⚠️ {tag} : tâche NON APPRISE (rappel macro train = "
              f"{info['train_macro_recall']:.3f})")
    return info


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def final_metrics(y, p, class_names):
    b = class_names.index('BENIGN')
    present = np.unique(y)
    per_class = {class_names[c]: {'recall': float(np.mean(p[y == c] == c)),
                                  'support': int(np.sum(y == c))} for c in present}
    att, ben = y != b, y == b
    return {
        'accuracy': float(np.mean(y == p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, p)),
        'f1_macro': float(f1_score(y, p, labels=present, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, p, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y, p)),
        'detection_rate': float(np.mean(p[att] != b)) if att.any() else None,
        'false_positive_rate': float(np.mean(p[ben] != b)) if ben.any() else None,
        'per_class': per_class,
    }


def forgetting(R):
    """Chaudhry et al. (2018) : moyenne sur i<T de max_{t<T} R[t][i] − R[T][i]."""
    T = len(R)
    if T < 2:
        return None
    vals = [max(R[t][i] for t in range(i, T - 1)) - R[T - 1][i] for i in range(T - 1)]
    return float(np.mean(vals))


def evaluate_all(model, tasks, cfg, class_names):
    ys, ps = zip(*[predict(model, t['test'], cfg) for t in tasks])
    return final_metrics(np.concatenate(ys), np.concatenate(ps), class_names)


# --------------------------------------------------------------------------- #
# Méthodes
# --------------------------------------------------------------------------- #
SEQUENTIAL = ('FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk')


def run_method(method, tasks, class_names, cfg, seed):
    set_seed(seed)
    n_classes, d_in = len(class_names), tasks[0]['train'].tensors[0].shape[1]
    risk = risk_vector(class_names, cfg)

    if method == 'Majority':
        b = class_names.index('BENIGN')
        y = np.concatenate([t['test'].tensors[1].numpy() for t in tasks])
        return {'method': method, 'seed': seed,
                'metrics': final_metrics(y, np.full_like(y, b), class_names)}

    model = MLP(d_in, cfg.HIDDEN, n_classes, cfg.DROPOUT).to(cfg.DEVICE)

    if method == 'Joint':
        cat = lambda part: TensorDataset(*[torch.cat([t[part].tensors[i] for t in tasks])
                                           for i in (0, 1)])
        info = train_on_task(model, cat('train'), cat('val'), cfg, n_classes, tag='[Joint]')
        return {'method': method, 'seed': seed, 'task_info': [info],
                'metrics': evaluate_all(model, tasks, cfg, class_names),
                'forgetting_macro_recall': None, 'forgetting_accuracy': None}

    if method not in SEQUENTIAL:
        raise ValueError(method)
    buffer = (ClassBalancedReservoir(cfg.REPLAY_PER_CLASS, risk,
                                     use_risk=(method == 'ReplayRisk' and cfg.RISK_ON_BUFFER),
                                     seed=seed)
              if method.startswith('Replay') else None)
    ewc = EWC() if method == 'EWC' else None
    loss_risk = risk if (method == 'ReplayRisk' and cfg.RISK_ON_LOSS) else None

    R_rec, R_acc, infos, state = [], [], [], {}
    for t, task in enumerate(tasks):
        train_ds, buf_arg = task['train'], buffer
        if cfg.PITFALL_REPLAY_CONCAT and buffer is not None:   # piège : replay à ~0,7 %
            buf_arg = None
            if buffer.size() > 0:
                n = min(int(len(train_ds) * cfg.REPLAY_BATCH_FRAC), 1200)
                Xr, yr = buffer.epoch_batches(1, n)
                train_ds = TensorDataset(torch.cat([train_ds.tensors[0], Xr[0]]),
                                         torch.cat([train_ds.tensors[1], yr[0]]))
        info = train_on_task(model, train_ds, task['val'], cfg, n_classes,
                             class_risk=loss_risk, buffer=buf_arg, ewc=ewc,
                             tag=f"[{method} s{seed} T{t+1}]", state=state)
        infos.append(info)
        if ewc is not None:
            ewc.consolidate(model, task['train'], cfg)
        if buffer is not None:   # train ORIGINAL seulement, après l'entraînement
            buffer.add(task['train'].tensors[0].numpy(), task['train'].tensors[1].numpy())
        row_r, row_a = [], []
        for i in range(t + 1):
            y, p = predict(model, tasks[i]['test'], cfg)
            row_r.append(macro_recall(y, p)); row_a.append(float(np.mean(y == p)))
        R_rec.append(row_r); R_acc.append(row_a)
        print(f"   [{method} s{seed}] après T{t+1} : rappel macro par tâche = "
              f"{[round(r, 3) for r in row_r]}")

    return {'method': method, 'seed': seed, 'task_info': infos,
            'metrics': evaluate_all(model, tasks, cfg, class_names),
            'R_macro_recall': R_rec, 'R_accuracy': R_acc,
            'forgetting_macro_recall': forgetting(R_rec),
            'forgetting_accuracy': forgetting(R_acc),
            'avg_final_macro_recall': float(np.mean(R_rec[-1])),
            'buffer_sizes': ({class_names[c]: len(v) for c, v in buffer.X.items()}
                             if buffer is not None else None)}
