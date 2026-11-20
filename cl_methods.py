"""
Continual-learning methods and shared training loop.

All methods (FineTuning, EWC, ReplayUniform, ReplayRisk, Joint) go through the
SAME train_on_task function: same backbone, same class-balanced focal loss, same
gradient clipping, a new Adam optimiser with constant learning rate at every
task, early stopping with restoration of the best checkpoint. Only the
anti-forgetting mechanism differs:
  - FineTuning    : none;
  - EWC           : diagonal Fisher penalty (Kirkpatrick et al., 2017);
  - Replay*       : class-balanced reservoir + lambda * CE(replay batch) at
                    every optimisation step (Eq. 3 of the paper);
  - ReplayRisk    : the AIM score of each class can act on (i) the reservoir
                    capacity, (ii) the replay sampling probability and
                    (iii) the current-task loss weight; each mechanism has its
                    own switch (RISK_ON_CAPACITY / RISK_ON_SAMPLING /
                    RISK_ON_LOSS) for the ablation of the revision;
  - Joint         : offline reference, one training run on the union of all
                    training partitions.

Defaults reproduce the configuration of the supplementary runs (Table 7/A1).
The main experiment uses --set CB_BETA=0.999 ES_CRITERION=plain
(see README; checked bit-for-bit against results/main.json).
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, recall_score)

from models.focal_loss import ClassBalancedFocalLoss
from fastload import FastLoader

# AIM scores of Table 1, indexed by class NAME (no silent default).
# CSE-CIC-IDS2018 classes are mapped to the same ATT&CK categories.
AIM_SCORES = {
    'BENIGN': 0.05, 'Bot': 0.82, 'DDoS': 0.67, 'DoS GoldenEye': 0.80,
    'DoS Hulk': 0.80, 'DoS Slowhttptest': 0.75, 'DoS slowloris': 0.75,
    'FTP-Patator': 0.58, 'Heartbleed': 0.58, 'Infiltration': 0.91,
    'PortScan': 0.58, 'SSH-Patator': 0.58, 'Web Attack - Brute Force': 0.58,
    'Web Attack - Sql Injection': 0.58, 'Web Attack - XSS': 0.58,
    # CSE-CIC-IDS2018
    'Benign': 0.05, 'FTP-BruteForce': 0.58, 'SSH-Bruteforce': 0.58,
    'DoS attacks-GoldenEye': 0.80, 'DoS attacks-Hulk': 0.80,
    'DoS attacks-SlowHTTPTest': 0.75, 'DoS attacks-Slowloris': 0.75,
    'DDoS attacks-LOIC-HTTP': 0.67, 'DDOS attack-LOIC-UDP': 0.67,
    'DDOS attack-HOIC': 0.67, 'Brute Force -Web': 0.58, 'Brute Force -XSS': 0.58,
    'SQL Injection': 0.58, 'Infilteration': 0.91,
}
BENIGN_NAMES = ('BENIGN', 'Benign')


def benign_index(class_names):
    return next(class_names.index(b) for b in BENIGN_NAMES if b in class_names)


class CLConfig:
    HIDDEN = [512, 256, 128]
    DROPOUT = 0.3
    LR = 1e-4
    BATCH = 512
    MAX_EPOCHS = 80
    PATIENCE = 10
    REL_MIN_DELTA = 1e-3          # improvement = relative decrease >= 0.1 %
    CB_BETA = 0.9999              # main experiment: 0.999
    FOCAL_GAMMA = 1.0
    CLIP_NORM = 1.0               # all methods
    ES_CRITERION = 'class_avg'    # 'class_avg' (supplementary) | 'plain' (main experiment)
    SELECT_CRITERION = 'current'  # 'current' | 'retention' (revision: retention-aware selection)
    VAL_MEMORY_PER_CLASS = 100    # validation memory used only by SELECT_CRITERION='retention'
    REPLAY_PER_CLASS = 500        # base quota per class
    REPLAY_BATCH_FRAC = 0.5       # replay batch = 0.5 x BATCH
    REPLAY_LAMBDA = 0.5           # lambda of Eq. (3)
    RISK_ON_CAPACITY = True       # ReplayRisk: capacity of class c = quota x (1 + AIM_c)
    RISK_ON_SAMPLING = True       # ReplayRisk: replay sampling probability of class c ~ (1 + AIM_c)
    RISK_ON_LOSS = True           # ReplayRisk: current-task loss weight x (1 + AIM_c)
    RISK_BUDGET_MATCHED = False   # ReplayRisk: rescale capacities so that their sum equals the uniform budget
    AIM_MODE = 'table'            # 'table' | 'uniform' | 'inverted' | 'permuted'
    AIM_PERMUTE_SEED = 0
    EWC_LAMBDA = 1000.0
    EWC_FISHER_BATCHES = 300
    EWC_FISHER_BS = 32
    EWC_FISHER_MODE = 'empirical_batch'   # 'empirical_batch' | 'empirical_example' | 'sampled_example'
    EWC_FISHER_BALANCED = False           # class-balanced sampling of the Fisher examples
    EWC_DIAG = False                      # record Fisher/drift diagnostics (EWC and FineTuning)
    FPR_TARGET = 0.001            # operating point of the fixed-FPR severity outcome
    # --- implementation pitfalls, each re-activated alone (Table 7) ---
    PITFALL_PERSISTENT_OPT = False   # P1: Adam AND ReduceLROnPlateau kept across tasks
    PITFALL_PERSISTENT_ADAM = False  # P1a: Adam state kept across tasks, no scheduler
    PITFALL_PERSISTENT_SCHED = False # P1b: fresh Adam, but plateau scheduler + LR carried over
    PITFALL_CB_LEGACY = False        # P2: maximal CB weights for absent classes
    PITFALL_REPLAY_CONCAT = False    # P3: <=1200 replay flows concatenated once per task
    FAST_LOADER = True               # identical batches and RNG stream, much faster
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VERBOSE = True

    @classmethod
    def as_dict(cls):
        return {k: (str(v) if k == 'DEVICE' else v) for k, v in vars(cls).items()
                if k.isupper() and k not in ('VERBOSE', 'FAST_LOADER')}


def make_loader(ds, cfg, batch_size, shuffle=False):
    if getattr(cfg, 'FAST_LOADER', True):
        return FastLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def aim_table(class_names):
    missing = [c for c in class_names if c not in AIM_SCORES]
    if missing:
        raise KeyError(f"Classes without AIM score: {missing}")
    return np.array([AIM_SCORES[c] for c in class_names], dtype=np.float64)


def risk_vector(class_names, cfg=None):
    r = aim_table(class_names)
    mode = getattr(cfg, 'AIM_MODE', 'table')
    att = np.array([c not in BENIGN_NAMES for c in class_names])
    if mode == 'uniform':                 # every attack gets the mean attack score
        r[att] = r[att].mean()
    elif mode == 'inverted':              # severity order inverted
        r[att] = r[att].max() + r[att].min() - r[att]
    elif mode == 'permuted':              # scores permuted between attacks
        rng = np.random.default_rng(getattr(cfg, 'AIM_PERMUTE_SEED', 0))
        r[att] = rng.permutation(r[att])
    elif mode != 'table':
        raise ValueError(mode)
    return r


class MLP(nn.Module):
    """78 -> 512 -> 256 -> 128 -> 15, ReLU, dropout 0.3."""
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


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Class-balanced reservoir
# --------------------------------------------------------------------------- #
class ClassBalancedReservoir:
    """Per-class reservoir sampling. `cap_weight` (capacity) and `sample_weight`
    (replay sampling) are per-class multipliers, or None for uniform."""
    def __init__(self, per_class, cap_weight=None, sample_weight=None, seed=0):
        self.per_class = per_class
        self.cap_weight = cap_weight
        self.sample_weight = sample_weight
        self.rng = np.random.default_rng(seed)
        self.X, self.seen = {}, {}

    def cap(self, c):
        if self.cap_weight is not None:
            return int(round(self.per_class * self.cap_weight[c]))
        return self.per_class

    def size(self):
        return sum(len(v) for v in self.X.values())

    def nbytes(self):
        """Stored features (float32) + labels (int64)."""
        return int(sum(v.nbytes for v in self.X.values()) + 8 * self.size())

    def add(self, X, y):
        """Reservoir sampling per class (Vitter, algorithm R), vectorised."""
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
        """Draw all replay batches of one epoch in advance."""
        classes = sorted(self.X)
        Xall = np.vstack([self.X[c] for c in classes]).astype(np.float32)
        yall = np.concatenate([np.full(len(self.X[c]), c) for c in classes])
        w_class = {c: (self.sample_weight[c] if self.sample_weight is not None else 1.0)
                   for c in classes}
        p = np.concatenate([np.full(len(self.X[c]), w_class[c] / len(self.X[c]))
                            for c in classes])
        p /= p.sum()
        idx = self.rng.choice(len(yall), size=(n_batches, batch_size), p=p)
        return torch.from_numpy(Xall[idx]), torch.from_numpy(yall[idx].astype(np.int64))


def build_buffer(method, risk, n_classes, cfg, seed):
    if not method.startswith('Replay'):
        return None
    cap_w = samp_w = None
    if method == 'ReplayRisk':
        w = 1.0 + risk
        if cfg.RISK_ON_CAPACITY:
            cap_w = w.copy()
            if cfg.RISK_BUDGET_MATCHED:        # same nominal total memory as uniform replay
                cap_w = cap_w * n_classes / cap_w.sum()
        if cfg.RISK_ON_SAMPLING:
            samp_w = w.copy()
    return ClassBalancedReservoir(cfg.REPLAY_PER_CLASS, cap_w, samp_w, seed=seed)


# --------------------------------------------------------------------------- #
# EWC
# --------------------------------------------------------------------------- #
def _linear_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def per_example_sq_grads(model, X, y, cfg, sampled):
    """Sum over examples of the squared per-example gradients of the CE loss.
    Exact for a network whose parameters are all in nn.Linear layers:
    for W, sum_n (delta_n x_n^T)^2 = (delta^2)^T (x^2); for b, sum_n delta_n^2."""
    acts, grads = {}, {}
    hooks = []
    for m in _linear_layers(model):
        hooks.append(m.register_forward_hook(lambda mod, i, o: acts.__setitem__(mod, i[0].detach())))
        hooks.append(m.register_full_backward_hook(
            lambda mod, gi, go: grads.__setitem__(mod, go[0].detach())))
    out = model(X)
    if sampled:   # true Fisher: labels drawn from the model's predictive distribution
        with torch.no_grad():
            y = torch.multinomial(F.softmax(out, 1), 1).squeeze(1)
    model.zero_grad()
    F.cross_entropy(out, y, reduction='sum').backward()
    for h in hooks:
        h.remove()
    res = {}
    for m in _linear_layers(model):
        a, d = acts[m], grads[m]
        res[m.weight] = (d ** 2).t() @ (a ** 2)
        res[m.bias] = (d ** 2).sum(0)
    model.zero_grad()
    return res


class EWC:
    """Diagonal Fisher of each past task + parameter anchor (one term per task).

    EWC_FISHER_MODE:
      'empirical_batch'   (main experiment) observed labels; gradient of the MEAN
                          CE of a mini-batch of EWC_FISHER_BS flows, squared, then
                          averaged over EWC_FISHER_BATCHES mini-batches. This is a
                          batch-level estimate: for uncorrelated per-example
                          gradients it is about 1/EWC_FISHER_BS of the per-example
                          Fisher.
      'empirical_example' observed labels, per-example squared gradients averaged
                          over EWC_FISHER_BATCHES x EWC_FISHER_BS flows.
      'sampled_example'   labels sampled from the model (true Fisher), per example.
    Dropout is disabled (model.eval()) during the estimation. Examples are drawn
    uniformly from the task's training partition (shuffled), or class-balanced if
    EWC_FISHER_BALANCED; a task with fewer flows uses all of them.
    """
    def __init__(self):
        self.terms = []

    def fisher(self, model, train_ds, cfg, seed=0):
        dev = cfg.DEVICE
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        model.eval()
        mode = cfg.EWC_FISHER_MODE
        if mode not in ('empirical_batch', 'empirical_example', 'sampled_example'):
            raise ValueError(mode)
        if mode == 'empirical_batch' and cfg.EWC_FISHER_BALANCED:
            raise ValueError("class-balanced sampling requires a per-example Fisher mode")
        if mode == 'empirical_batch':
            loader = make_loader(train_ds, cfg, cfg.EWC_FISHER_BS, shuffle=True)
            nb = 0
            for xb, yb in loader:
                if nb >= cfg.EWC_FISHER_BATCHES:
                    break
                model.zero_grad()
                F.cross_entropy(model(xb.to(dev)), yb.to(dev)).backward()
                for n, p in params.items():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                nb += 1
            model.zero_grad()
            return {n: f / max(1, nb) for n, f in fisher.items()}
        # per-example estimators (revision), private RNG
        X, y = train_ds.tensors
        N = min(cfg.EWC_FISHER_BATCHES * cfg.EWC_FISHER_BS, len(y))
        rng = np.random.default_rng(seed)
        if cfg.EWC_FISHER_BALANCED:
            yn = y.numpy(); cls = np.unique(yn)
            idx = np.concatenate([rng.choice(np.where(yn == c)[0], size=-(-N // len(cls)),
                                             replace=(yn == c).sum() < -(-N // len(cls)))
                                  for c in cls])
        else:
            idx = rng.permutation(len(y))[:N]
        name_of = {p: n for n, p in params.items()}
        state = torch.get_rng_state(); torch.manual_seed(int(rng.integers(2**31)))
        for i in range(0, len(idx), 1024):
            b = torch.from_numpy(idx[i:i + 1024])
            sq = per_example_sq_grads(model, X[b].to(dev), y[b].to(dev), cfg,
                                      sampled=(mode == 'sampled_example'))
            for p, v in sq.items():
                fisher[name_of[p]] += v
        torch.set_rng_state(state)          # global random stream left untouched
        return {n: f / len(idx) for n, f in fisher.items()}

    def consolidate(self, model, train_ds, cfg, seed=0):
        f = self.fisher(model, train_ds, cfg, seed)
        means = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.terms.append((f, means))
        model.zero_grad()

    def nbytes(self):
        return int(sum(sum(v.numel() * v.element_size() for v in d.values())
                       for term in self.terms for d in term))

    def penalty(self, model):
        loss = 0.0
        for fisher, means in self.terms:
            for n, p in model.named_parameters():
                loss = loss + (fisher[n] * (p - means[n]) ** 2).sum()
        return loss


def drift_diagnostics(model, fisher, anchor):
    """Squared parameter drift since the anchor, for the parameters with the 1 %
    largest Fisher values and for the 50 % smallest, plus overall statistics."""
    f = torch.cat([fisher[n].flatten() for n in fisher])
    d = torch.cat([((p.detach() - anchor[n]) ** 2).flatten()
                   for n, p in model.named_parameters()])
    order = torch.argsort(f, descending=True)
    k1, k50 = max(1, len(f) // 100), len(f) // 2
    return {'drift_top1pct_fisher': float(d[order[:k1]].mean()),
            'drift_bottom50pct_fisher': float(d[order[-k50:]].mean()),
            'drift_all': float(d.mean()),
            'fisher_weighted_drift': float((f * d).sum()),
            'fisher_sum': float(f.sum()), 'fisher_max': float(f.max()),
            'fisher_top1pct_mass': float(f[order[:k1]].sum() / f.sum()) if f.sum() > 0 else None}


# --------------------------------------------------------------------------- #
# Shared training
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
           for xb, _ in make_loader(ds, cfg, bs)]
    return ds.tensors[1].numpy(), torch.cat(out).numpy()


@torch.no_grad()
def predict_proba_norng(model, X, cfg, bs=8192):
    """Softmax outputs; does not touch any random generator."""
    model.eval()
    return torch.cat([F.softmax(model(X[i:i + bs].to(cfg.DEVICE)), 1).cpu()
                      for i in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def val_ce(model, ds, cfg, bs=8192, criterion=None):
    """Validation cross-entropy used for early stopping and checkpoint selection.
    'class_avg': mean over the classes present of the per-class mean CE;
    'plain'    : mean CE over all validation flows (main experiment)."""
    criterion = criterion or cfg.ES_CRITERION
    model.eval(); sums, counts = {}, {}
    tot, n = 0.0, 0
    for xb, yb in make_loader(ds, cfg, bs):
        yb = yb.to(cfg.DEVICE)
        ce = F.cross_entropy(model(xb.to(cfg.DEVICE)), yb, reduction='none')
        if criterion == 'plain':
            tot += ce.sum().item(); n += len(yb)
            continue
        for c in torch.unique(yb).tolist():
            m = yb == c
            sums[c] = sums.get(c, 0.0) + ce[m].sum().item()
            counts[c] = counts.get(c, 0) + int(m.sum().item())
    if criterion == 'plain':
        return tot / max(1, n)
    return float(np.mean([sums[c] / counts[c] for c in sums])) if sums else 0.0


def macro_recall(y, p):
    return float(recall_score(y, p, labels=np.unique(y), average='macro', zero_division=0))


def train_on_task(model, train_ds, val_ds, cfg, n_classes, class_risk=None,
                  buffer=None, ewc=None, tag='', state=None, val_memory=None):
    dev = cfg.DEVICE
    t0 = time.perf_counter()
    y_tr = train_ds.tensors[1].numpy()
    spc = np.bincount(y_tr, minlength=n_classes)
    crit = ClassBalancedFocalLoss(spc, beta=cfg.CB_BETA, gamma=cfg.FOCAL_GAMMA,
                                  class_risk=class_risk, device=dev,
                                  legacy=cfg.PITFALL_CB_LEGACY)
    sched = None
    state = state if state is not None else {}
    if cfg.PITFALL_PERSISTENT_OPT:            # P1: Adam + scheduler kept across tasks
        if 'opt' not in state:
            state['opt'] = torch.optim.Adam(model.parameters(), lr=cfg.LR)
            state['sched'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                state['opt'], mode='min', factor=0.5, patience=5)
        opt, sched = state['opt'], state['sched']
    elif cfg.PITFALL_PERSISTENT_ADAM:         # P1a: Adam moments kept, constant LR
        if 'opt' not in state:
            state['opt'] = torch.optim.Adam(model.parameters(), lr=cfg.LR)
        opt = state['opt']
    elif cfg.PITFALL_PERSISTENT_SCHED:        # P1b: fresh Adam, LR schedule carried over
        opt = torch.optim.Adam(model.parameters(), lr=state.get('lr', cfg.LR))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        if 'sched_state' in state:
            sched.load_state_dict(state['sched_state'])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)   # reset at every task
    loader = make_loader(train_ds, cfg, cfg.BATCH, shuffle=True)
    stopper = EarlyStopper(cfg.PATIENCE, cfg.REL_MIN_DELTA)
    rb = int(cfg.BATCH * cfg.REPLAY_BATCH_FRAC)
    use_replay = buffer is not None and buffer.size() > 0
    use_ewc = ewc is not None and len(ewc.terms) > 0
    use_mem = cfg.SELECT_CRITERION == 'retention' and val_memory is not None
    n_steps, ewc_trace = 0, []

    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        if use_replay:
            Xr, yr = buffer.epoch_batches(len(loader), rb)
        tot, correct, n = 0.0, 0, 0
        pen_sum, ce_sum = 0.0, 0.0
        for b, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            loss = crit(out, yb)
            if use_replay:   # Eq. (3): L = L_cb-focal + lambda * CE(replay)
                loss = loss + cfg.REPLAY_LAMBDA * F.cross_entropy(
                    model(Xr[b].to(dev)), yr[b].to(dev))
            if use_ewc:
                pen = 0.5 * cfg.EWC_LAMBDA * ewc.penalty(model)
                if cfg.EWC_DIAG:
                    pen_sum += float(pen); ce_sum += float(loss)
                loss = loss + pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.CLIP_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
            opt.step()
            n_steps += 1
            tot += loss.item() * len(yb); n += len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        if use_ewc and cfg.EWC_DIAG:
            ewc_trace.append(pen_sum / max(ce_sum, 1e-12))
        v = val_ce(model, val_ds, cfg)
        if use_mem:   # retention-aware: equal weight to current task and past-task memory
            v = 0.5 * v + 0.5 * val_ce(model, val_memory, cfg, criterion='class_avg')
        if sched is not None:
            sched.step(v)
        stop = stopper.step(v, model, epoch)
        if cfg.VERBOSE:
            print(f"   {tag} ep{epoch+1:>2} loss={tot/n:.4f} acc={correct/n:.4f} "
                  f"val={v:.4f} lr={opt.param_groups[0]['lr']:.1e}")
        if stop:
            break
    stopper.restore(model)
    if cfg.PITFALL_PERSISTENT_SCHED:
        state['lr'] = opt.param_groups[0]['lr']
        state['sched_state'] = sched.state_dict()
    t_train = time.perf_counter() - t0

    yv, pv = predict(model, val_ds, cfg)
    yt, pt = predict(model, train_ds, cfg)
    info = {'epochs': epoch + 1, 'best_epoch': stopper.best_epoch + 1,
            'train_macro_recall': macro_recall(yt, pt),
            'val_macro_recall': macro_recall(yv, pv) if len(yv) else None,
            'train_seconds': t_train, 'n_updates': n_steps,
            'seconds_per_update': t_train / max(1, n_steps)}
    if ewc_trace:
        info['ewc_penalty_over_loss_by_epoch'] = ewc_trace
    if info['train_macro_recall'] < 0.5:
        print(f"   WARNING {tag}: task NOT LEARNED (train macro-recall = "
              f"{info['train_macro_recall']:.3f})")
    return info


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def final_metrics(y, p, class_names, proba=None, threshold=None):
    b = benign_index(class_names)
    present = np.unique(y)
    per_class = {class_names[c]: {'recall': float(np.mean(p[y == c] == c)),
                                  'detected': float(np.mean(p[y == c] != b)),
                                  'support': int(np.sum(y == c))} for c in present}
    att, ben = y != b, y == b
    m = {
        'accuracy': float(np.mean(y == p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, p)),
        'f1_macro': float(f1_score(y, p, labels=present, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, p, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y, p)),
        'detection_rate': float(np.mean(p[att] != b)) if att.any() else None,
        'false_positive_rate': float(np.mean(p[ben] != b)) if ben.any() else None,
        'false_positive_flows': int(np.sum(p[ben] != b)),
        'per_class': per_class,
    }
    # Severity-sensitive outcomes (revision). Weights = AIM table scores, whatever AIM_MODE.
    try:
        aim = aim_table(class_names)
    except KeyError:
        aim = None
    atk = [c for c in present if c != b]
    if aim is not None and atk:
        w = np.array([aim[c] for c in atk])
        det = np.array([np.mean(p[y == c] != b) for c in atk])
        n_c = np.array([np.sum(y == c) for c in atk])
        miss = np.array([np.sum((y == c) & (p == b)) for c in atk])
        m['sw_detection_rate'] = float((w * det).sum() / w.sum())          # class-macro, AIM-weighted
        m['sw_missed_attack_rate'] = float((w * miss).sum() / (w * n_c).sum())  # flow-level, AIM-weighted
        m['missed_attack_flows'] = int(miss.sum())
        if proba is not None and threshold is not None:
            s = 1.0 - proba[:, b]
            alarm = s > threshold
            det_t = np.array([np.mean(alarm[y == c]) for c in atk])
            m['at_fixed_fpr'] = {
                'threshold': float(threshold),
                'test_fpr': float(np.mean(alarm[ben])) if ben.any() else None,
                'detection_rate': float(np.mean(alarm[att])),
                'sw_detection_rate': float((w * det_t).sum() / w.sum()),
                'macro_detection_rate': float(det_t.mean()),
            }
    return m


def forgetting(R):
    """Chaudhry et al. (2018): mean over i<T of max_{t<T} R[t][i] - R[T][i]."""
    T = len(R)
    if T < 2:
        return None
    vals = [max(R[t][i] for t in range(i, T - 1)) - R[T - 1][i] for i in range(T - 1)]
    return float(np.mean(vals))


def evaluate_all(model, tasks, cfg, class_names):
    ys, ps = zip(*[predict(model, t['test'], cfg) for t in tasks])
    y, p = np.concatenate(ys), np.concatenate(ps)
    # operating point fixed on the aggregated VALIDATION benign flows (no RNG used)
    b = benign_index(class_names)
    Xv = torch.cat([t['val'].tensors[0] for t in tasks])
    yv = torch.cat([t['val'].tensors[1] for t in tasks]).numpy()
    sv = 1.0 - predict_proba_norng(model, Xv[yv == b], cfg)[:, b]
    thr = float(np.quantile(sv, 1.0 - cfg.FPR_TARGET)) if len(sv) else None
    Xt = torch.cat([t['test'].tensors[0] for t in tasks])
    proba = predict_proba_norng(model, Xt, cfg)
    return final_metrics(y, p, class_names, proba=proba, threshold=thr)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
SEQUENTIAL = ('FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk')


class ValMemory:
    """Class-balanced memory of past VALIDATION flows, used only for checkpoint
    selection when SELECT_CRITERION='retention' (never for training)."""
    def __init__(self, per_class, seed):
        self.res = ClassBalancedReservoir(per_class, seed=seed + 7919)

    def add(self, ds):
        self.res.add(ds.tensors[0].numpy(), ds.tensors[1].numpy())

    def dataset(self):
        if self.res.size() == 0:
            return None
        cl = sorted(self.res.X)
        X = np.vstack([self.res.X[c] for c in cl]).astype(np.float32)
        y = np.concatenate([np.full(len(self.res.X[c]), c) for c in cl]).astype(np.int64)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def run_method(method, tasks, class_names, cfg, seed):
    set_seed(seed)
    t_start = time.perf_counter()
    n_classes, d_in = len(class_names), tasks[0]['train'].tensors[0].shape[1]
    risk = risk_vector(class_names, cfg)

    if method == 'Majority':
        b = benign_index(class_names)
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
                'forgetting_macro_recall': None, 'forgetting_accuracy': None,
                'resources': {'n_params': n_params(model), 'memory_bytes': 0,
                              'total_seconds': time.perf_counter() - t_start}}

    if method not in SEQUENTIAL:
        raise ValueError(method)
    buffer = build_buffer(method, risk, n_classes, cfg, seed)
    ewc = EWC() if method == 'EWC' else None
    diag_ewc = EWC() if (cfg.EWC_DIAG and method == 'FineTuning') else None
    loss_risk = risk if (method == 'ReplayRisk' and cfg.RISK_ON_LOSS) else None
    val_mem = ValMemory(cfg.VAL_MEMORY_PER_CLASS, seed) if cfg.SELECT_CRITERION == 'retention' else None

    R_rec, R_acc, infos, state, diags = [], [], [], {}, []
    for t, task in enumerate(tasks):
        train_ds, buf_arg = task['train'], buffer
        if cfg.PITFALL_REPLAY_CONCAT and buffer is not None:   # P3
            buf_arg = None
            if buffer.size() > 0:
                n = min(int(len(train_ds) * cfg.REPLAY_BATCH_FRAC), 1200)
                Xr, yr = buffer.epoch_batches(1, n)
                train_ds = TensorDataset(torch.cat([train_ds.tensors[0], Xr[0]]),
                                         torch.cat([train_ds.tensors[1], yr[0]]))
        info = train_on_task(model, train_ds, task['val'], cfg, n_classes,
                             class_risk=loss_risk, buffer=buf_arg, ewc=ewc,
                             tag=f"[{method} s{seed} T{t+1}]", state=state,
                             val_memory=val_mem.dataset() if val_mem is not None else None)
        # EWC diagnostics: drift w.r.t. the previous anchor, ranked by its Fisher
        tracker = ewc if ewc is not None else diag_ewc
        if cfg.EWC_DIAG and tracker is not None and tracker.terms:
            fisher, anchor = tracker.terms[-1]
            info['diag'] = drift_diagnostics(model, fisher, anchor)
        infos.append(info)
        if ewc is not None:
            ewc.consolidate(model, task['train'], cfg, seed=seed * 100 + t)
        if diag_ewc is not None:
            diag_ewc.terms = []     # keep only the latest Fisher for the drift diagnostic
            diag_ewc.consolidate(model, task['train'], cfg, seed=seed * 100 + t)
        if buffer is not None:   # ORIGINAL training data only, after training
            buffer.add(task['train'].tensors[0].numpy(), task['train'].tensors[1].numpy())
        if val_mem is not None:
            val_mem.add(task['val'])
        row_r, row_a = [], []
        for i in range(t + 1):
            y, p = predict(model, tasks[i]['test'], cfg)
            row_r.append(macro_recall(y, p)); row_a.append(float(np.mean(y == p)))
        R_rec.append(row_r); R_acc.append(row_a)
        print(f"   [{method} s{seed}] after T{t+1}: per-task macro-recall = "
              f"{[round(r, 3) for r in row_r]}")

    mem = (buffer.nbytes() if buffer is not None else 0) + (ewc.nbytes() if ewc is not None else 0)
    return {'method': method, 'seed': seed, 'task_info': infos,
            'metrics': evaluate_all(model, tasks, cfg, class_names),
            'R_macro_recall': R_rec, 'R_accuracy': R_acc,
            'forgetting_macro_recall': forgetting(R_rec),
            'forgetting_accuracy': forgetting(R_acc),
            'avg_final_macro_recall': float(np.mean(R_rec[-1])),
            'buffer_sizes': ({class_names[c]: len(v) for c, v in buffer.X.items()}
                             if buffer is not None else None),
            'resources': {'n_params': n_params(model), 'memory_bytes': mem,
                          'buffer_flows': buffer.size() if buffer is not None else 0,
                          'total_seconds': time.perf_counter() - t_start,
                          'total_updates': sum(i['n_updates'] for i in infos)}}
