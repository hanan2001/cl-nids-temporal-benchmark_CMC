"""
Apprentissage continu — VERSION PAPER-READY (split externe train/val/test).

CORRECTIONS vs version précédente:
- train_task reçoit train_dataset ET val_dataset DÉJÀ splittés (plus de
  re-split interne). Le split 60/20/20 est fait une seule fois dans
  generator.py, avec seed variable.
- L'évaluation "toutes tâches vues" se fait sur les TEST sets tenus à l'écart
  (all_test_tasks), jamais sur les données d'entraînement/validation.
- SMOTE et replay restent appliqués uniquement sur X_train.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import Trainer
from data.oversampling import AdaptiveOversampler

import copy


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

    def get_stats(self):
        return {'counter': self.counter, 'patience': self.patience,
                'best_loss': self.best_loss, 'best_epoch': self.best_epoch,
                'early_stop': self.early_stop}


class ReplayBuffer:
    """
    Réservoir ÉQUILIBRÉ PAR CLASSE (reservoir sampling par classe).
    Chaque classe vue conserve au plus `per_class` échantillons, tirés
    uniformément dans son flux via reservoir sampling. L'échantillonnage
    pour le replay tire équitablement entre classes -> BENIGN ne domine plus.
    Le risque module la TAILLE du réservoir par classe (classes à haut risque
    gardées plus nombreuses) et la probabilité de tirage.
    """
    def __init__(self, max_size=50000, risk_mapping=None, replay_ratio=0.5,
                 max_per_class=500):
        self.risk_mapping = risk_mapping or {}
        self.replay_ratio = replay_ratio
        self.per_class = max_per_class
        self.buffers = {}   # classe -> liste d'échantillons (X)
        self.seen = {}      # classe -> nb total vu (pour reservoir sampling)

    def _cap_for(self, cls):
        """Taille du réservoir pour une classe (risque augmente la capacité)."""
        base = self.per_class
        if self.risk_mapping:
            risk = self.risk_mapping.get(cls, 0.5)
            return int(base * (1.0 + risk))   # risque 0.7 -> +70% de places
        return base

    def add_samples(self, X, y):
        if len(X) == 0:
            return
        for x_i, y_i in zip(X, y):
            y_i = int(y_i)
            cap = self._cap_for(y_i)
            self.buffers.setdefault(y_i, [])
            self.seen[y_i] = self.seen.get(y_i, 0) + 1
            buf = self.buffers[y_i]
            if len(buf) < cap:
                buf.append(np.asarray(x_i))
            else:
                # reservoir sampling : remplace avec proba décroissante
                j = np.random.randint(0, self.seen[y_i])
                if j < cap:
                    buf[j] = np.asarray(x_i)

    def sample(self, n_samples, risk_weighted=True):
        classes = [c for c, b in self.buffers.items() if len(b) > 0]
        if not classes or n_samples <= 0:
            return None, None
        # poids par classe : uniforme, ou pondéré risque
        if risk_weighted and self.risk_mapping:
            w = np.array([1.0 + self.risk_mapping.get(c, 0.5) for c in classes])
        else:
            w = np.ones(len(classes))
        w = w / w.sum()
        # nombre d'échantillons par classe (au moins 1 si la classe est tirée)
        per_class_draw = np.random.multinomial(n_samples, w)
        Xs, ys = [], []
        for c, k in zip(classes, per_class_draw):
            if k == 0:
                continue
            buf = self.buffers[c]
            idx = np.random.choice(len(buf), size=min(k, len(buf)),
                                   replace=len(buf) < k)
            for i in idx:
                Xs.append(buf[i]); ys.append(c)
        if not Xs:
            return None, None
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    def get_stats(self):
        return {'size': sum(len(b) for b in self.buffers.values()),
                'classes': {c: len(b) for c, b in self.buffers.items()}}
class ContinualLearner:
    def __init__(self, model, gpm, config):
        self.model = model
        self.gpm = gpm
        self.config = config
        self.trainer = None

        if getattr(config, 'USE_ADAPTIVE_OVERSAMPLING', True):
            self.oversampler = AdaptiveOversampler(
                threshold=getattr(config, 'RARE_CLASS_THRESHOLD', 500),
                target_samples=getattr(config, 'TARGET_SAMPLES', 300),
                k_neighbors=getattr(config, 'SMOTE_K_NEIGHBORS', 3))
        else:
            self.oversampler = None

        self.replay_buffer = None
        self.replay_buffer_size = getattr(config, 'REPLAY_BUFFER_SIZE', 3000)
        self.replay_ratio = getattr(config, 'REPLAY_RATIO', 0.10)
        self.max_per_class = getattr(config, 'MAX_PER_CLASS_REPLAY', 200)
        self.use_replay = getattr(config, 'USE_REPLAY', True)

        self.performance_history = []
        self.task_models = []
        self.epochs_per_task = []
        self.replay_stats = []
        self.accuracies_history = []

    def _get_replay_samples(self, n_samples, risk_mapping):
        if self.replay_buffer is None:
            return None, None
        if self.replay_buffer.get_stats()['size'] == 0:
            return None, None
        # H1 : le risque ne pilote le tirage du buffer que si RISK_ON_BUFFER
        rw = (getattr(self.config, 'USE_RISK_GUIDED', True)
              and getattr(self.config, 'RISK_ON_BUFFER', True))
        return self.replay_buffer.sample(n_samples, risk_weighted=rw)

    def train_task(self, train_dataset, val_dataset, task_id,
                   risk_mapping=None, all_test_tasks=None):
        """
        train_dataset / val_dataset : déjà splittés en amont (generator.py).
        all_test_tasks : liste des TEST datasets des tâches 0..task_id
                         (évaluation du forgetting, tenue à l'écart).
        """
        use_projection = getattr(self.config, 'USE_PROJECTION', True)

        print(f"\n🔄 Tâche {task_id+1} | "
              f"proj={use_projection} risk={getattr(self.config,'USE_RISK_GUIDED',True)} "
              f"replay={self.use_replay} smote={self.oversampler is not None} "
              f"focal={getattr(self.config,'USE_FOCAL_LOSS',True)}")

        X_train = train_dataset.tensors[0].numpy()
        y_train = train_dataset.tensors[1].numpy()
        import numpy as _np
        u, c = _np.unique(y_train, return_counts=True)
        print(f"   [DEBUG train] tâche {task_id+1} classes en entraînement: {dict(zip(u.tolist(), c.tolist()))}")

        if self.trainer is None:
            # H1 : le risque ne pilote la focal loss que si RISK_ON_LOSS
            risk_for_loss = (risk_mapping
                if (getattr(self.config, 'USE_RISK_GUIDED', True)
                    and getattr(self.config, 'RISK_ON_LOSS', True))
                else None)
            self.trainer = Trainer(
                self.model, self.gpm, self.config,
                use_focal_loss=getattr(self.config, 'USE_FOCAL_LOSS', True),
                focal_gamma=getattr(self.config, 'FOCAL_GAMMA', 2.0),
                risk_mapping=risk_for_loss)

        if self.use_replay and self.replay_buffer is None:
            # H1 : le risque ne module le buffer (capacité + tirage) que si RISK_ON_BUFFER
            risk_for_buffer = (risk_mapping
                if (getattr(self.config, 'USE_RISK_GUIDED', True)
                    and getattr(self.config, 'RISK_ON_BUFFER', True))
                else None)
            self.replay_buffer = ReplayBuffer(
                max_size=self.replay_buffer_size,
                risk_mapping=risk_for_buffer,
                replay_ratio=self.replay_ratio,
                max_per_class=self.max_per_class)

        # --- SMOTE sur TRAIN uniquement ---
        if self.oversampler and getattr(self.config, 'USE_ADAPTIVE_OVERSAMPLING', True):
            unique, counts = np.unique(y_train, return_counts=True)
            rares = [(c, n) for c, n in zip(unique, counts) if n < self.oversampler.threshold]
            if rares:
                print(f"   🔄 SMOTE sur {len(rares)} classes rares (train)")
                X_train, y_train = self.oversampler.balance_task(X_train, y_train)

        # --- Replay sur TRAIN uniquement ---
        if self.use_replay and self.replay_buffer is not None and len(X_train) >= 1000:
            n_replay = max(1, min(int(len(X_train) * self.replay_ratio), 1200))
            Xr, yr = self._get_replay_samples(n_replay, risk_mapping)
            if Xr is not None and len(Xr) > 0:
                print(f"   🔄 +{len(Xr)} échantillons de replay")
                X_train = np.vstack([X_train, Xr])
                y_train = np.concatenate([y_train, yr])

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        # alimenter le buffer avec les données de la tâche courante
        if self.use_replay and self.replay_buffer is not None:
            self.replay_buffer.add_samples(X_train, y_train)
            self.replay_stats.append(self.replay_buffer.get_stats())

        # --- Construction de la mémoire GPM (avant d'entraîner la nouvelle tâche) ---
        if use_projection:
            all_s, all_l = [], []
            for xb, yb in train_loader:
                all_s.append(xb)
                all_l.append(yb)
            if all_s:
                all_s = torch.cat(all_s, dim=0)
                all_l = torch.cat(all_l, dim=0)
                self.gpm.update_memory(self.model, all_s, all_l, risk_mapping)
                print(f"   ✅ Mémoire GPM: {len(self.gpm.memory)} couches")
        else:
            self.gpm.clear()

        # --- Boucle d'entraînement ---
        early = EarlyStopping(
            patience=getattr(self.config, 'EARLY_STOPPING_PATIENCE', 10),
            min_delta=getattr(self.config, 'EARLY_STOPPING_MIN_DELTA', 0.001))

        loss_hist = []
                # === Class-Balanced Loss : fréquences réelles APRÈS replay ===
        if getattr(self.config, 'USE_CLASS_BALANCED', True):
            from models.focal_loss import ClassBalancedFocalLoss
            n_classes_total = self.model.network[-1].out_features
            # fréquences du y réellement utilisé pour entraîner (avec replay/smote)
            y_used = y_train  # <-- remplace par le y final si tu le renommes après replay
            spc = np.zeros(n_classes_total)
            u_cb, c_cb = np.unique(y_used, return_counts=True)
            spc[u_cb.astype(int)] = c_cb
            self.trainer.criterion = ClassBalancedFocalLoss(
                
                spc, beta=0.99,
                gamma=getattr(self.config, 'FOCAL_GAMMA', 1.0),
                device=self.config.DEVICE)
            print(f"   ⚖️ Class-Balanced Loss activée (spc min={int(spc[spc>0].min())}, max={int(spc.max())})")
        for epoch in range(self.config.MAX_EPOCHS_PER_TASK):
            loss, acc = self.trainer.train_epoch(train_loader, use_projection=use_projection)
            loss_hist.append(loss)
            val_acc, _, _ = self.trainer.evaluate(val_loader)
            val_loss = self.trainer.compute_val_loss(val_loader, step_scheduler=True)
            print(f"  Epoch {epoch+1}: loss={loss:.4f} acc={acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if early(val_loss, epoch):
                print(f"   ⏹️ Early stop à l'époque {epoch+1} "
                      f"(best={early.best_loss:.4f} @ {early.best_epoch+1})")
                break

        self.epochs_per_task.append(len(loss_hist))
        self.task_models.append(copy.deepcopy(self.model.state_dict()))

        # --- Évaluation forgetting sur TEST sets ---
        all_task_acc = []
        if all_test_tasks is not None:
            for past_id in range(task_id + 1):
                loader = DataLoader(all_test_tasks[past_id],
                                    batch_size=self.config.BATCH_SIZE, shuffle=False)
                pa, _, _ = self.trainer.evaluate(loader)
                all_task_acc.append(pa)
            self.accuracies_history.append(all_task_acc)
        else:
            va, _, _ = self.trainer.evaluate(val_loader)
            self.accuracies_history.append([va])

        val_acc, _, _ = self.trainer.evaluate(val_loader)
        self.performance_history.append(val_acc)
        print(f"✅ Tâche {task_id+1} — val_acc={val_acc:.4f} | "
              f"test acc/tâche={[f'{a:.4f}' for a in all_task_acc]}")
        return val_acc