"""
Protocole temporel CIC-IDS2017 — VERSION CORRIGÉE.

Corrections par rapport à la version soumise :
1. Split chronologique PAR CLASSE à l'intérieur de chaque jour : pour chaque
   classe, ses flux (dans l'ordre du fichier) sont coupés en 60/20/20.
   -> chaque classe présente un jour apparaît en train, val ET test ;
   -> aucun flux futur d'une classe ne sert à apprendre ses flux passés.
   (L'ancien split par lignes plaçait les attaques web/infiltration hors du
   test et rendait les partitions de T3/T7 quasi pures.)
2. Mise à l'échelle UNIQUE et causale : log signé (sans état) puis
   StandardScaler ajusté sur le train du PREMIER jour uniquement, appliqué
   à tous les jours. Plus de scaler par jour (qui revenait à donner
   l'identité de la tâche au moment du test).
3. Normalisation des libellés (tiret cadratin cp1252, U+FFFD, \\x96).
4. Rapport des effectifs par classe ET par partition (nouvelle Table II).
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset

CIC_DAYS_ORDER = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
]


def _safe_read_csv(path):
    for enc in ('utf-8', 'cp1252', 'latin1'):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception:
            continue
    raise IOError(f"Lecture impossible : {path}")


def _clean_label(s):
    for bad in ('\ufffd', '\u2013', '\x96'):
        s = s.replace(bad, '-')
    return ' '.join(s.split())  # espaces multiples -> un seul


def _clean_day(df):
    df.columns = [c.strip() for c in df.columns]
    if 'Label' not in df.columns:
        return None
    df['Label'] = df['Label'].astype(str).map(_clean_label)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if c != 'Label':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(how='any')
    return df if len(df) else None


def chrono_split_per_class(y, train_frac=0.6, val_frac=0.2):
    """Split chronologique par classe (ordre du fichier conservé)."""
    tr, va, te = [], [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]                     # ordre croissant = ordre du fichier
        n1 = int(round(len(idx) * train_frac))
        n2 = int(round(len(idx) * (train_frac + val_frac)))
        tr.append(idx[:n1]); va.append(idx[n1:n2]); te.append(idx[n2:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def chrono_split_rows(n, train_frac=0.6, val_frac=0.2):
    """Ancien split (conservé uniquement pour l'ablation de protocole)."""
    n1, n2 = int(n * train_frac), int(n * (train_frac + val_frac))
    idx = np.arange(n)
    return idx[:n1], idx[n1:n2], idx[n2:]


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def _ds(X, y):
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                         torch.from_numpy(np.asarray(y, dtype=np.int64)))


def create_tasks_temporal(data_path='./data/CIC-IDS2017/MachineLearningCSV/',
                          split='per_class', max_per_day=None, verbose=True):
    """
    Retourne (tasks, class_names, counts) ;
    tasks[i] = {'name', 'train', 'val', 'test'} ;
    counts[i] = {'train': {classe: n}, 'val': {...}, 'test': {...}}.
    """
    frames, all_labels = [], []
    for fname in CIC_DAYS_ORDER:
        fpath = os.path.join(data_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        df = _clean_day(_safe_read_csv(fpath))
        if max_per_day is not None and len(df) > max_per_day:
            df = df.iloc[:max_per_day].copy()
        frames.append((fname.replace('.pcap_ISCX.csv', ''), df))
        all_labels.extend(df['Label'].tolist())

    le = LabelEncoder().fit(np.array(all_labels, dtype=str))
    class_names = list(le.classes_)

    raw, scaler = [], None
    for day_id, (name, df) in enumerate(frames):
        y = le.transform(df['Label'].values).astype(np.int64)
        X = np.nan_to_num(df.drop(columns=['Label']).values.astype(np.float64))
        X = signed_log1p(X).astype(np.float32)   # float32 : mémoire divisée par 2
        if split == 'per_class':
            tr, va, te = chrono_split_per_class(y)
        elif split == 'rows':
            tr, va, te = chrono_split_rows(len(y))
        else:
            raise ValueError(split)
        if day_id == 0:                       # scaler causal : premier jour seulement
            scaler = StandardScaler().fit(X[tr])
        raw.append((name, X, y, tr, va, te))

    tasks, counts = [], []
    for name, X, y, tr, va, te in raw:
        Xs = scaler.transform(X).astype(np.float32)
        tasks.append({'name': name,
                      'train': _ds(Xs[tr], y[tr]),
                      'val': _ds(Xs[va], y[va]),
                      'test': _ds(Xs[te], y[te])})
        c = {}
        for part, idx in (('train', tr), ('val', va), ('test', te)):
            u, n = np.unique(y[idx], return_counts=True)
            c[part] = {class_names[k]: int(v) for k, v in zip(u, n)}
        counts.append(c)

    if verbose:
        print_partition_table(tasks, counts)
    return tasks, class_names, counts


def print_partition_table(tasks, counts):
    print("\nEffectifs par tâche, classe et partition (nouvelle Table II)")
    for t, c in zip(tasks, counts):
        print(f"  {t['name']}")
        for cls in sorted(c['train'].keys() | c['val'].keys() | c['test'].keys()):
            print(f"     {cls:<28} train={c['train'].get(cls, 0):>8,} "
                  f"val={c['val'].get(cls, 0):>8,} test={c['test'].get(cls, 0):>8,}")


# Compatibilité avec l'ancien appel create_tasks(mode='temporal', ...)
def create_tasks(X=None, y=None, num_tasks=8, mode='temporal',
                 data_path='./data/CIC-IDS2017/MachineLearningCSV/', seed=42,
                 max_per_day=None, split='per_class'):
    tasks, names, _ = create_tasks_temporal(data_path, split=split,
                                            max_per_day=max_per_day)
    return tasks, names


def generate_complex_data(n_samples=10000, n_features=32, n_classes=10):
    """Conservée pour compatibilité des imports de l'ancien code (non utilisée)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=15, n_redundant=8, n_classes=n_classes,
                               n_clusters_per_class=2, random_state=42)
    return X, y
