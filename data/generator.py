"""
Temporal task construction for CIC-IDS2017.

Default behaviour (order='file', split='per_class', dedup=None) is the protocol of
the main experiment and is unchanged from the submitted version:
  * one task per capture file, in chronological order (Monday -> Friday PM);
  * within each day, the flows of each class are split 60/20/20 in file order
    (class-stratified chronological split);
  * signed log transform, then a StandardScaler fitted on the training
    partition of the first day only;
  * labels normalised (dash encodings), rows with NaN/inf removed.

Options added for the revision (all off by default):
  order='time'     rows of each day are stably sorted by the flow timestamp
                   taken from the TrafficLabelling CSV files (same rows, same
                   order as MachineLearningCSV; see revision/check_align.py).
  split='rows'     shared cutoffs at 60/80 % of the rows of the day (all
                   classes cut at the same position).
  split='time_cut' shared chronological cutoffs on whole minutes: every
                   training flow of a day starts strictly before every
                   validation flow, which starts strictly before every test
                   flow. Classes may be absent from some partitions.
  dedup='eval'     validation and test flows whose raw feature vector also
                   occurs in ANY training partition (any task, any class) are
                   removed. Duplicates are defined on the raw numeric features
                   before any preprocessing.
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
DEFAULT_TS_PATH = './data/CIC-IDS2017/TrafficLabelling/'
CACHE_DIR = './data/cache/'


def _safe_read_csv(path, **kw):
    for enc in ('utf-8', 'cp1252', 'latin1'):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, **kw)
        except Exception:
            continue
    raise IOError(f"Cannot read: {path}")


def _clean_label(s):
    for bad in ('�', '–', '\x96'):
        s = s.replace(bad, '-')
    return ' '.join(s.split())


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


def parse_cic2017_timestamps(series):
    """CIC-IDS2017 TrafficLabelling timestamps: d/m/yyyy H:MM (minute resolution) or
    dd/mm/yyyy HH:MM:SS (Monday), written on a 12-hour clock without AM/PM.
    Captures run from about 08:40 to 17:05, so hours 1-7 are afternoon hours."""
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    m = dt.isna()
    if m.any():
        dt[m] = pd.to_datetime(s[m], format='%d/%m/%Y %H:%M', errors='coerce')
    if dt.isna().any():
        raise ValueError(f"{int(dt.isna().sum())} unparsable timestamps")
    dt = dt + pd.to_timedelta(np.where(dt.dt.hour < 8, 12, 0), unit='h')
    return (dt.values.astype('datetime64[s]').astype(np.int64))


def load_day(fname, data_path, ts_path=None, use_cache=True):
    """Cleaned day: dict(Xlog float32, labels str, hash uint64, ts int64 or None)."""
    cache = os.path.join(CACHE_DIR, fname.replace('.csv', '.npz'))
    tsp = ts_path if ts_path is not None else (DEFAULT_TS_PATH if os.path.exists(
        os.path.join(DEFAULT_TS_PATH, fname)) else None)
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d['ts'] = d['ts'] if d['ts'].size else None
        if not (tsp is not None and d['ts'] is None):   # else rebuild with timestamps
            return d
    df = _clean_day(_safe_read_csv(os.path.join(data_path, fname)))
    Xraw = np.nan_to_num(df.drop(columns=['Label']).values.astype(np.float64))
    # duplicate key: raw numeric feature vector, before any preprocessing
    h = pd.util.hash_pandas_object(pd.DataFrame(Xraw + 0.0), index=False).values.astype(np.uint64)
    X = signed_log1p(Xraw).astype(np.float32)
    ts = None
    if tsp is not None:
        raw = _safe_read_csv(os.path.join(tsp, fname), skipinitialspace=True)
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw.dropna(how='all')
        # same rows as MachineLearningCSV (checked in revision/check_align.py)
        assert len(raw) >= df.index.max() + 1
        ts = parse_cic2017_timestamps(raw['Timestamp'].iloc[df.index.values])
    d = {'X': X, 'labels': df['Label'].values.astype(str), 'hash': h,
         'ts': ts if ts is not None else np.empty(0, dtype=np.int64)}
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **d)
    d['ts'] = ts
    return d


def chrono_split_per_class(y, train_frac=0.6, val_frac=0.2):
    """Class-stratified chronological split (positions in the given order)."""
    tr, va, te = [], [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        n1 = int(round(len(idx) * train_frac))
        n2 = int(round(len(idx) * (train_frac + val_frac)))
        tr.append(idx[:n1]); va.append(idx[n1:n2]); te.append(idx[n2:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def chrono_split_rows(n, train_frac=0.6, val_frac=0.2):
    """Shared cutoffs at fixed fractions of the rows (all classes together)."""
    n1, n2 = int(n * train_frac), int(n * (train_frac + val_frac))
    idx = np.arange(n)
    return idx[:n1], idx[n1:n2], idx[n2:]


def chrono_split_time_cut(ts_sorted, train_frac=0.6, val_frac=0.2):
    """Shared cutoffs on whole minutes (ts must be sorted). No minute is shared
    between two partitions, so the partitions are strictly ordered in time."""
    minute = ts_sorted // 60
    n = len(minute)
    u, first = np.unique(minute, return_index=True)
    ends = np.append(first[1:], n)                # end position of each minute
    def cut(frac):                                # first minute boundary >= frac
        j = np.searchsorted(ends, frac * n)
        return int(ends[min(j, len(ends) - 1)])
    n1, n2 = cut(train_frac), cut(train_frac + val_frac)
    if n2 <= n1:
        n2 = n1
    idx = np.arange(n)
    return idx[:n1], idx[n1:n2], idx[n2:]


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def _ds(X, y):
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                         torch.from_numpy(np.asarray(y, dtype=np.int64)))


def _hhmm(t):
    return pd.to_datetime(int(t), unit='s').strftime('%H:%M:%S')


def create_tasks_temporal(data_path='./data/CIC-IDS2017/MachineLearningCSV/',
                          split='per_class', max_per_day=None, verbose=True,
                          order='file', dedup=None, ts_path=None, return_info=False):
    """
    Returns (tasks, class_names, counts[, info]);
    tasks[i] = {'name', 'train', 'val', 'test'};
    counts[i] = {'train': {class: n}, 'val': {...}, 'test': {...}};
    info (return_info=True) = per-task timestamp ranges and dedup removals.
    """
    need_ts = order == 'time' or split == 'time_cut'
    days, all_labels = [], []
    for fname in CIC_DAYS_ORDER:
        if not os.path.exists(os.path.join(data_path, fname)):
            raise FileNotFoundError(os.path.join(data_path, fname))
        d = load_day(fname, data_path, ts_path)
        if need_ts and d['ts'] is None:
            raise FileNotFoundError("Timestamps required: TrafficLabelling CSV files not found")
        if max_per_day is not None and len(d['labels']) > max_per_day:
            d = {k: (v[:max_per_day] if v is not None else None) for k, v in d.items()}
        days.append((fname.replace('.pcap_ISCX.csv', ''), d))
        all_labels.extend(d['labels'].tolist())

    le = LabelEncoder().fit(np.array(all_labels, dtype=str))
    class_names = list(le.classes_)

    raw, scaler = [], None
    for day_id, (name, d) in enumerate(days):
        y = le.transform(d['labels']).astype(np.int64)
        X, h, ts = d['X'], d['hash'], d['ts']
        if order == 'time':
            perm = np.argsort(ts, kind='stable')
            X, y, h, ts = X[perm], y[perm], h[perm], ts[perm]
        elif order != 'file':
            raise ValueError(order)
        if split == 'per_class':
            tr, va, te = chrono_split_per_class(y)
        elif split == 'rows':
            tr, va, te = chrono_split_rows(len(y))
        elif split == 'time_cut':
            if order != 'time':
                raise ValueError("split='time_cut' requires order='time'")
            tr, va, te = chrono_split_time_cut(ts)
        else:
            raise ValueError(split)
        if day_id == 0:
            scaler = StandardScaler().fit(X[tr])
        raw.append([name, X, y, h, ts, tr, va, te])

    removed = [None] * len(raw)
    if dedup == 'eval':
        train_hashes = np.unique(np.concatenate([r[3][r[5]] for r in raw]))
        for i, r in enumerate(raw):
            h, y = r[3], r[2]
            rem = {}
            for j, part in ((6, 'val'), (7, 'test')):
                idx = r[j]
                dup = np.isin(h[idx], train_hashes)
                u, n = np.unique(y[idx][dup], return_counts=True)
                rem[part] = {class_names[k]: int(v) for k, v in zip(u, n)}
                r[j] = idx[~dup]
            removed[i] = rem
    elif dedup is not None:
        raise ValueError(dedup)

    tasks, counts, info = [], [], []
    for (name, X, y, h, ts, tr, va, te), rem in zip(raw, removed):
        Xs = scaler.transform(X).astype(np.float32)
        tasks.append({'name': name,
                      'train': _ds(Xs[tr], y[tr]),
                      'val': _ds(Xs[va], y[va]),
                      'test': _ds(Xs[te], y[te])})
        c, ti = {}, {'name': name, 'removed_duplicates': rem}
        for part, idx in (('train', tr), ('val', va), ('test', te)):
            u, n = np.unique(y[idx], return_counts=True)
            c[part] = {class_names[k]: int(v) for k, v in zip(u, n)}
            if ts is not None and len(idx):
                ti[part] = {'start': _hhmm(ts[idx].min()), 'end': _hhmm(ts[idx].max()),
                            'per_class': {class_names[k]: [_hhmm(ts[idx][y[idx] == k].min()),
                                                           _hhmm(ts[idx][y[idx] == k].max())]
                                          for k in u}}
        counts.append(c); info.append(ti)

    if verbose:
        print_partition_table(tasks, counts)
    if return_info:
        return tasks, class_names, counts, info
    return tasks, class_names, counts


def print_partition_table(tasks, counts):
    print("\nFlows per task, class and partition")
    for t, c in zip(tasks, counts):
        print(f"  {t['name']}")
        for cls in sorted(c['train'].keys() | c['val'].keys() | c['test'].keys()):
            print(f"     {cls:<28} train={c['train'].get(cls, 0):>8,} "
                  f"val={c['val'].get(cls, 0):>8,} test={c['test'].get(cls, 0):>8,}")


# Backward compatibility with the earlier create_tasks(mode='temporal', ...) call
def create_tasks(X=None, y=None, num_tasks=8, mode='temporal',
                 data_path='./data/CIC-IDS2017/MachineLearningCSV/', seed=42,
                 max_per_day=None, split='per_class'):
    tasks, names, _ = create_tasks_temporal(data_path, split=split,
                                            max_per_day=max_per_day)
    return tasks, names


def generate_complex_data(n_samples=10000, n_features=32, n_classes=10):
    """Kept for the imports of the earlier code (not used)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=15, n_redundant=8, n_classes=n_classes,
                               n_clusters_per_class=2, random_state=42)
    return X, y
