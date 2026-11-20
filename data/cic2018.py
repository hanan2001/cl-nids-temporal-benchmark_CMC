"""
Temporal task construction for CSE-CIC-IDS2018 (second dataset of the revision).

Files: "Processed Traffic Data for ML Algorithms" (CICFlowMeter-V3 flows), public
AWS bucket s3://cse-cic-ids2018/. One task per capture day (10 days, 14 Feb ->
2 Mar 2018), in calendar order. Unlike the CIC-IDS2017 ML files, these files keep
the flow timestamp (dd/mm/yyyy HH:MM:SS, 24-hour clock), so the rows of each day
are sorted by timestamp before splitting.

To keep CPU training time comparable with CIC-IDS2017, each day is subsampled
once, with a fixed seed, before any split: at most MAX_BENIGN benign flows and
MAX_ATTACK flows per attack class and day, drawn uniformly at random; the
selected rows keep their time order. Identifier columns (Flow ID, Src IP,
Src Port, Dst IP) present in one file are dropped; Timestamp is used for
ordering only, never as a feature. Rows repeating the header are removed.

Protocol (same as the main CIC-IDS2017 protocol): class-stratified
chronological 60/20/20 split within each day (split='per_class'), or shared
whole-minute cutoffs (split='time_cut'); signed log + StandardScaler fitted on
the first day's training partition only.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data.generator import (signed_log1p, _ds, chrono_split_per_class,
                            chrono_split_time_cut, print_partition_table, _hhmm)

DAYS_2018 = [
    'Wednesday-14-02-2018', 'Thursday-15-02-2018', 'Friday-16-02-2018',
    'Thuesday-20-02-2018', 'Wednesday-21-02-2018', 'Thursday-22-02-2018',
    'Friday-23-02-2018', 'Wednesday-28-02-2018', 'Thursday-01-03-2018',
    'Friday-02-03-2018',
]
SUFFIX = '_TrafficForML_CICFlowMeter.csv'
DROP = ('Flow ID', 'Src IP', 'Src Port', 'Dst IP')
CACHE_DIR = './data/cache2018/'
MAX_BENIGN = 150_000
MAX_ATTACK = 50_000
SUBSAMPLE_SEED = 2018


def _read_labels(path):
    labs = []
    for ch in pd.read_csv(path, usecols=['Label'], chunksize=1_000_000, dtype=str):
        labs.append(ch['Label'].values)
    return np.concatenate(labs)


def load_day_2018(day, data_path, use_cache=True):
    cache = os.path.join(CACHE_DIR, f'{day}_{MAX_BENIGN}_{MAX_ATTACK}.npz')
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    path = os.path.join(data_path, day + SUFFIX)
    labels = _read_labels(path)
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    keep = np.zeros(len(labels), bool)
    for c in np.unique(labels):
        if c == 'Label':                       # repeated header rows
            continue
        idx = np.where(labels == c)[0]
        cap = MAX_BENIGN if c == 'Benign' else MAX_ATTACK
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep[idx] = True
    parts, start = [], 0
    for ch in pd.read_csv(path, chunksize=500_000, dtype=str):
        k = keep[start:start + len(ch)]
        start += len(ch)
        if k.any():
            parts.append(ch[k])
    df = pd.concat(parts)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    ts = pd.to_datetime(df.pop('Timestamp'), format='%d/%m/%Y %H:%M:%S', errors='coerce')
    lab = df.pop('Label').astype(str).str.strip().values
    X = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    ok = np.isfinite(X).all(1) & ts.notna().values
    X, lab, ts = X[ok], lab[ok], ts.values[ok].astype('datetime64[s]').astype(np.int64)
    h = pd.util.hash_pandas_object(pd.DataFrame(X + 0.0), index=False).values.astype(np.uint64)
    order = np.argsort(ts, kind='stable')
    d = {'X': signed_log1p(X[order]).astype(np.float32), 'labels': lab[order],
         'ts': ts[order], 'hash': h[order], 'columns': np.array(df.columns, dtype=str)}
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **d)
    return d


def create_tasks_2018(data_path='./data/CSE-CIC-IDS2018/', split='per_class',
                      dedup=None, verbose=True, return_info=False):
    days = [(d, load_day_2018(d, data_path)) for d in DAYS_2018]
    cols = days[0][1]['columns']
    for _, d in days:
        assert (d['columns'] == cols).all(), "feature columns differ between days"
    le = LabelEncoder().fit(np.concatenate([d['labels'] for _, d in days]))
    class_names = list(le.classes_)
    raw, scaler = [], None
    for i, (name, d) in enumerate(days):
        y = le.transform(d['labels']).astype(np.int64)
        if split == 'per_class':
            tr, va, te = chrono_split_per_class(y)
        elif split == 'time_cut':
            tr, va, te = chrono_split_time_cut(d['ts'])
        else:
            raise ValueError(split)
        if i == 0:
            scaler = StandardScaler().fit(d['X'][tr])
        raw.append([name, d['X'], y, d['hash'], d['ts'], tr, va, te])
    removed = [None] * len(raw)
    if dedup == 'eval':
        train_h = np.unique(np.concatenate([r[3][r[5]] for r in raw]))
        for i, r in enumerate(raw):
            rem = {}
            for j, part in ((6, 'val'), (7, 'test')):
                dup = np.isin(r[3][r[j]], train_h)
                u, n = np.unique(r[2][r[j]][dup], return_counts=True)
                rem[part] = {class_names[k]: int(v) for k, v in zip(u, n)}
                r[j] = r[j][~dup]
            removed[i] = rem
    tasks, counts, info = [], [], []
    for (name, X, y, h, ts, tr, va, te), rem in zip(raw, removed):
        Xs = scaler.transform(X).astype(np.float32)
        tasks.append({'name': name, 'train': _ds(Xs[tr], y[tr]),
                      'val': _ds(Xs[va], y[va]), 'test': _ds(Xs[te], y[te])})
        c, ti = {}, {'name': name, 'removed_duplicates': rem}
        for part, idx in (('train', tr), ('val', va), ('test', te)):
            u, n = np.unique(y[idx], return_counts=True)
            c[part] = {class_names[k]: int(v) for k, v in zip(u, n)}
            if len(idx):
                ti[part] = {'start': _hhmm(ts[idx].min()), 'end': _hhmm(ts[idx].max())}
        counts.append(c); info.append(ti)
    if verbose:
        print_partition_table(tasks, counts)
    if return_info:
        return tasks, class_names, counts, info
    return tasks, class_names, counts
