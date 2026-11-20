"""
Protocol audit for the revision (no training).

1. Timestamps: capture-file ranges, ordering of the rows of each file, and
   whether the class-stratified split in file order (main protocol) respects
   chronology within and across classes.
2. Duplicates: exact duplicate feature vectors shared by the evaluation and the
   training partitions, by task and class, defined (a) on the raw numeric
   features before preprocessing and (b) on the preprocessed float32 vectors.

Writes results/audit_protocol.json and prints the tables used in the paper.
Usage: python revision/audit_protocol.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data.generator import (CIC_DAYS_ORDER, load_day, chrono_split_per_class,
                            chrono_split_time_cut)

ML = './data/CIC-IDS2017/MachineLearningCSV/'
T = lambda t: pd.to_datetime(int(t), unit='s').strftime('%H:%M:%S')

ORDER = sys.argv[1] if len(sys.argv) > 1 else 'file'      # 'file' (submitted) or 'time' (revised)
days = [(f.replace('.pcap_ISCX.csv', ''), load_day(f, ML)) for f in CIC_DAYS_ORDER]
if ORDER == 'time':
    days = [(n, {k: v[np.argsort(d['ts'], kind='stable')] for k, v in d.items()}) for n, d in days]
le = LabelEncoder().fit(np.concatenate([d['labels'] for _, d in days]))
names = list(le.classes_)
out = {'files': [], 'split_chronology': [], 'duplicates': [], 'time_vs_file_split': []}

# ------------------------------------------------------------------ timestamps
print("\n=== Capture files ===")
prev_end = None
for name, d in days:
    ts = d['ts']
    dec = np.diff(ts) < 0
    runmax = np.maximum.accumulate(ts)
    late = ts < runmax - 60          # row starts > 1 min before an earlier row of the file
    rank_file = np.arange(len(ts))
    rank_time = np.empty(len(ts), int); rank_time[np.argsort(ts, kind='stable')] = rank_file
    rec = {'task': name, 'n': int(len(ts)), 'start': T(ts.min()), 'end': T(ts.max()),
           'adjacent_decreases': int(dec.sum()),
           'rows_more_than_1min_out_of_order': int(late.sum()),
           'frac_out_of_order': float(late.mean()),
           'spearman_file_vs_time': float(pd.Series(rank_file).corr(pd.Series(rank_time), method='pearson')),
           'overlaps_previous_file': bool(prev_end is not None and ts.min() < prev_end)}
    prev_end = ts.max()
    out['files'].append(rec)
    print(f"{name:<45} {rec['start']}-{rec['end']}  out-of-order(>1 min)={rec['rows_more_than_1min_out_of_order']:>6} "
          f"({100*rec['frac_out_of_order']:.2f}%)  rank corr={rec['spearman_file_vs_time']:.4f}  overlap prev={rec['overlaps_previous_file']}")

# --------------------------------------------- chronology of the main split
print("\n=== Main protocol (class-stratified split in file order): chronology ===")
for name, d in days:
    y = le.transform(d['labels']); ts = d['ts']
    tr, va, te = chrono_split_per_class(y)
    rec = {'task': name, 'partitions': {}, 'per_class': {}}
    for part, idx in (('train', tr), ('val', va), ('test', te)):
        rec['partitions'][part] = [T(ts[idx].min()), T(ts[idx].max())]
    # across classes: training flows that start after the earliest test flow of the day
    t_test0 = ts[te].min()
    rec['train_after_first_test_any_class'] = int((ts[tr] > t_test0).sum())
    rec['train_after_first_test_any_class_frac'] = float((ts[tr] > t_test0).mean())
    for k in np.unique(y):
        trk, vak, tek = tr[y[tr] == k], va[y[va] == k], te[y[te] == k]
        if len(tek) == 0:
            continue
        rec['per_class'][names[k]] = {
            'train': [T(ts[trk].min()), T(ts[trk].max())],
            'val': [T(ts[vak].min()), T(ts[vak].max())] if len(vak) else None,
            'test': [T(ts[tek].min()), T(ts[tek].max())],
            # within class: training flows starting after the first test flow of that class
            'train_after_first_test': int((ts[trk] > ts[tek].min()).sum()),
            'train_same_minute_as_test': int(np.isin(ts[trk] // 60, ts[tek] // 60).sum()),
            'n_train': int(len(trk)), 'n_test': int(len(tek)),
        }
    out['split_chronology'].append(rec)
    print(f"{name:<45} train {rec['partitions']['train']} val {rec['partitions']['val']} test {rec['partitions']['test']}"
          f"  train flows after first test flow (any class): {rec['train_after_first_test_any_class']:,} "
          f"({100*rec['train_after_first_test_any_class_frac']:.1f}%)")
    for c, v in rec['per_class'].items():
        print(f"      {c:<28} train {v['train'][0]}-{v['train'][1]}  test {v['test'][0]}-{v['test'][1]}  "
              f"train>first test: {v['train_after_first_test']:,}  same minute: {v['train_same_minute_as_test']:,}")

    # agreement between file-order and time-order class-stratified splits
    perm = np.argsort(ts, kind='stable')
    tr2, va2, te2 = chrono_split_per_class(y[perm])
    lab_f = np.empty(len(y), int); lab_f[tr], lab_f[va], lab_f[te] = 0, 1, 2
    lab_t = np.empty(len(y), int); lab_t[perm[tr2]], lab_t[perm[va2]], lab_t[perm[te2]] = 0, 1, 2
    out['time_vs_file_split'].append({'task': name, 'flows_changing_partition': int((lab_f != lab_t).sum()),
                                      'frac': float((lab_f != lab_t).mean())})

print("\n=== File order vs timestamp order (class-stratified split) ===")
for r in out['time_vs_file_split']:
    print(f"{r['task']:<45} flows changing partition: {r['flows_changing_partition']:,} ({100*r['frac']:.2f}%)")

# ------------------------------------------------------------------ duplicates
print("\n=== Exact duplicates between evaluation and training partitions (main protocol) ===")
splits, Xs_all = [], []
scaler = None
for i, (name, d) in enumerate(days):
    y = le.transform(d['labels'])
    tr, va, te = chrono_split_per_class(y)
    if i == 0:
        scaler = StandardScaler().fit(d['X'][tr])
    Xs = scaler.transform(d['X']).astype(np.float32)
    hp = pd.util.hash_pandas_object(pd.DataFrame(Xs.astype(np.float64)), index=False).values
    splits.append((name, y, tr, va, te, d['hash'], hp))

for key, label in (('raw', 'raw features (before preprocessing)'), ('pre', 'preprocessed float32 vectors')):
    col = 5 if key == 'raw' else 6
    all_train = np.unique(np.concatenate([s[col][s[2]] for s in splits]))
    pairs = pd.DataFrame({'h': np.concatenate([s[col][s[2]] for s in splits]),
                          'y': np.concatenate([s[1][s[2]] for s in splits])}).drop_duplicates()
    n_labels = pairs.groupby('h').size()               # distinct training labels per vector
    pair_set = set(zip(pairs['h'].values.tolist(), pairs['y'].values.tolist()))
    print(f"\n-- defined on {label}")
    tot = {'test': 0, 'test_dup': 0}
    for name, y, tr, va, te, h_raw, h_pre in splits:
        h = h_raw if key == 'raw' else h_pre
        same_task_train = h[tr]
        for part, idx in (('val', va), ('test', te)):
            for k in np.unique(y[idx]):
                ik = idx[y[idx] == k]
                in_same_class = np.isin(h[ik], h[tr][y[tr] == k])
                in_same_task = np.isin(h[ik], same_task_train)
                in_any = np.isin(h[ik], all_train)
                # duplicates whose training copy carries another label
                hd = h[ik][in_any]
                nl = n_labels.reindex(hd).fillna(0).values
                has_k = np.array([(int(x), int(k)) in pair_set for x in hd], bool)
                conflict = (nl - has_k) > 0
                rec = {'definition': key, 'task': name, 'partition': part, 'class': names[k],
                       'n': int(len(ik)), 'dup_same_task_same_class': int(in_same_class.sum()),
                       'dup_same_task_any_class': int(in_same_task.sum()),
                       'dup_any_task_any_class': int(in_any.sum()),
                       'dup_with_other_label': int(conflict.sum())}
                out['duplicates'].append(rec)
                if part == 'test':
                    tot['test'] += rec['n']; tot['test_dup'] += rec['dup_any_task_any_class']
                    print(f"   {name:<40} {names[k]:<28} n={rec['n']:>7,}  dup(same class)={rec['dup_same_task_same_class']:>7,} "
                          f"({100*rec['dup_same_task_same_class']/max(1,rec['n']):5.1f}%)  dup(any train)={rec['dup_any_task_any_class']:>7,} "
                          f"({100*rec['dup_any_task_any_class']/max(1,rec['n']):5.1f}%)  other label={rec['dup_with_other_label']:,}")
    print(f"   TOTAL test flows {tot['test']:,}; with an identical training vector: {tot['test_dup']:,} "
          f"({100*tot['test_dup']/tot['test']:.1f}%)")
    out[f'total_test_dup_{key}'] = tot

os.makedirs('results', exist_ok=True)
with open('results/audit_protocol.json' if ORDER == 'file' else 'results/audit_protocol_time.json', 'w') as f:
    json.dump(out, f, indent=1)
print("\nwritten: results/audit_protocol.json")
