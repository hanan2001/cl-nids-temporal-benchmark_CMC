"""
Diagnostic with non-neural offline classifiers (revision, Reviewer 2).

Trains a random forest and a histogram gradient-boosting classifier offline on
the union of the training partitions of the main protocol (same partitions and
preprocessing as joint training) and reports MCC and per-class recall on the
aggregated test set. If the classes that remain difficult for the MLP are also
difficult for these models, the difficulty is less likely to come from the MLP
optimisation alone.

Usage: python revision/other_classifiers.py [--seeds 42 123 456] [--dedup]
Writes results/other_classifiers[_dedup].json
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from data.generator import create_tasks_temporal
from cl_methods import final_metrics

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
ap.add_argument('--dedup', action='store_true')
ap.add_argument('--jobs', type=int, default=6)
args = ap.parse_args()

from data.prepared import load_or_prepare
tasks, names, _, _ = load_or_prepare('cic2017', './data/CIC-IDS2017/MachineLearningCSV/', order='time', dedup='eval' if args.dedup else None)
cat = lambda part, i: torch.cat([t[part].tensors[i] for t in tasks]).numpy()
Xtr, ytr, Xte, yte = cat('train', 0), cat('train', 1), cat('test', 0), cat('test', 1)
print(f"train {len(ytr):,}  test {len(yte):,}")

out = {'dedup': args.dedup, 'runs': []}
for seed in args.seeds:
    for name in ('RandomForest', 'HistGradientBoosting'):
        t0 = time.time()
        if name == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=args.jobs, random_state=seed,
                                         class_weight='balanced_subsample', min_samples_leaf=1)
            clf.fit(Xtr, ytr)
        else:
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed,
                                                 early_stopping=True, validation_fraction=0.1)
            clf.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        p = clf.predict(Xte)
        m = final_metrics(yte, p, names)
        m['fit_seconds'] = time.time() - t0
        out['runs'].append({'method': name, 'seed': seed, 'metrics': m})
        print(f"{name:<22} seed {seed}: MCC={m['mcc']:.4f} bal.acc={m['balanced_accuracy']:.4f} "
              f"FPR={100*m['false_positive_rate']:.2f}% ({m['fit_seconds']:.0f}s)")
        print("     " + "  ".join(f"{c[:14]}={v['recall']:.3f}" for c, v in m['per_class'].items()))

fn = 'results/other_classifiers' + ('_dedup' if args.dedup else '') + '.json'
with open(fn, 'w') as f:
    json.dump(out, f, indent=1)
print('written', fn)
