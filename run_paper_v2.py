"""
Expériences du papier — VERSION CORRIGÉE.

Exemples :
  # 1) Vérifier le protocole (aucun entraînement) : effectifs + classifieur constant
  python run_paper_v2.py --check

  # 2) Test rapide (1 seed, Uniform seul) : T7 doit être appris
  python run_paper_v2.py --seeds 42 --methods ReplayUniform --out results/smoke.json

  # 3) Expérience principale
  python run_paper_v2.py --out results/main.json

  # 4) Ablations (une sortie par réglage)
  python run_paper_v2.py --methods ReplayUniform --set REPLAY_LAMBDA=0.25 --out results/abl_lambda025.json
  python run_paper_v2.py --methods EWC --set EWC_LAMBDA=100 --out results/abl_ewc100.json
  python run_paper_v2.py --split rows --out results/abl_split_rows.json
"""

import argparse
import json
import os
import platform
from datetime import datetime

import numpy as np
import torch
from scipy import stats

from data.generator import create_tasks_temporal
from cl_methods import CLConfig, run_method, final_metrics

ALL_METHODS = ['Majority', 'FineTuning', 'EWC', 'ReplayUniform', 'ReplayRisk', 'Joint']
SUMMARY_KEYS = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'mcc',
                'detection_rate', 'false_positive_rate']
PLANNED = [('ReplayRisk', 'ReplayUniform'), ('ReplayUniform', 'EWC'),
           ('ReplayUniform', 'FineTuning'), ('EWC', 'FineTuning')]
PRIMARY = 'mcc'


def env_info():
    info = {'platform': platform.platform(), 'python': platform.python_version(),
            'torch': torch.__version__, 'cpu': platform.processor(),
            'cuda': torch.cuda.is_available()}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def apply_overrides(pairs):
    for kv in pairs or []:
        k, v = kv.split('=', 1)
        if not hasattr(CLConfig, k):
            raise KeyError(f"Paramètre inconnu : {k}")
        old = getattr(CLConfig, k)
        setattr(CLConfig, k, type(old)(v) if not isinstance(old, bool)
                else v.lower() in ('1', 'true', 'yes'))


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i])); adj[i] = run
    return adj


def metric_by_seed(runs, method, key):
    return {r['seed']: r['metrics'][key] for r in runs if r['method'] == method
            and r['metrics'].get(key) is not None}


def summarize(out):
    runs = out['runs']
    methods = [m for m in ALL_METHODS if any(r['method'] == m for r in runs)]
    print("\n" + "=" * 100)
    print("TABLE IV (moyenne ± écart-type sur les seeds)")
    print(f"{'Méthode':<15}" + "".join(f"{k[:12]:>14}" for k in SUMMARY_KEYS)
          + f"{'Forget(MR)':>14}")
    for m in methods:
        row = f"{m:<15}"
        for k in SUMMARY_KEYS:
            v = list(metric_by_seed(runs, m, k).values())
            row += f"{np.mean(v):>8.3f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f}" if v else f"{'—':>14}"
        f = [r['forgetting_macro_recall'] for r in runs
             if r['method'] == m and r.get('forgetting_macro_recall') is not None]
        row += f"{np.mean(f):>8.3f}±{np.std(f, ddof=1) if len(f) > 1 else 0:.3f}" if f else f"{'N/A':>14}"
        print(row)

    print(f"\nTests appariés sur {PRIMARY.upper()} (t de Student apparié, correction de Holm)")
    res, pv = [], []
    for a, b in PLANNED:
        da, db = metric_by_seed(runs, a, PRIMARY), metric_by_seed(runs, b, PRIMARY)
        seeds = sorted(set(da) & set(db))
        if len(seeds) < 3:
            continue
        d = np.array([da[s] - db[s] for s in seeds])
        if np.allclose(d, 0):
            print(f"   ⚠️ {a} et {b} donnent des résultats IDENTIQUES : vérifier les flags")
            continue
        t, p = stats.ttest_rel([da[s] for s in seeds], [db[s] for s in seeds])
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        res.append((a, b, d.mean(), half, len(d))); pv.append(p)
    if pv:
        for (a, b, md, half, n), p, pa in zip(res, pv, holm(np.array(pv))):
            print(f"   {a:>13} − {b:<13} Δ={md:+.4f}  IC95=[{md-half:+.4f}, {md+half:+.4f}]  "
                  f"n={n}  p={p:.4f}  p_Holm={pa:.4f}")

    print("\nRappel par classe (moyenne sur les seeds) — nouvelle table du papier")
    classes = sorted({c for r in runs for c in r['metrics']['per_class']})
    print(f"{'Classe':<28}{'support':>9}" + "".join(f"{m[:11]:>12}" for m in methods))
    for c in classes:
        sup = next(r['metrics']['per_class'][c]['support'] for r in runs
                   if c in r['metrics']['per_class'])
        row = f"{c:<28}{sup:>9}"
        for m in methods:
            v = [r['metrics']['per_class'][c]['recall'] for r in runs
                 if r['method'] == m and c in r['metrics']['per_class']]
            row += f"{np.mean(v):>12.3f}" if v else f"{'—':>12}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/CIC-IDS2017/MachineLearningCSV/')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1011])
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS)
    ap.add_argument('--split', default='per_class', choices=['per_class', 'rows'])
    ap.add_argument('--max-per-day', type=int, default=None)
    ap.add_argument('--out', default='results/main.json')
    ap.add_argument('--set', nargs='*', help='surcharges CLConfig, ex. REPLAY_LAMBDA=0.25')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    apply_overrides(args.set)
    CLConfig.VERBOSE = not args.quiet
    tasks, class_names, counts = create_tasks_temporal(args.data, split=args.split,
                                                       max_per_day=args.max_per_day)
    if args.check:
        b = class_names.index('BENIGN')
        y = np.concatenate([t['test'].tensors[1].numpy() for t in tasks])
        m = final_metrics(y, np.full_like(y, b), class_names)
        print(f"\nClassifieur constant « BENIGN » : acc={m['accuracy']:.4f} "
              f"F1w={m['f1_weighted']:.4f} MCC={m['mcc']:.4f}")
        absent = [c for c in class_names if c not in m['per_class']]
        print(f"Classes absentes du test agrégé : {absent or 'aucune'}")
        return

    meta = {'config': CLConfig.as_dict(), 'split': args.split,
            'max_per_day': args.max_per_day, 'class_names': class_names}
    out = load(args.out)
    if out is None or out.get('meta') != meta:
        if out is not None:
            print("⚠️ Configuration différente de celle du fichier existant : nouveau fichier.")
        out = {'meta': meta, 'env': env_info(), 'counts': counts, 'runs': [],
               'started': datetime.now().isoformat()}
    done = {(r['method'], r['seed']) for r in out['runs']}

    for seed in args.seeds:
        for method in args.methods:
            if (method, seed) in done:
                print(f"déjà fait : {method} seed {seed}")
                continue
            print(f"\n{'#' * 70}\n# {method} | seed {seed}\n{'#' * 70}")
            r = run_method(method, tasks, class_names, CLConfig, seed)
            out['runs'].append(r)
            out['updated'] = datetime.now().isoformat()
            save(args.out, out)
            mm = r['metrics']
            print(f"   → acc={mm['accuracy']:.4f} bacc={mm['balanced_accuracy']:.4f} "
                  f"F1m={mm['f1_macro']:.4f} MCC={mm['mcc']:.4f}")
    summarize(out)
    print(f"\nRésultats : {args.out}")


if __name__ == '__main__':
    main()
