"""
Script principal du papier IJACSA — comparaison des 5 méthodes de CL.

Méthodes comparées (protocole temporel CICIDS-2017, split chronologique) :
  1. FineTuning         (borne basse, aucun mécanisme anti-oubli)
  2. EWC                (régularisation)
  3. Replay uniforme    (buffer, échantillonnage uniforme, sans risque)
  4. Risk-Aware Replay  (TA MÉTHODE : replay pondéré risque + SMOTE + focal)
  5. Joint              (borne haute, tout vu en une fois)

Les méthodes 3 et 4 réutilisent ton ContinualLearner (USE_PROJECTION=False) :
  - Risk-Aware Replay : USE_RISK_GUIDED=True
  - Replay uniforme   : USE_RISK_GUIDED=False
Les méthodes 1, 2, 5 viennent de baselines.py.

Sortie : ./results/paper_results.json + tableau imprimé (moyenne ± std, p-values).

Lancement :
    python run_paper.py
Test rapide (sous-échantillon chronologique par jour) :
    python -c "from run_paper import run_paper; run_paper(max_per_day=20000, n_seeds=2)"
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime

from config import Config
from data.generator import create_tasks
from data.risk_mapping import get_risk_scores
from models import CyberNetwork, GradientProjectionMemory
from training import ContinualLearner
from evaluation.metrics import (compute_all_metrics,
                                compute_forgetting_rate_matrix,
                                compute_plasticity_stability_metrics)
from baselines import run_finetuning, run_ewc, run_joint
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _eval(model, dataset, device, bs):
    model.eval()
    preds, labels = [], []
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            labels.extend(yb.numpy().tolist())
    return np.array(labels), np.array(preds)


def run_replay_method(tasks, class_names, config, risk_mapping, seed,
                      use_risk, method_name, n_classes):
    """Méthodes 3 et 4 via ContinualLearner (projection désactivée)."""
    device = config.DEVICE
    in_size = tasks[0]['train'].tensors[0].shape[1]
    model = CyberNetwork(in_size, config.HIDDEN_SIZES, n_classes).to(device)

    # GPM instancié mais INACTIF (USE_PROJECTION=False dans config)
    gpm = GradientProjectionMemory(
        k=config.GPM_K, sample_size=config.GPM_SAMPLE_SIZE, use_risk=use_risk,
        device=device)

    cfg = config
    cfg.USE_RISK_GUIDED = use_risk   # bascule risque on/off
    learner = ContinualLearner(model, gpm, cfg)

    test_tasks = [t['test'] for t in tasks]
    for tid, task in enumerate(tasks):
        learner.train_task(train_dataset=task['train'], val_dataset=task['val'],
                           task_id=tid,
                           risk_mapping=risk_mapping,
                           all_test_tasks=test_tasks)
        print(f"   [{method_name}] tâche {tid+1}/{len(tasks)} done")

    # métriques finales sur agrégat test
    all_l, all_p = [], []
    for t in tasks:
        yl, yp = _eval(model, t['test'], device, config.BATCH_SIZE)
        all_l.extend(yl.tolist()); all_p.extend(yp.tolist())
    metrics = compute_all_metrics(all_l, all_p, risk_mapping)
    # H2 : MCC par groupe de risque (mesure, ne change pas l'entrainement)
    from evaluation.metrics import compute_mcc_by_risk_group
    grp = compute_mcc_by_risk_group(all_l, all_p, risk_mapping, threshold=0.7)
    metrics['mcc_high_risk'] = grp['mcc_high_risk']
    metrics['mcc_low_risk'] = grp['mcc_low_risk']
    print(f"   MCC haut-risque={grp['mcc_high_risk']} (n={grp['n_high']}) | "
          f"bas-risque={grp['mcc_low_risk']} (n={grp['n_low']}) | "
          f"flags buffer={getattr(config,'RISK_ON_BUFFER',True)} loss={getattr(config,'RISK_ON_LOSS',True)} mult={getattr(config,'RISK_MULTIPLIER',3.0)}")
    hist = learner.accuracies_history
    if len(hist) > 1:
        forget = compute_forgetting_rate_matrix(hist)
        ps = compute_plasticity_stability_metrics(hist)
    else:
        forget, ps = 0.0, {'plasticity': 0.0, 'stability': 0.0, 'tradeoff': 0.0}
    return {'method': method_name, 'seed': seed, 'metrics': metrics,
            'forgetting_rate': forget, 'plasticity': ps['plasticity'],
            'stability': ps['stability'], 'accuracies_history': hist}


def run_paper(data_path='./data/CIC-IDS2017/MachineLearningCSV/',
              max_per_day=None, n_seeds=5, seeds=(42, 123, 456, 789, 1011)):
    seeds = list(seeds)[:n_seeds]
    print("\n" + "=" * 70)
    print("🔬 PAPER IJACSA — Risk-Aware Replay vs baselines (protocole temporel)")
    print("=" * 70)

    results = {'FineTuning': [], 'EWC': [], 'ReplayUniform': [],
               'RiskAwareReplay': [], 'Joint': []}

    for si, seed in enumerate(seeds):
        print(f"\n{'#'*70}\n# SEED {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        set_seed(seed)

        # tâches temporelles (mêmes pour toutes les méthodes de cette seed)
        tasks, class_names = create_tasks(mode='temporal', data_path=data_path,
                                          max_per_day=max_per_day)
        if not tasks:
            print("   ❌ pas de tâches — vérifie data_path")
            return None
        n_classes = len(class_names)
        risk_mapping = get_risk_scores(class_names)

        cfg = Config()

        # 1. FineTuning
        results['FineTuning'].append(
            run_finetuning(tasks, class_names, cfg, risk_mapping, seed, n_classes))
        # 2. EWC
        set_seed(seed)
        results['EWC'].append(
            run_ewc(tasks, class_names, cfg, risk_mapping, seed, n_classes))
        # 3. Replay uniforme
        set_seed(seed)
        results['ReplayUniform'].append(
            run_replay_method(tasks, class_names, cfg, risk_mapping, seed,
                              use_risk=False, method_name='ReplayUniform',
                              n_classes=n_classes))
        # 4. Risk-Aware Replay (ta méthode)
        set_seed(seed)
        results['RiskAwareReplay'].append(
            run_replay_method(tasks, class_names, cfg, risk_mapping, seed,
                              use_risk=True, method_name='RiskAwareReplay',
                              n_classes=n_classes))
        # 5. Joint
        set_seed(seed)
        results['Joint'].append(
            run_joint(tasks, class_names, cfg, risk_mapping, seed, n_classes))

    # ---- agrégation + tableau ----
    metrics_keys = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'mcc',
                    'forgetting_rate']
    print("\n" + "=" * 90)
    print("📊 RÉSULTATS FINAUX (moyenne ± std sur", len(seeds), "seeds)")
    print("=" * 90)
    header = f"{'Méthode':<18}" + "".join(f"{m[:12]:>15}" for m in metrics_keys)
    print(header); print("-" * len(header))

    summary = {}
    order = ['FineTuning', 'EWC', 'ReplayUniform', 'RiskAwareReplay', 'Joint']
    for method in order:
        runs = results[method]
        if not runs:
            continue
        summary[method] = {}
        row = f"{method:<18}"
        for mk in metrics_keys:
            vals = []
            for r in runs:
                v = r.get('metrics', {}).get(mk)
                if v is None:
                    v = r.get(mk)
                if v is not None:
                    vals.append(v)
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                summary[method][mk] = {'mean': m, 'std': s}
                row += f"{m:>8.3f}±{s:.2f}"
            else:
                row += f"{'—':>15}"
        print(row)
    print("-" * len(header))

    # test de significativité : RiskAwareReplay vs ReplayUniform
    try:
        from scipy import stats
        print("\n🔎 Significativité (Risk-Aware Replay vs Replay uniforme) :")
        for mk in metrics_keys:
            a = [r['metrics'].get(mk, r.get(mk)) for r in results['RiskAwareReplay']]
            b = [r['metrics'].get(mk, r.get(mk)) for r in results['ReplayUniform']]
            a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
            if len(a) > 1 and len(b) > 1:
                _, p = stats.ttest_ind(a, b)
                sig = "significatif ✅" if p < 0.05 else "non significatif ❌"
                print(f"   {mk:<20} p={p:.4f}  {sig}")
    except Exception as e:
        print(f"   (scipy indisponible: {e})")

    # ---- sauvegarde ----
    os.makedirs('./results/', exist_ok=True)
    out = {'protocol': 'temporal_chronological', 'dataset': 'CICIDS-2017',
           'seeds': seeds, 'n_seeds': len(seeds),
           'summary': summary, 'raw': results,
           'timestamp': datetime.now().isoformat()}
    with open('./results/paper_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print("\n✅ Résultats -> ./results/paper_results.json")
    print("\nLecture : Risk-Aware Replay doit battre ReplayUniform et EWC,")
    print("et s'approcher de Joint (borne haute). FineTuning = borne basse.")
    return out


if __name__ == "__main__":
    run_paper(n_seeds=5)