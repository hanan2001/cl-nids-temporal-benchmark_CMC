"""
Comparaison RGGP vs Baseline — VERSION PAPER-READY.

CORRECTIONS vs version précédente:
- Consomme le nouveau format de generator.create_tasks : chaque tâche est un
  dict {'train','val','test'} (split 60/20/20, seed variable).
- L'entraînement voit train+val ; les métriques finales et le forgetting sont
  calculés SUR LES TEST SETS tenus à l'écart.
- Seed propagée jusqu'au split (create_tasks(..., seed=seed)).
- Debug GPM activé sur la 1re seed RGGP pour vérifier que la projection s'active.
"""

import torch
import numpy as np
import json
import os
import random
from datetime import datetime

from config import Config
from config.config_baseline import ConfigBaseline
from data.generator import generate_complex_data, create_tasks
from data.risk_mapping import get_risk_scores
from data.real_loader import load_cic_ids2017_all_files, load_cic_ids2017_small
from evaluation.per_class_metrics import compute_per_class_metrics, print_class_metrics, get_weak_classes
from torch.utils.data import TensorDataset, DataLoader
from models import CyberNetwork, GradientProjectionMemory
from training import ContinualLearner
from evaluation.metrics import (
    compute_all_metrics,
    compute_plasticity_stability_metrics,
    Timer,
    compute_forgetting_rate_matrix,
)
from evaluation.visualizer import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(config, use_risk=True, dataset='synthetic',
                   use_small=False, n_samples=50000, seed=42, gpm_debug=False):
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"🚀 {'RGGP' if use_risk else 'BASELINE'} | dataset={dataset} | seed={seed}")
    print(f"{'='*60}")

    device = config.DEVICE
    timer = Timer()

    # ============================================================
    # 1. Charger les données brutes
    # ============================================================
    if dataset == 'synthetic':
        X, y = generate_complex_data(n_samples=10000, n_features=32,
                                     n_classes=config.OUTPUT_SIZE)
        class_names = [f'Class_{i}' for i in range(config.OUTPUT_SIZE)]
    elif dataset == 'cic-ids2017':
        data_path = './data/CIC-IDS2017/MachineLearningCSV/'
        if use_small:
            X, y, class_names = load_cic_ids2017_small(data_path, n_samples, seed=seed)
        else:
            X, y, class_names = load_cic_ids2017_all_files(data_path)
        if X is None:
            print("   ❌ Erreur de chargement")
            return None
    else:
        from data.real_loader import load_dataset_by_name
        res = load_dataset_by_name(dataset)
        if res is None or len(res) < 3:
            return None
        X, y, class_names = res[0], res[1], res[2]

    # ============================================================
    # 2. Créer les tâches (dict train/val/test), seed propagée
    # ============================================================
    tasks, class_names = create_tasks(
        X, y, num_tasks=config.NUM_TASKS, mode='class_incremental',
        data_path='./data/CIC-IDS2017/MachineLearningCSV/', seed=seed)
    if len(tasks) == 0:
        print("   ❌ Aucune tâche créée")
        return None

    # ============================================================
    # 3. Détecter les classes (sur les trois splits)
    # ============================================================
    all_y = []
    for task in tasks:
        for split in ('train', 'val', 'test'):
            all_y.extend(task[split].tensors[1].numpy())
    unique_classes = np.unique(all_y)
    actual_output_size = len(unique_classes)
    print(f"   ✅ {actual_output_size} classes détectées")

    # ============================================================
    # 4. Remap 0..n-1 sur les trois splits, noms conservés
    # ============================================================
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(all_y)
    original_classes = le.classes_

    new_to_original = {}
    for orig_idx, orig_class in enumerate(original_classes):
        if class_names is not None and orig_class < len(class_names):
            new_to_original[orig_idx] = class_names[orig_class]
        else:
            new_to_original[orig_idx] = f'Class_{orig_class}'

    new_tasks = []
    for task in tasks:
        nt = {}
        for split in ('train', 'val', 'test'):
            Xs = task[split].tensors[0].numpy()
            ys = le.transform(task[split].tensors[1].numpy())
            nt[split] = TensorDataset(torch.FloatTensor(Xs), torch.LongTensor(ys))
        new_tasks.append(nt)
    tasks = new_tasks

    class_names = [new_to_original.get(i, f'Class_{i}') for i in range(actual_output_size)]
    print("   📋 Classes:", {i: n for i, n in enumerate(class_names)})

    # ============================================================
    # 5. Modèle
    # ============================================================
    hidden_sizes = getattr(config, 'HIDDEN_SIZES', [128, 64, 32])
    input_size = tasks[0]['train'].tensors[0].shape[1]
    model = CyberNetwork(input_size=input_size, hidden_sizes=hidden_sizes,
                         output_size=actual_output_size).to(device)
    print(f"   🧠 {sum(p.numel() for p in model.parameters()):,} paramètres | "
          f"in={input_size} out={actual_output_size}")

    # ============================================================
    # 6. GPM (+ debug éventuel)
    # ============================================================
    gpm = GradientProjectionMemory(
        k=config.GPM_K, sample_size=config.GPM_SAMPLE_SIZE, use_risk=use_risk,
        risk_threshold=getattr(config, 'RISK_WEIGHT_THRESHOLD', 0.5),
        risk_multiplier=getattr(config, 'RISK_MULTIPLIER', 2.0),
        device=device, use_adaptive_risk=getattr(config, 'USE_ADAPTIVE_RISK', True))
    gpm.debug = gpm_debug

    # ============================================================
    # 7. Learner + mapping de risque
    # ============================================================
    learner = ContinualLearner(model, gpm, config)
    risk_mapping = get_risk_scores(class_names)

    # ============================================================
    # 8. Entraînement séquentiel — éval sur TEST
    # ============================================================
    timer.start()
    all_accuracies = []
    all_predictions = []
    all_labels = []
    accuracies_history = []

    test_tasks = [t['test'] for t in tasks]

    for task_id, task in enumerate(tasks):
        print(f"\n📝 TÂCHE {task_id+1}/{len(tasks)}")
        acc = learner.train_task(
            train_dataset=task['train'], val_dataset=task['val'],
            task_id=task_id, risk_mapping=risk_mapping, all_test_tasks=test_tasks)
        all_accuracies.append(acc)

        task_acc = []
        for past_id in range(task_id + 1):
            loader = DataLoader(test_tasks[past_id],
                                batch_size=config.BATCH_SIZE, shuffle=False)
            pa, preds, labels = learner.trainer.evaluate(loader)
            task_acc.append(pa)
            if task_id == len(tasks) - 1:
                all_predictions.extend(preds)
                all_labels.extend(labels)
        accuracies_history.append(task_acc)
        print(f"   Test accuracies: {[f'{a:.4f}' for a in task_acc]}")

    timer.stop()
    training_time = timer.elapsed()

    # ============================================================
    # 9. Sauvegarde du modèle
    # ============================================================
    os.makedirs('./results/models/', exist_ok=True)
    model_path = f'./results/models/{dataset}_seed{seed}_{"rggp" if use_risk else "baseline"}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'class_names': class_names, 'seed': seed,
                'risk_mapping': risk_mapping}, model_path)

    # ============================================================
    # 10. Métriques finales (sur TEST agrégé)
    # ============================================================
    if all_predictions and all_labels:
        per_class_df = compute_per_class_metrics(all_labels, all_predictions, class_names)
        print_class_metrics(per_class_df)
        weak_classes = get_weak_classes(per_class_df, threshold=0.7)

        os.makedirs('./results/per_class/', exist_ok=True)
        per_class_df.to_csv(
            f'./results/per_class/{dataset}_seed{seed}_'
            f'{"rggp" if use_risk else "baseline"}.csv', index=False)

        metrics = compute_all_metrics(all_labels, all_predictions, risk_mapping)

        from evaluation.visualizer import (plot_confusion_matrix,
                                            plot_per_class_metrics,
                                            plot_risk_vs_performance)
        os.makedirs('./results/figures/', exist_ok=True)
        tag = f'{dataset}_seed{seed}_{"rggp" if use_risk else "baseline"}'
        for fn, args in [
            (plot_confusion_matrix, (all_labels, all_predictions)),
            (plot_per_class_metrics, (per_class_df,)),
            (plot_risk_vs_performance, (per_class_df, risk_mapping, class_names)),
        ]:
            try:
                if fn is plot_confusion_matrix:
                    fn(*args, class_names=class_names,
                       save_path=f'./results/figures/cm_{tag}.png')
                elif fn is plot_per_class_metrics:
                    fn(*args, save_path=f'./results/figures/pc_{tag}.png')
                else:
                    fn(*args, save_path=f'./results/figures/rvp_{tag}.png')
            except Exception as e:
                print(f"   ⚠️ Figure {fn.__name__}: {e}")
    else:
        metrics = {'accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
                   'f1_weighted': 0.0, 'f1_macro': 0.0,
                   'balanced_accuracy': 0.0, 'mcc': 0.0}
        per_class_df = None
        weak_classes = []

    # ============================================================
    # 11. Forgetting / plasticité (sur historique TEST)
    # ============================================================
    if len(learner.accuracies_history) > 1:
        forgetting_rate = compute_forgetting_rate_matrix(learner.accuracies_history)
        ps_metrics = compute_plasticity_stability_metrics(learner.accuracies_history)
    else:
        forgetting_rate = 0.0
        ps_metrics = {'plasticity': all_accuracies[-1] if all_accuracies else 0.0,
                      'stability': np.mean(all_accuracies[:-1]) if len(all_accuracies) > 1 else 0.0,
                      'tradeoff': 0.0}

    if learner.accuracies_history:
        avg_incremental_acc = np.mean([sum(h) / len(h) for h in learner.accuracies_history])
    else:
        avg_incremental_acc = 0.0

    # ============================================================
    # 12. Dictionnaire résultats
    # ============================================================
    results = {
        'method': 'RGGP' if use_risk else 'Baseline',
        'dataset': dataset,
        'n_classes': actual_output_size,
        'class_names': list(class_names),
        'use_small': use_small,
        'seed': seed,
        'accuracies': all_accuracies,
        'accuracies_history': accuracies_history,
        'forgetting_rate': forgetting_rate,
        'avg_incremental_accuracy': avg_incremental_acc,
        'training_time': training_time,
        'metrics': metrics,
        'plasticity': ps_metrics['plasticity'],
        'stability': ps_metrics['stability'],
        'tradeoff': ps_metrics['tradeoff'],
        'avg_epochs': learner.epochs_per_task,
        'gpm_stats': gpm.get_stats(),
    }
    if per_class_df is not None:
        results['per_class_metrics'] = per_class_df.to_dict('records')
        results['weak_classes'] = weak_classes.to_dict('records') if len(weak_classes) > 0 else []

    print(f"\n📈 acc={metrics['accuracy']:.4f} "
          f"bal_acc={metrics.get('balanced_accuracy',0):.4f} "
          f"f1={metrics['f1_weighted']:.4f} mcc={metrics.get('mcc',0):.4f} "
          f"forget={forgetting_rate:.4f} | GPM proj={gpm.get_stats()['total_projections']}")
    return results


def compare_rggp_baseline(dataset='cic-ids2017', use_small=False, n_samples=500000, n_seeds=5):
    print("\n" + "=" * 60)
    print(f"🔬 COMPARAISON RGGP vs BASELINE | dataset={dataset} | seeds={n_seeds}")
    print("=" * 60)
    seeds = [42, 123, 456, 789, 1011]
    rggp_results, baseline_results = [], []

    for i, seed in enumerate(seeds[:n_seeds]):
        print(f"\n{'='*60}\n🎲 SEED {seed} ({i+1}/{n_seeds})\n{'='*60}")

        # debug GPM uniquement sur la 1re seed RGGP
        rggp_result = run_experiment(Config(), use_risk=True, dataset=dataset,
                                     use_small=use_small, n_samples=n_samples,
                                     seed=seed, gpm_debug=(i == 0))
        if rggp_result is not None:
            rggp_results.append(rggp_result)

        baseline_result = run_experiment(ConfigBaseline(), use_risk=False, dataset=dataset,
                                         use_small=use_small, n_samples=n_samples, seed=seed)
        if baseline_result is not None:
            baseline_results.append(baseline_result)

    if not rggp_results or not baseline_results:
        print("❌ Aucun résultat")
        return None

    # ============================================================
    # Statistiques + tests
    # ============================================================
    print("\n" + "=" * 60)
    print(f"📊 MOYENNES SUR {len(rggp_results)} SEEDS")
    print("=" * 60)

    metrics_keys = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'mcc',
                    'risk_weighted_accuracy', 'forgetting_rate',
                    'avg_incremental_accuracy', 'plasticity', 'stability']
    summary = {}
    for metric in metrics_keys:
        def _collect(rs):
            out = []
            for r in rs:
                if metric in r.get('metrics', {}):
                    out.append(r['metrics'][metric])
                elif metric in r:
                    out.append(r[metric])
            return out
        rggp_vals, baseline_vals = _collect(rggp_results), _collect(baseline_results)
        if not (rggp_vals and baseline_vals):
            continue
        rggp_mean, rggp_std = np.mean(rggp_vals), np.std(rggp_vals)
        base_mean, base_std = np.mean(baseline_vals), np.std(baseline_vals)
        try:
            from scipy import stats
            _, p_value = stats.ttest_ind(rggp_vals, baseline_vals)
            significant = p_value < 0.05
        except Exception:
            p_value, significant = 1.0, False
        if 'forgetting' in metric:
            improvement = (base_mean - rggp_mean) / (base_mean + 1e-8) * 100
            arrow = '✅' if rggp_mean < base_mean else '❌'
        else:
            improvement = (rggp_mean - base_mean) / (base_mean + 1e-8) * 100
            arrow = '✅' if rggp_mean > base_mean else '❌'
        summary[metric] = {'rggp_mean': rggp_mean, 'rggp_std': rggp_std,
                           'baseline_mean': base_mean, 'baseline_std': base_std,
                           'improvement': improvement, 'arrow': arrow,
                           'p_value': p_value, 'significant': significant}
        print(f"\n{metric.replace('_',' ').title()}:")
        print(f"  RGGP:     {rggp_mean:.4f} ± {rggp_std:.4f}")
        print(f"  Baseline: {base_mean:.4f} ± {base_std:.4f}")
        print(f"  {arrow} {improvement:+.1f}% (p={p_value:.4f})")

    # ============================================================
    # Figures récap
    # ============================================================
    os.makedirs('./results/figures/', exist_ok=True)
    suffix = f"_{dataset}" + ("_small" if use_small else "")
    try:
        r0, b0 = rggp_results[0], baseline_results[0]
        save_figure(plot_comparison([{'accuracy': a} for a in r0['accuracies']],
                                    [{'accuracy': a} for a in b0['accuracies']],
                                    f"RGGP vs Baseline ({dataset})"),
                    f"comparison{suffix}.png")
        save_figure(plot_forgetting_curves(r0['accuracies_history'],
                                           f"Forgetting - RGGP ({dataset})"),
                    f"forgetting{suffix}.png")
        save_figure(plot_task_accuracies(r0['accuracies'],
                                         f"Task Accuracies - RGGP ({dataset})"),
                    f"task_accuracies{suffix}.png")
    except Exception as e:
        print(f"   ⚠️ Figures récap: {e}")

    # ============================================================
    # Sauvegarde JSON
    # ============================================================
    os.makedirs('./results/', exist_ok=True)
    comparison_results = {'dataset': dataset, 'use_small': use_small,
                          'n_seeds': len(rggp_results), 'seeds': seeds[:n_seeds],
                          'rggp': rggp_results, 'baseline': baseline_results,
                          'summary': summary, 'timestamp': datetime.now().isoformat()}
    with open(f'./results/comparison{suffix}.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"\n✅ Résultats -> ./results/comparison{suffix}.json")
    return comparison_results


if __name__ == "__main__":
    compare_rggp_baseline(dataset='cic-ids2017', use_small=True,
                          n_samples=500000, n_seeds=5)