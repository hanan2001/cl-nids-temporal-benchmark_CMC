"""
Étude d'ablation RGGP — isole la contribution de chaque composant.

Objectif: montrer que le gain vient bien du RISK-GUIDING, et pas seulement de
l'empilement replay + SMOTE + focal loss sur GPM. Le passage clé à observer est
(2) -> (3) : ajout du risk-guiding à un pipeline déjà complet.

Les 4 configurations (définies dans config_ablation.py) :
    (1) AblGPMOnly              : GPM nu (projection, sans risk, sans replay/smote/focal)
    (2) AblPlusReplaySmoteFocal : + replay + SMOTE + focal   (SANS risk)
    (3) AblPlusRiskProj         : + risk-guiding             (pipeline complet)
    (4) AblFull                 : RGGP complet = Config

Chaque config est évaluée sur les mêmes seeds. Les métriques finales sont
calculées sur les TEST sets (via run_experiment de run_comparison.py).

Lancement :
    python run_ablation.py
ou, en réglant dataset/échantillon :
    python -c "from run_ablation import run_ablation; \
               run_ablation(dataset='cic-ids2017', use_small=True, n_samples=500000, n_seeds=3)"
"""

import os
import json
import numpy as np
from datetime import datetime

# run_experiment vient du fichier run_comparison.py déjà corrigé.
# Adapte le chemin d'import à ton arborescence si besoin :
#   - à plat        : from run_comparison import run_experiment
#   - en package    : from experiments.run_comparison import run_experiment
try:
    from experiments.run_comparison import run_experiment
except Exception:
    from run_comparison import run_experiment

from config_ablation import (
    AblGPMOnly,
    AblPlusReplaySmoteFocal,
    AblPlusRiskProj,
    AblFull,
)

# Ordre = ordre de lecture du tableau d'ablation.
# use_risk pilote get_weighted_samples de la GPM ; il DOIT suivre USE_RISK_GUIDED.
ABLATION_CONFIGS = [
    ("1_gpm_only",            AblGPMOnly,              False),
    ("2_replay_smote_focal",  AblPlusReplaySmoteFocal, False),
    ("3_plus_risk",           AblPlusRiskProj,         True),
    ("4_full_rggp",           AblFull,                 True),
]

# Métriques rapportées dans le tableau final.
REPORT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_weighted",
    "mcc",
    "forgetting_rate",
]


def _get_metric(result, key):
    """Récupère une métrique que run_experiment range dans 'metrics' ou à la racine."""
    if key in result.get("metrics", {}):
        return result["metrics"][key]
    if key in result:
        return result[key]
    return None


def run_ablation(dataset="cic-ids2017", use_small=True,
                 n_samples=500000, n_seeds=3, seeds=(42, 123, 456)):
    """Exécute les 4 configs d'ablation sur n_seeds seeds et agrège."""
    seeds = list(seeds)[:n_seeds]

    print("\n" + "=" * 70)
    print(f"🔬 ÉTUDE D'ABLATION | dataset={dataset} | seeds={seeds}")
    print("=" * 70)

    # results_by_config[nom_config] = liste de dicts résultats (un par seed)
    results_by_config = {name: [] for name, _, _ in ABLATION_CONFIGS}

    for name, config_cls, use_risk in ABLATION_CONFIGS:
        print("\n" + "#" * 70)
        print(f"# CONFIG {name}  (use_risk={use_risk})")
        print("#" * 70)

        for i, seed in enumerate(seeds):
            print(f"\n--- {name} | seed {seed} ({i+1}/{len(seeds)}) ---")
            # debug GPM seulement pour la 1re config, 1re seed (vérif projection active)
            gpm_debug = (name == "1_gpm_only" and i == 0)
            res = run_experiment(
                config_cls(), use_risk=use_risk, dataset=dataset,
                use_small=use_small, n_samples=n_samples,
                seed=seed, gpm_debug=gpm_debug,
            )
            if res is not None:
                results_by_config[name].append(res)

    # ------------------------------------------------------------------ #
    # Agrégation : moyenne ± écart-type par config et par métrique
    # ------------------------------------------------------------------ #
    summary = {}
    for name, _, _ in ABLATION_CONFIGS:
        runs = results_by_config[name]
        if not runs:
            continue
        summary[name] = {}
        for metric in REPORT_METRICS:
            vals = [_get_metric(r, metric) for r in runs]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[name][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
        # trace GPM pour vérifier que la projection tourne dans les configs projetées
        proj = [r.get("gpm_stats", {}).get("total_projections", 0) for r in runs]
        summary[name]["gpm_total_projections_mean"] = float(np.mean(proj)) if proj else 0.0

    # ------------------------------------------------------------------ #
    # Tableau lisible
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("📊 TABLEAU D'ABLATION (moyenne ± écart-type sur les seeds)")
    print("=" * 70)
    header = f"{'Config':<24}" + "".join(f"{m[:10]:>14}" for m in REPORT_METRICS)
    print(header)
    print("-" * len(header))
    for name, _, _ in ABLATION_CONFIGS:
        if name not in summary:
            continue
        row = f"{name:<24}"
        for metric in REPORT_METRICS:
            cell = summary[name].get(metric)
            row += f"{cell['mean']:>8.3f}±{cell['std']:.2f}" if cell else f"{'—':>14}"
        print(row)
    print("-" * len(header))
    print("Lecture clé : comparer la ligne 2 (sans risk) à la ligne 3 (avec risk).")
    print("Le gain 2->3 mesure l'apport PROPRE du risk-guiding.")

    # contrôle de sanité projection
    print("\nProjections GPM moyennes par config (doit être > 0 quand USE_PROJECTION) :")
    for name, _, _ in ABLATION_CONFIGS:
        if name in summary:
            print(f"   {name:<24} {summary[name]['gpm_total_projections_mean']:.0f}")

    # ------------------------------------------------------------------ #
    # Sauvegarde
    # ------------------------------------------------------------------ #
    os.makedirs("./results/", exist_ok=True)
    suffix = f"_{dataset}" + ("_small" if use_small else "")
    out = {
        "dataset": dataset,
        "use_small": use_small,
        "seeds": seeds,
        "configs": [name for name, _, _ in ABLATION_CONFIGS],
        "summary": summary,
        "raw": results_by_config,
        "timestamp": datetime.now().isoformat(),
    }
    path = f"./results/ablation{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✅ Ablation sauvegardée -> {path}")
    return out


if __name__ == "__main__":
    run_ablation(dataset="cic-ids2017", use_small=True,
                 n_samples=500000, n_seeds=3)