# cl-nids-temporal-benchmark

Code and per-seed results for the paper:

> H. Moufid, M. El Ghazouani, M. A. El Kiram. *Replay Works; Severity Weighting Shows No Detectable Benefit: A Controlled Temporal Benchmark of Continual Learning for Network Intrusion Detection.* Submitted to Computers, Materials & Continua (CMC), 2026.

The benchmark compares six strategies on CIC-IDS2017 under a leakage-controlled temporal protocol:
a constant majority classifier, fine-tuning, Elastic Weight Consolidation (EWC), class-balanced uniform replay,
replay weighted by Adversarial Impact Metrics (AIM) severity scores, and joint training.
All trained methods share one backbone (MLP 78–512–256–128–15), one class-balanced focal loss and one training loop.

## Protocol

- One task per CIC-IDS2017 capture file (8 tasks, Monday to Friday afternoon).
- Within each day, the flows of each class are split chronologically into 60/20/20 train/validation/test.
- Features: signed log transform, then a `StandardScaler` fitted on the first day's training partition only.
- Five seeds: 42, 123, 456, 789, 1011.

## Installation

Python 3.10 (all results were produced on CPU with Python 3.10.11 and PyTorch 1.13.0):

```bash
pip install -r requirements.txt
```

## Data

Download the `MachineLearningCSV` files of CIC-IDS2017 from the Canadian Institute for Cybersecurity
(https://www.unb.ca/cic/datasets/ids-2017.html) and place the eight CSV files in:

```
data/CIC-IDS2017/MachineLearningCSV/
```

The dataset is not included in this repository.

## Reproducing the results

All experiments are run by `run_paper_v2.py`. Each command writes one JSON file with the configuration,
environment, per-class counts and per-seed results (metrics, per-class recall, retention matrices, buffer sizes).
Runs already present in the output file are skipped, so an interrupted command can be restarted.

Check the protocol without training (per-class counts and majority classifier):

```bash
python run_paper_v2.py --check
```

Main experiment (Tables 4–6, Figs. 2–4):

```bash
python run_paper_v2.py --set CB_BETA=0.999 --out results/main.json
```

Supplementary EWC runs (Table A1):

```bash
python run_paper_v2.py --methods EWC --set EWC_LAMBDA=10000 --out results/ewc_1e4.json
python run_paper_v2.py --methods EWC --set EWC_LAMBDA=100000 --out results/ewc_1e5.json
python run_paper_v2.py --methods EWC --set EWC_LAMBDA=1000000 --out results/ewc_1e6.json
```

Robustness checks (Table 7):

```bash
python run_paper_v2.py --methods ReplayRisk --set AIM_MODE=uniform --out results/abl_aim_uniform.json
python run_paper_v2.py --methods ReplayRisk --set AIM_MODE=inverted --out results/abl_aim_inverted.json
python run_paper_v2.py --methods ReplayRisk --set AIM_MODE=permuted --out results/abl_aim_permuted.json
python run_paper_v2.py --methods ReplayUniform ReplayRisk --set REPLAY_LAMBDA=0.25 --out results/abl_lam025.json
python run_paper_v2.py --methods ReplayUniform ReplayRisk --set REPLAY_LAMBDA=1.0 --out results/abl_lam1.json
python run_paper_v2.py --methods ReplayUniform ReplayRisk --set REPLAY_PER_CLASS=100 --out results/abl_q100.json
```

Implementation pitfalls P1–P4 (Table 7):

```bash
python run_paper_v2.py --methods ReplayUniform --set PITFALL_PERSISTENT_OPT=True --out results/pit_opt.json
python run_paper_v2.py --methods ReplayUniform --set PITFALL_CB_LEGACY=True --out results/pit_cb.json
python run_paper_v2.py --methods ReplayUniform --set PITFALL_REPLAY_CONCAT=True --out results/pit_replay.json
python run_paper_v2.py --methods ReplayUniform --split rows --out results/pit_split.json
```

`results/smoke.json` is a single-seed check (uniform replay, seed 42) of the main configuration.

### Note on the main configuration

`results/main.json` and `results/smoke.json` were produced with an earlier revision of `cl_methods.py`
(class-balancing constant β = 0.999 and early stopping on the validation cross-entropy).
The current code uses β = 0.9999 and early stopping on the class-averaged validation cross-entropy,
which is the configuration of all supplementary runs. `--set CB_BETA=0.999` restores β, but not the earlier
early-stopping criterion, so re-running the main experiment with the current code gives close but not identical values.

## Repository layout

| Path | Content |
|---|---|
| `run_paper_v2.py` | Experiment runner, summary tables and paired tests with Holm correction |
| `cl_methods.py` | Training loop, methods, replay reservoir, EWC, metrics and forgetting |
| `models/focal_loss.py` | Class-balanced focal loss |
| `data/generator.py` | Temporal task construction from the CIC-IDS2017 CSV files |
| `results/*.json` | Per-seed results used in the paper |

Other scripts (`main.py`, `run_paper.py`, `run_ablation.py`, `baselines.py`, `experiments/`, `training/`,
`evaluation/`, `config/`, `models/gpm.py`) belong to an earlier version of the project and are not used for the paper's results.
Some of them are still imported by the package `__init__` files.

## Citation

If you use this code, please cite the paper above (citation details will be added after publication).
