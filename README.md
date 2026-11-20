# cl-nids-temporal-benchmark

Code and per-seed results for the paper:

> H. Moufid, M. El Ghazouani, M. A. El Kiram. *Replay Works; Severity Weighting Shows No Detectable Benefit: A Controlled Temporal Benchmark of Continual Learning for Network Intrusion Detection.* Submitted to Computers, Materials & Continua (CMC), 2026; revised version (round 1) in preparation.

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

## Reproducing the results of the submitted version

Table and figure numbers in this section refer to the submitted manuscript; the revised experiments are described in
the section *Revision (CMC, round 1)* below.

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

## Revision (CMC, round 1)

The revision adds a timestamp-verified protocol, sensitivity protocols, a second dataset and new controls.
All new options are off by default, so the commands above reproduce the submitted results unchanged
(`revision/check_generator.py` and `revision/check_fastload.py` check this; a rerun of the main configuration
gives results identical to `results/main.json`).

Additional data:
- `data/CIC-IDS2017/TrafficLabelling/`: the GeneratedLabelledFlows CSV files of CIC-IDS2017 (same rows as
  MachineLearningCSV, plus the flow timestamps; alignment checked by `revision/check_align.py`).
- `data/CSE-CIC-IDS2018/`: the ten "Processed Traffic Data for ML Algorithms" CSV files
  (`s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/`).

New options of `run_paper_v2.py`:

| Option | Meaning |
|---|---|
| `--order time` | rows of each day sorted by flow timestamp (revised main protocol) |
| `--split time_cut` | shared whole-minute chronological cutoffs across classes |
| `--dedup eval` | remove val/test flows whose raw feature vector occurs in any training partition |
| `--dataset cic2018` | CSE-CIC-IDS2018, one task per day, subsampled (see `data/cic2018.py`) |
| `--set RISK_ON_CAPACITY/RISK_ON_SAMPLING/RISK_ON_LOSS=False`, `RISK_BUDGET_MATCHED=True` | severity decomposition |
| `--set SELECT_CRITERION=retention` | retention-aware checkpoint selection |
| `--set PITFALL_PERSISTENT_ADAM=True` / `PITFALL_PERSISTENT_SCHED=True` | P1 decomposed |
| `--set EWC_FISHER_MODE=sampled_example EWC_FISHER_BALANCED=True EWC_DIAG=True` | EWC diagnostics |
| `--set CB_BETA=0.999 ES_CRITERION=plain` | configuration of the submitted main experiment |

Pipeline:

```bash
python revision/audit_protocol.py file     # timestamp and duplicate audit, submitted order
python revision/audit_protocol.py time     # same, revised order (Tables A2, A3)
python revision/jobs.py --workers 5 --threads 2   # all revision runs -> results/rev/<experiment>/
python revision/other_classifiers.py       # random forest / gradient boosting diagnostic (Table A5)
python revision/analyze.py                 # -> results/rev/summary.json
python revision/figures.py                 # -> results/rev/figures/
```

`revision/jobs.py --list` shows the remaining runs; the queue can be stopped by creating `results/rev/STOP`
and restarted at any time (finished runs are skipped).

Example: one run of the revised main protocol (uniform replay, seed 42):

```bash
python run_paper_v2.py --order time --methods ReplayUniform --seeds 42 --out results/rev/example.json
```

Results of the revision:

| Path | Content |
|---|---|
| `results/rev/<experiment>/<Method>_s<seed>.json` | one file per run (configuration, metrics, per-class recall, retention matrix) |
| `results/rev/summary.json` | all tables of the revised paper: means, paired differences, confidence intervals, Holm and TOST tests |
| `results/rev/figures/` | figures of the revised paper |
| `results/audit_protocol*.{json,txt}` | timestamp-order and duplicate audit (submitted and revised order) |
| `results/other_classifiers.json` | random forest and gradient boosting diagnostic |

`revision/_gen_old.py` and `revision/_cl_old.py` are copies of the submitted `data/generator.py` and `cl_methods.py`,
used only by `revision/check_generator.py` and `revision/check_fastload.py` to verify that the default options
reproduce the submitted code exactly.
