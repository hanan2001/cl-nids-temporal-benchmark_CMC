"""
Configuration globale — VERSION PAPER IJACSA (Risk-Aware Replay).

CHANGEMENT CLÉ vs versions précédentes :
- La PROJECTION DE GRADIENT (GPM) est ABANDONNÉE (USE_PROJECTION = False).
  La contribution du papier n'est plus la projection (qui saturait) mais le
  REPLAY PONDÉRÉ PAR LE RISQUE sous protocole temporel.
- USE_RISK_GUIDED pilote désormais le REPLAY et la FOCAL LOSS (pondération par
  la sévérité des attaques), pas la projection.
"""

import torch


class Config:
    # ============ ARCHITECTURE ============
    INPUT_SIZE = 78
    HIDDEN_SIZES = [512, 256, 128]
    OUTPUT_SIZE = 15

    # ============ ENTRAÎNEMENT ============
    MAX_EPOCHS_PER_TASK = 80
    BATCH_SIZE = 512

    LEARNING_RATE = 1e-4        # au lieu de 5e-4
    WEIGHT_DECAY = 1e-5

    # ============ PROJECTION (ABANDONNÉE) ============
    USE_PROJECTION = False          # <-- plus de GPM

    # ============ EARLY STOPPING ============
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 0.0005

    # ============ TÂCHES ============
    NUM_TASKS = 8                   # protocole temporel = 8 jours
    TASK_MODE = 'temporal'

    # ============ FOCAL LOSS ============
    USE_FOCAL_LOSS = True
    USE_CLASS_BALANCED = True
    # Découplage du risque (Hypothèse 1)
    RISK_ON_BUFFER = False    # risque pilote la sélection du buffer
    RISK_ON_LOSS = True     # risque pilote la focal loss
    FOCAL_GAMMA = 1.0          

    # ============ OVERSAMPLING ============
    USE_ADAPTIVE_OVERSAMPLING = False
    RARE_CLASS_THRESHOLD = 1000
    TARGET_SAMPLES = 1000        # au lieu de 3000
    SMOTE_K_NEIGHBORS = 7

    # ============ REPLAY BUFFER (cœur de la méthode) ============
    USE_REPLAY = True
    REPLAY_BUFFER_SIZE = 50000
    REPLAY_RATIO = 0.5
    MAX_PER_CLASS_REPLAY = 500

    # ============ RISK-GUIDING (pilote replay + focal) ============
    USE_RISK_GUIDED = True
    USE_ADAPTIVE_RISK = True
    RISK_WEIGHT_THRESHOLD = 0.4
    RISK_MULTIPLIER = 3.0

    # ============ GPM (inactif, conservé pour compat imports) ============
    GPM_K = 6
    GPM_SAMPLE_SIZE = 500

    # ============ EWC (baseline) ============
    EWC_LAMBDA = 1000.0             # force de la régularisation EWC

    # ============ DEVICE ============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============ CHEMINS ============
    DATA_PATH = './data/datasets/'
    RESULTS_PATH = './results/'
    MODELS_PATH = './results/models/'