"""
Configuration Baseline - GPM sans risk-guiding.
SEULE différence vs Config: USE_PROJECTION et USE_RISK_GUIDED.
Tout le reste (replay, SMOTE, focal, epochs, buffer) est STRICTEMENT identique.
"""

from config import Config

class ConfigBaseline(Config):
    # === Les DEUX seuls flags qui changent ===
    USE_RISK_GUIDED = False   # pas de pondération par risque
    USE_PROJECTION = False    # pas de projection de gradient

    # RIEN d'autre n'est modifié : REPLAY_BUFFER_SIZE, MAX_PER_CLASS_REPLAY,
    # GPM_K, GPM_SAMPLE_SIZE, RISK_MULTIPLIER restent ceux de Config.