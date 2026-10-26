from config import Config

class AblNoRisk(Config):        # replay+smote+focal SANS risque
    USE_PROJECTION = False; USE_RISK_GUIDED = False
    USE_REPLAY = True; USE_ADAPTIVE_OVERSAMPLING = True; USE_FOCAL_LOSS = True

class AblRiskReplay(Config):    # + risque sur replay/focal (ta méthode)
    USE_PROJECTION = False; USE_RISK_GUIDED = True
    USE_REPLAY = True; USE_ADAPTIVE_OVERSAMPLING = True; USE_FOCAL_LOSS = True

class AblNoReplay(Config):      # ablation : sans replay du tout
    USE_PROJECTION = False; USE_RISK_GUIDED = True
    USE_REPLAY = False; USE_ADAPTIVE_OVERSAMPLING = True; USE_FOCAL_LOSS = True

class AblNoSmote(Config):       # ablation : sans SMOTE
    USE_PROJECTION = False; USE_RISK_GUIDED = True
    USE_REPLAY = True; USE_ADAPTIVE_OVERSAMPLING = False; USE_FOCAL_LOSS = True