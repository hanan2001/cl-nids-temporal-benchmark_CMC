import numpy as np
from training.continual_learner import ReplayBuffer

rng = np.random.RandomState(0)
risk = {0: 0.1, 7: 0.6, 11: 0.8, 3: 0.7}   # BENIGN faible risque, attaques élevé
buf = ReplayBuffer(risk_mapping=risk, max_per_class=500)

# simuler l'arrivée massive de BENIGN puis quelques attaques
buf.add_samples(rng.randn(300000, 10), np.zeros(300000, int))   # BENIGN massif
buf.add_samples(rng.randn(8000, 10), np.full(8000, 7))          # attaque
buf.add_samples(rng.randn(3000, 10), np.full(3000, 11))         # attaque rare
buf.add_samples(rng.randn(500, 10), np.full(500, 3))            # très rare

print("Contenu du buffer par classe:", buf.get_stats()['classes'])
Xs, ys = buf.sample(2000)
u, c = np.unique(ys, return_counts=True)
print("Échantillon de replay par classe:", dict(zip(u.tolist(), c.tolist())))