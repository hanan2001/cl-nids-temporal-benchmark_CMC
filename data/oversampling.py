"""
Sur-échantillonnage adaptatif pour les classes rares
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

class AdaptiveOversampler:
    """
    Sur-échantillonnage adaptatif pour les classes rares
    Utilise SMOTE pour les classes avec peu d'échantillons
    """
    def __init__(self, threshold=100, target_samples=200, k_neighbors=5):
        """
        Args:
            threshold: seuil pour considérer une classe comme rare
            target_samples: nombre cible d'échantillons pour les classes rares
            k_neighbors: nombre de voisins pour SMOTE
        """
        self.threshold = threshold
        self.target_samples = target_samples
        self.k_neighbors = k_neighbors
    
    def get_class_counts(self, y):
        """Compte les échantillons par classe"""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    def is_rare_class(self, count):
        """Détermine si une classe est rare"""
        return count < self.threshold
    
    def smote(self, X, y, target_count):
        """
        Génère des échantillons synthétiques avec SMOTE
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        if n_samples == 0:
            return X, y
        
        if n_samples == 1:
            # Si un seul échantillon, dupliquer avec bruit
            X_new = np.repeat(X, target_count, axis=0)
            X_new += np.random.randn(*X_new.shape) * 0.01
            y_new = np.repeat(y, target_count)
            return X_new, y_new
        
        # Créer des échantillons synthétiques
        n_to_generate = target_count - n_samples
        if n_to_generate <= 0:
            return X, y
        
        # Utiliser les k plus proches voisins
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, n_samples))
        nn.fit(X)
        
        # Générer des échantillons synthétiques
        synthetic_X = []
        synthetic_y = []
        
        for i in range(n_to_generate):
            # Sélectionner un échantillon aléatoire
            idx = np.random.randint(0, n_samples)
            # Trouver ses voisins
            distances, indices = nn.kneighbors(X[idx].reshape(1, -1))
            # Sélectionner un voisin aléatoire
            neighbor_idx = indices[0][np.random.randint(1, len(indices[0]))]
            
            # Interpoler entre les deux
            diff = X[neighbor_idx] - X[idx]
            alpha = np.random.random()
            synthetic = X[idx] + alpha * diff
            
            synthetic_X.append(synthetic)
            synthetic_y.append(y[idx])
        
        # Combiner
        X_new = np.vstack([X, np.array(synthetic_X)])
        y_new = np.concatenate([y, np.array(synthetic_y)])
        
        return X_new, y_new
    
    def balance_task(self, X, y):
        """
        Équilibre une tâche en sur-échantillonnant les classes rares
        """
        class_counts = self.get_class_counts(y)
        rare_classes = {cls: count for cls, count in class_counts.items() 
                       if self.is_rare_class(count)}
        
        if not rare_classes:
            return X, y
        
        print(f"   📊 Classes rares détectées: {len(rare_classes)}")
        for cls, count in rare_classes.items():
            print(f"      Classe {cls}: {count} échantillons → {self.target_samples}")
        
        # Sur-échantillonner chaque classe rare
        all_X = []
        all_y = []
        
        for cls in np.unique(y):
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            
            if cls in rare_classes:
                target = max(self.target_samples, len(X_cls) * 2)
                X_cls, y_cls = self.smote(X_cls, y_cls, target)
                print(f"      ✅ Classe {cls}: {len(X_cls)} échantillons après SMOTE")
            
            all_X.append(X_cls)
            all_y.append(y_cls)
        
        X_balanced = np.vstack(all_X)
        y_balanced = np.concatenate(all_y)
        
        return X_balanced, y_balanced