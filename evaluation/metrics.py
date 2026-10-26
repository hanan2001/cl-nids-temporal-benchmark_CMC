"""
Métriques d'évaluation avancées pour l'apprentissage continu
VERSION PAPER-READY - Balanced Accuracy + MCC standard
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
import time


def compute_f1_score(labels, predictions, average='weighted'):
    """Calcule le F1-score"""
    return f1_score(labels, predictions, average=average)


def compute_balanced_accuracy(labels, predictions):
    """Calcule la Balanced Accuracy (moyenne des recalls par classe)"""
    return balanced_accuracy_score(labels, predictions)


def compute_mcc(labels, predictions):
    """Calcule le MCC standard (scikit-learn gère déjà le multi-classe)"""
    return matthews_corrcoef(labels, predictions)


def compute_mcc_weighted(labels, predictions):
    """🔧 DEPRECATED - Utiliser compute_mcc à la place"""
    return compute_mcc(labels, predictions)


def compute_forgetting_rate_simple(accuracies):
    """Calcule le taux d'oubli simple"""
    if len(accuracies) <= 1:
        return 0.0
    
    forgetting_rates = []
    for i in range(len(accuracies) - 1):
        for j in range(i + 1, len(accuracies)):
            forgetting = accuracies[i] - accuracies[j]
            if forgetting > 0:
                forgetting_rates.append(forgetting)
    
    return np.mean(forgetting_rates) if forgetting_rates else 0.0


def compute_forgetting_rate_matrix(accuracies_history):
    """
    Calcule le taux d'oubli à partir de la matrice des accuracies
    """
    n_tasks = len(accuracies_history)
    if n_tasks < 2:
        return 0.0
    
    forgetting_rates = []
    
    for i in range(n_tasks - 1):
        if i < len(accuracies_history[i]):
            initial_acc = accuracies_history[i][i]
        else:
            initial_acc = 0.0
        
        if i < len(accuracies_history[-1]):
            final_acc = accuracies_history[-1][i]
        else:
            final_acc = 0.0
        
        forgetting = initial_acc - final_acc
        if forgetting > 0:
            forgetting_rates.append(forgetting)
    
    return np.mean(forgetting_rates) if forgetting_rates else 0.0


def compute_plasticity_stability_metrics(accuracies_history):
    """Calcule les métriques de plasticité et stabilité"""
    n_tasks = len(accuracies_history)
    if n_tasks < 2:
        return {'plasticity': 0.0, 'stability': 0.0, 'tradeoff': 0.0}
    
    plasticity = accuracies_history[-1][-1] if n_tasks > 0 and len(accuracies_history[-1]) > 0 else 0.0
    
    stability = 0.0
    if n_tasks > 1 and len(accuracies_history[-1]) > 1:
        stability = np.mean(accuracies_history[-1][:-1])
    
    return {
        'plasticity': plasticity,
        'stability': stability,
        'tradeoff': plasticity - stability
    }


def compute_risk_weighted_accuracy(labels, predictions, risk_mapping):
    """Calcule l'accuracy pondérée par le risque"""
    total_weight = 0
    correct_weight = 0
    
    for label, pred in zip(labels, predictions):
        risk = risk_mapping.get(label, 0.5)
        total_weight += risk
        if label == pred:
            correct_weight += risk
    
    return correct_weight / total_weight if total_weight > 0 else 0.0


def compute_all_metrics(labels, predictions, risk_mapping=None):
    """Calcule toutes les métriques standard"""
    metrics = {
        'accuracy': np.mean(np.array(labels) == np.array(predictions)),
        'balanced_accuracy': compute_balanced_accuracy(labels, predictions),
        'f1_weighted': compute_f1_score(labels, predictions, 'weighted'),
        'f1_macro': compute_f1_score(labels, predictions, 'macro'),
        'mcc': compute_mcc(labels, predictions)
    }
    
    if risk_mapping:
        metrics['risk_weighted_accuracy'] = compute_risk_weighted_accuracy(
            labels, predictions, risk_mapping
        )
    
    return metrics


def compute_mcc_by_risk_group(labels, predictions, risk_mapping, threshold=0.7):
    """
    H2 : MCC calcule separement sur les classes HAUT risque (>= threshold)
    et BAS risque (< threshold). Permet de voir si le risk-guiding aide sur
    les classes qu'il cible, meme si le MCC global n'en profite pas.
    Retourne aussi les effectifs pour interpreter la fiabilite du sous-MCC.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    high = [c for c, r in risk_mapping.items() if r >= threshold]
    low = [c for c, r in risk_mapping.items() if r < threshold]

    def _sub_mcc(classes):
        mask = np.isin(labels, classes)
        if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
            return None
        return float(matthews_corrcoef(labels[mask], predictions[mask]))

    return {
        'mcc_high_risk': _sub_mcc(high),
        'mcc_low_risk': _sub_mcc(low),
        'n_high': int(np.isin(labels, high).sum()),
        'n_low': int(np.isin(labels, low).sum()),
        'high_classes': high,
        'low_classes': low,
    }


class Timer:
    """Chronomètre pour mesurer le temps d'exécution"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def elapsed(self):
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time