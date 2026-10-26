"""
Métriques détaillées par classe pour l'évaluation
VERSION PAPER-READY - Pour analyse fine des performances
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

def compute_per_class_metrics(labels, predictions, class_names=None):
    """
    Calcule les métriques détaillées par classe
    
    Args:
        labels: Vrais labels
        predictions: Prédictions du modèle
        class_names: Noms des classes (optionnel)
    
    Returns:
        DataFrame avec precision, recall, f1, support par classe
    """
    # Matrice de confusion
    cm = confusion_matrix(labels, predictions)
    n_classes = cm.shape[0]
    
    # Métriques par classe
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    metrics = []
    for i in range(n_classes):
        class_name = class_names[i] if class_names and i < len(class_names) else f'Class_{i}'
        metrics.append({
            'class': class_name,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        })
    
    df = pd.DataFrame(metrics)
    
    # Ajouter les moyennes pondérées
    weighted_avg = {
        'class': 'Weighted Avg',
        'precision': np.average(df['precision'], weights=df['support']),
        'recall': np.average(df['recall'], weights=df['support']),
        'f1': np.average(df['f1'], weights=df['support']),
        'support': int(df['support'].sum())
    }
    df = pd.concat([df, pd.DataFrame([weighted_avg])], ignore_index=True)
    
    return df

def get_weak_classes(df, threshold=0.7):
    """
    Identifie les classes avec F1 < threshold
    
    Args:
        df: DataFrame des métriques par classe
        threshold: Seuil F1 (défaut: 0.7)
    
    Returns:
        DataFrame des classes faibles
    """
    weak = df[df['f1'] < threshold]
    if len(weak) > 0:
        print("\n🔍 CLASSES FAIBLES (F1 < {:.2f}):".format(threshold))
        for _, row in weak.iterrows():
            print(f"   ⚠️ {row['class']}: F1={row['f1']:.3f}, Recall={row['recall']:.3f}, Support={int(row['support'])}")
    return weak

def get_confusion_matrix(labels, predictions, class_names=None):
    """
    Retourne la matrice de confusion avec les noms des classes
    """
    cm = confusion_matrix(labels, predictions)
    if class_names:
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    else:
        df_cm = pd.DataFrame(cm)
    return df_cm

def print_class_metrics(df):
    """
    Affiche les métriques par classe de manière lisible
    """
    print("\n" + "="*80)
    print("📊 MÉTRIQUES PAR CLASSE")
    print("="*80)
    print()
    
    # Séparer les classes des moyennes
    classes_df = df[df['class'] != 'Weighted Avg']
    avg_df = df[df['class'] == 'Weighted Avg']
    
    # Afficher par ordre de F1 décroissant
    classes_df = classes_df.sort_values('f1', ascending=False)
    
    print(f"{'Classe':<30} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Support':>10}")
    print("-"*75)
    
    for _, row in classes_df.iterrows():
        print(f"{row['class']:<30} | {row['precision']:>10.3f} | {row['recall']:>10.3f} | {row['f1']:>10.3f} | {int(row['support']):>10}")
    
    print("-"*75)
    print(f"{'Weighted Avg':<30} | {avg_df['precision'].values[0]:>10.3f} | {avg_df['recall'].values[0]:>10.3f} | {avg_df['f1'].values[0]:>10.3f} | {int(avg_df['support'].values[0]):>10}")
    print("="*80)