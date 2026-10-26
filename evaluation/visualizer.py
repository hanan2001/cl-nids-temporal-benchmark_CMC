"""
Visualisations pour l'article
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_comparison(rggp_results, baseline_results, title="RGGP vs Baseline"):
    tasks = [f"Task {i+1}" for i in range(len(rggp_results))]
    rggp_acc = [r['accuracy'] for r in rggp_results]
    baseline_acc = [r['accuracy'] for r in baseline_results]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, rggp_acc, width, label='RGGP', color='#2E86AB')
    bars2 = plt.bar(x + width/2, baseline_acc, width, label='Baseline', color='#A23B72')
    
    plt.xlabel('Tasks', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x, tasks)
    plt.legend()
    plt.ylim(0, 1.0)
    
    for bar, acc in zip(bars1, rggp_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, acc in zip(bars2, baseline_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return plt

def plot_task_accuracies(accuracies, title="Task Accuracies"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    return plt

def plot_forgetting_curves(accuracies_history, title="Forgetting Curves"):
    plt.figure(figsize=(12, 6))
    
    n_tasks = len(accuracies_history)
    colors = plt.cm.tab10(np.linspace(0, 1, n_tasks))
    
    for i in range(n_tasks):
        accuracies = accuracies_history[i]
        if len(accuracies) > 0:
            plt.plot(range(1, len(accuracies) + 1), accuracies, 
                    marker='o', color=colors[i], label=f'Task {i+1}')
    
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    return plt

def plot_risk_group_comparison(risk_accuracies, title="Accuracy by Risk Group"):
    if not risk_accuracies:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No risk data available", ha='center', va='center', fontsize=14)
        return fig
    
    groups = list(risk_accuracies.keys())
    accuracies = list(risk_accuracies.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#D62828' if g == 'critical' else '#F77F00' if g == 'high' 
              else '#FCBF49' if g == 'medium' else '#2E86AB' for g in groups]
    bars = plt.bar(groups, accuracies, color=colors)
    plt.xlabel('Risk Group', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1.0)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return plt

def save_figure(plt, filename, path='./results/figures/'):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    return full_path
# ============================================================================
# NOUVELLES FONCTIONS DE VISUALISATION
# ============================================================================

def plot_confusion_matrix(labels, predictions, class_names=None, title="Confusion Matrix", save_path=None):
    """
    Affiche et sauvegarde la matrice de confusion
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(12, 10))
    
    if class_names and len(class_names) <= 20:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 8})
    else:
        sns.heatmap(cm, annot=False, cmap='Blues')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Matrice de confusion sauvegardée: {save_path}")
    
    return plt


def plot_per_class_metrics(per_class_df, title="Per-Class Performance", save_path=None):
    """
    Affiche les métriques par classe sous forme de barres
    """
    # Exclure la ligne "Weighted Avg"
    df_plot = per_class_df[per_class_df['class'] != 'Weighted Avg'].copy()
    
    if len(df_plot) == 0:
        print("   ⚠️ Aucune classe à afficher")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df_plot))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_plot['precision'], width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x, df_plot['recall'], width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + width, df_plot['f1'], width, label='F1-Score', color='#F18F01')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['class'], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Métriques par classe sauvegardées: {save_path}")
    
    return plt


def plot_class_performance_over_time(accuracies_history, class_names=None, title="Class Performance Over Time", save_path=None):
    """
    Affiche l'évolution des performances par classe au fil des tâches
    """
    n_tasks = len(accuracies_history)
    if n_tasks == 0:
        return None
    
    # Déterminer le nombre de classes
    n_classes = max(len(hist) for hist in accuracies_history) if accuracies_history else 0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_classes, 1)))
    
    for class_idx in range(n_classes):
        class_acc = []
        for task_idx in range(n_tasks):
            if class_idx < len(accuracies_history[task_idx]):
                class_acc.append(accuracies_history[task_idx][class_idx])
            else:
                class_acc.append(0.0)
        
        class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Class_{class_idx}'
        ax.plot(range(1, n_tasks + 1), class_acc, 
                marker='o', color=colors[class_idx % len(colors)],
                label=class_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Task Number', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, n_tasks + 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Performance par classe sauvegardée: {save_path}")
    
    return plt


def plot_risk_vs_performance(per_class_df, risk_mapping, class_names, title="Risk Score vs F1 Performance", save_path=None):
    """
    Affiche la corrélation entre le score de risque et la performance F1
    """
    # Extraire les données
    df_plot = per_class_df[per_class_df['class'] != 'Weighted Avg'].copy()
    
    if len(df_plot) == 0:
        return None
    
    risks = []
    f1_scores = []
    classes = []
    
    for _, row in df_plot.iterrows():
        class_name = row['class']
        # Trouver l'index de la classe
        class_idx = None
        for i, name in enumerate(class_names):
            if name == class_name:
                class_idx = i
                break
        if class_idx is not None and class_idx in risk_mapping:
            risks.append(risk_mapping[class_idx])
            f1_scores.append(row['f1'])
            classes.append(class_name)
    
    if not risks:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot avec couleurs selon le risque
    scatter = ax.scatter(risks, f1_scores, s=100, alpha=0.7, 
                         c=risks, cmap='RdYlGn_r', vmin=0, vmax=1)
    
    # Ajouter les noms des classes
    for i, class_name in enumerate(classes):
        ax.annotate(class_name, (risks[i], f1_scores[i]), 
                    fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Risk Score (AIM)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Ajouter une ligne de corrélation
    if len(risks) > 1:
        z = np.polyfit(risks, f1_scores, 1)
        p = np.poly1d(z)
        ax.plot([0, 1], [p(0), p(1)], '--', color='gray', alpha=0.5, 
                label=f'Corrélation (r={np.corrcoef(risks, f1_scores)[0,1]:.2f})')
        ax.legend()
    
    # Ajouter une colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Risk Score', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Risk vs Performance sauvegardé: {save_path}")
    
    return plt