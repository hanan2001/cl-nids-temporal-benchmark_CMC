"""
Chargement du dataset CIC-IDS2017
VERSION PAPER-READY - Normalisation APRÈS split (sans fuite de données)

IMPORTANT: 
- Les données sont chargées BRUTES (sans normalisation)
- La normalisation est effectuée APRÈS le split train/validation/test
- Le scaler est fitté UNIQUEMENT sur les données d'entraînement
- Cela évite toute fuite de données des ensembles de validation/test
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


def load_cic_ids2017_all_files(data_path='./data/CIC-IDS2017/MachineLearningCSV/'):
    """
    Charge TOUS les fichiers CSV de CIC-IDS2017
    Retourne les données BRUTES (non normalisées)
    
    Returns:
        X: numpy array (n_samples, n_features) - données brutes
        y_encoded: numpy array (n_samples,) - labels encodés
        class_names: list - noms des classes
    """
    try:
        print("   📥 Chargement de CIC-IDS2017 (tous les fichiers)...")
        
        csv_files = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
        ]
        
        all_dfs = []
        
        for filename in csv_files:
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                print(f"      📄 Chargement: {filename}")
                
                try:
                    df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
                    df.columns = [col.strip() for col in df.columns]
                    
                    # Identifier la colonne label
                    label_col = None
                    for col in df.columns:
                        if col.lower() in ['label', 'attack', 'class', 'malicious']:
                            label_col = col
                            break
                    
                    if label_col is None:
                        if 'Label' in df.columns:
                            label_col = 'Label'
                        elif 'label' in df.columns:
                            label_col = 'label'
                        else:
                            print(f"      ⚠️ Colonne label introuvable dans {filename}")
                            continue
                    
                    if label_col != 'Label':
                        df = df.rename(columns={label_col: 'Label'})
                    
                    # Nettoyage
                    df = df.replace([np.inf, -np.inf], 0)
                    df = df.dropna()
                    
                    # Conversion des colonnes numériques
                    for col in df.columns:
                        if col != 'Label':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Supprimer les colonnes avec toutes des NaN
                    df = df.dropna(axis=1, how='all')
                    
                    if len(df) == 0:
                        print(f"      ⚠️ {filename} vide après nettoyage")
                        continue
                    
                    all_dfs.append(df)
                    print(f"      ✅ {len(df):,} échantillons")
                    
                except Exception as e:
                    print(f"      ⚠️ Erreur: {e}")
            else:
                print(f"      ⚠️ Fichier non trouvé: {filename}")
        
        if not all_dfs:
            print("   ❌ Aucun fichier valide chargé")
            return None, None, None
        
        print(f"\n   🔄 Fusion des {len(all_dfs)} fichiers...")
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.dropna()
        
        # Séparer features et labels
        X = df.drop(['Label'], axis=1).values.astype(np.float32)
        y = df['Label'].values
        
        # Encoder les labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        
        # 🔥 NE PAS NORMALISER ICI - sera fait après split
        # Remplacer les NaN/inf par 0 (cas extrêmes)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"\n   ✅ CIC-IDS2017 chargé: {X.shape[0]:,} échantillons, {X.shape[1]} features")
        print(f"      {len(class_names)} classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        return X, y_encoded, class_names
    
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_cic_ids2017_small(data_path='./data/CIC-IDS2017/MachineLearningCSV/',
                           n_samples=500000, min_per_class=2000, seed=42):
    """
    Échantillon réduit qui PRÉSERVE les classes rares.
    Les classes majoritaires sont plafonnées ; les classes < min_per_class
    sont conservées intégralement.
    """
    print(f"   🔄 Chargement (préservation des classes rares)...")
    X, y, class_names = load_cic_ids2017_all_files(data_path)
    if X is None:
        return None, None, None
    if X.shape[0] <= n_samples:
        return X, y, class_names

    rng = np.random.RandomState(seed)
    unique, counts = np.unique(y, return_counts=True)

    # budget pour les classes NON-rares, après avoir réservé toutes les rares
    reserved = int(sum(c for u, c in zip(unique, counts) if c < min_per_class))
    budget = max(0, n_samples - reserved)
    big_classes = [u for u, c in zip(unique, counts) if c >= min_per_class]
    big_total = int(sum(c for u, c in zip(unique, counts) if c >= min_per_class))

    keep_idx = []
    for u, c in zip(unique, counts):
        cls_idx = np.where(y == u)[0]
        if c < min_per_class:
            keep_idx.append(cls_idx)                      # garder TOUTE la classe rare
        else:
            quota = int(round(budget * c / big_total)) if big_total > 0 else c
            quota = max(min_per_class, min(quota, c))
            keep_idx.append(rng.choice(cls_idx, size=quota, replace=False))

    keep = np.concatenate(keep_idx)
    rng.shuffle(keep)
    X_s, y_s = X[keep], y[keep]

    print(f"   ⚡ Réduit à {len(X_s):,} échantillons | classes rares préservées")
    for u, c in zip(*np.unique(y_s, return_counts=True)):
        name = class_names[u] if u < len(class_names) else f"Class_{u}"
        print(f"      {name}: {c:,}")
    return X_s, y_s, class_names
def load_dataset_by_name(name, data_path='./data/', n_samples=None):
    """
    Interface unifiée pour charger un dataset par nom
    Retourne les données BRUTES (non normalisées)
    """
    name = name.lower()
    
    if name == 'cic-ids2017':
        if n_samples is not None:
            return load_cic_ids2017_small(data_path + 'CIC-IDS2017/MachineLearningCSV/', n_samples)
        return load_cic_ids2017_all_files(data_path + 'CIC-IDS2017/MachineLearningCSV/')
    
    elif name == 'unsw-nb15':
        from data.loader import load_unsw_nb15
        return load_unsw_nb15(data_path + 'UNSW-NB15/')
    
    else:
        print(f"   ❌ Dataset '{name}' non reconnu")
        print(f"   💡 Datasets disponibles: 'cic-ids2017', 'unsw-nb15'")
        return None, None, None