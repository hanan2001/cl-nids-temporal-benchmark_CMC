"""
Point d'entrée principal - Choisissez votre dataset
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_comparison import compare_rggp_baseline

def print_menu():
    """Affiche le menu des datasets"""
    print("\n" + "="*60)
    print("📊 CHOIX DU DATASET")
    print("="*60)
    print("   1. Synthétique (recommandé pour test rapide)")
    print("   2. CIC-IDS2017 (complet - 2.8M échantillons)")
    print("   3. CIC-IDS2017 (échantillon - 500K échantillons)")
    print("   4. Tous les datasets")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("🔬 RGGP - Risk-Guided Gradient Projection")
    print("="*60)
    
    print_menu()
    
    try:
        choice = input("\n👉 Votre choix (1-4): ").strip()
        
        if choice == "1":
            print("\n🔬 Lancement sur dataset synthétique...")
            compare_rggp_baseline(dataset='synthetic', use_small=False, n_seeds=5)
            
        elif choice == "2":
            print("\n🔬 Lancement sur CIC-IDS2017 complet...")
            compare_rggp_baseline(dataset='cic-ids2017', use_small=False, n_seeds=5)
                
        elif choice == "3":
            print("\n🔬 Lancement sur CIC-IDS2017 (échantillon 500K)...")
            compare_rggp_baseline(dataset='cic-ids2017', use_small=True, n_samples=500000, n_seeds=3)
                
        elif choice == "4":
            print("\n🔬 Lancement sur tous les datasets...")
            try:
                from run_all_experiments import main as run_all
                run_all()
            except ImportError:
                print("   ❌ run_all_experiments.py non trouvé")
                print("   💡 Lancez les expériences individuellement")
            
        else:
            print("❌ Choix invalide")
            
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()