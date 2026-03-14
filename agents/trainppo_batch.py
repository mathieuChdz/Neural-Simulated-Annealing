# agents/trainppo_all_sizes.py

import random
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.ppo import PPOAgent

# ===== Paramètres généraux =====
SIZES = [50, 100, 200]   # tailles à entraîner
SIZE =[ 500, 1000 ]                            # taille pour test rapide (remplace SIZES pour un test rapide)
N_INSTANCES = 500                     # instances par taille
N_STEPS = 500                        # étapes de SA par instance
BIN_CAPACITY = 100                   

# dossier pour sauvegarder les modèles
MODEL_DIR = "agents"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fixer seeds pour reproductibilité
random.seed(42)
torch.manual_seed(42)

def train_for_size(n_items):

    n_bins = n_items
    print(f"\n=== Entraînement pour N={n_items} ===")

    model_path = os.path.join(MODEL_DIR, f"ppo_model_{n_items}.pth")

    agent = PPOAgent()

    # charger modèle existant
    if os.path.exists(model_path):
        print("Chargement modèle existant")
        agent.load(model_path)

    for ep in range(N_INSTANCES):

        random.seed(ep)
        torch.manual_seed(ep)

        items = [random.randint(5, 20) for _ in range(n_items)]

        problem = BinPackingProblemNSA(items, BIN_CAPACITY, n_bins)

        sa = NeuralSimulatedAnnealing(problem, n_steps=N_STEPS, agent=agent)

        best_state, best_energy = sa.solve()

        if ep % 10 == 0:
            print(f"Episode {ep}/{N_INSTANCES} energy {best_energy}")

    agent.save(model_path)

    print(f"Modèle PPO pour N={n_items} sauvegardé : {model_path}")
if __name__ == "__main__":
    for size in SIZES:
        train_for_size(size)