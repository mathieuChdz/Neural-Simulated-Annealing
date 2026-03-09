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
SIZES = [50, 100, 200, 500, 1000]   # tailles à entraîner
N_INSTANCES = 1000                     # instances par taille
N_STEPS = 1000                       # étapes de SA par instance
BIN_CAPACITY = 100                   

# dossier pour sauvegarder les modèles
MODEL_DIR = "agents"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fixer seeds pour reproductibilité
random.seed(42)
torch.manual_seed(42)

def train_for_size(n_items):
    n_bins = n_items  # nombre de bins = nombre d’items, comme dans l’article
    print(f"\n=== Entraînement pour N={n_items} ===")

    # Création de l'agent PPO
    state_dim = n_items * n_bins + 1
    action_dim = n_items * n_bins
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    for ep in range(N_INSTANCES):
        random.seed(ep)
        torch.manual_seed(ep)
        # Générer instance aléatoire
        items = [random.randint(5, 20) for _ in range(n_items)]
        problem = BinPackingProblemNSA(items, BIN_CAPACITY, n_bins)

        # SA avec PPO
        sa = NeuralSimulatedAnnealing(problem, n_steps=N_STEPS, agent=agent)
        best_state, best_energy = sa.solve()
        print(f"Episode {ep} energy {best_energy}")
        print(f"[Episode {ep+1}/{N_INSTANCES}] Bins utilisés: {best_energy}")

    # Sauvegarde du modèle
    model_path = os.path.join(MODEL_DIR, f"ppo_model_{n_items}.pth")
    agent.save(model_path)
    print(f"Modèle PPO pour N={n_items} sauvegardé : {model_path}")

if __name__ == "__main__":
    for size in SIZES:
        train_for_size(size)