# compare_results_simple.py

import random
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from SimulatedAnnealing import SimulatedAnnealing
from agents.ppo import PPOAgent

# ===== Paramètres =====
SIZES = [50, 100, 200]  # exemple : tu peux ajouter 500, 1000, 2000 si tu as entraîné les modèles
SIZE = [50] 
N_INSTANCES = 5  # nombre d'instances test
BIN_CAPACITY = 100
MODEL_DIR = "agents"

random.seed(42)
torch.manual_seed(42)

def run_comparison():
    results = []

    for N in SIZE:
        print(f"\n--- Test pour N={N} ---")
        vanilla_scores = []
        ppo_scores = []

        # Charger le modèle PPO correspondant
        state_dim = N * N + 1
        action_dim = N * N
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        model_path = os.path.join(MODEL_DIR, f"ppo_model_{N}.pth")
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"⚠️  Attention : {model_path} introuvable. PPO non utilisé pour N={N}")
            agent = None

        for _ in range(N_INSTANCES):
            items = [random.randint(5, 20) for _ in range(N)]
            n_bins = N
            problem = BinPackingProblemNSA(items, BIN_CAPACITY, n_bins)

            # --- Vanilla SA ---
            sa_vanilla = SimulatedAnnealing(problem, n_steps=1000)
            _, energy_vanilla, _ = sa_vanilla.solve()
            vanilla_scores.append(energy_vanilla)

            # --- PPO (Neural SA) ---
            if agent:
                sa_ppo = NeuralSimulatedAnnealing(problem, n_steps=1000, agent=agent)
                _, energy_ppo = sa_ppo.solve()
                ppo_scores.append(energy_ppo)

        vanilla_mean = np.mean(vanilla_scores)
        ppo_mean = np.mean(ppo_scores) if ppo_scores else float('nan')

        results.append((N, vanilla_mean, ppo_mean))

    # ===== Affichage du tableau =====
    print("\n=== Résultats comparatifs ===")
    print(f"{'N':>6} | {'Vanilla SA':>12} | {'PPO SA':>10}")
    print("-"*36)
    for r in results:
        N, v, p = r
        print(f"{N:>6} | {v:>12.2f} | {p:>10.2f}")
        

if __name__ == "__main__":
    run_comparison()