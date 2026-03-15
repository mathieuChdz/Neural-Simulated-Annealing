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

SIZES = [50, 100]

N_INSTANCES = 20
N_STEPS = 1000

BIN_CAPACITY = 1
MODEL_DIR = "agents"

random.seed(42)
torch.manual_seed(42)


def run_comparison():

    results = []

    for N in SIZES:

        print(f"\n--- Test pour N={N} ---")

        vanilla_scores = []
        ppo_scores = []

        # charger modèle PPO
        agent = PPOAgent()

        model_path = os.path.join(MODEL_DIR, f"ppo_model_norma_{N}.pth")

        if os.path.exists(model_path):

            print("Chargement modèle PPO")
            agent.load(model_path)

        else:

            print(f"⚠ modèle {model_path} introuvable")
            agent = None

        for inst in range(N_INSTANCES):

            items = [random.uniform(0, 1) for _ in range(N)]

            problem = BinPackingProblemNSA(items, BIN_CAPACITY, N)

            # ---------- Vanilla SA ----------

            sa_vanilla = SimulatedAnnealing(problem, n_steps=N_STEPS)

            _, energy_vanilla, _ = sa_vanilla.solve()

            vanilla_scores.append(energy_vanilla)

            # ---------- Neural SA ----------

            if agent:

                sa_ppo = NeuralSimulatedAnnealing(
                    problem,
                    n_steps=N_STEPS,
                    agent=agent
                )

                _, energy_ppo = sa_ppo.solve()

                ppo_scores.append(energy_ppo)

        vanilla_mean = np.mean(vanilla_scores)

        ppo_mean = np.mean(ppo_scores) if ppo_scores else float('nan')

        results.append((N, vanilla_mean, ppo_mean))

    # ===== Tableau final =====

    print("\n=== Résultats comparatifs ===")

    print(f"{'N':>6} | {'Vanilla SA':>12} | {'Neural SA':>12}")

    print("-" * 38)

    for N, v, p in results:

        print(f"{N:>6} | {v:>12.2f} | {p:>12.2f}")


if __name__ == "__main__":
    run_comparison()