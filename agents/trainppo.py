import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.ppo import PPOAgent
import torch

def train():
    random.seed(42)
    torch.manual_seed(42)

    n_items = 100
    n_bins = 100
    bin_capacity = 100

    agent = PPOAgent(state_dim=n_items*n_bins + 1, action_dim=n_items*n_bins)

    n_instances = 50
    n_steps = 1000

    for ep in range(n_instances):
        items = [random.randint(5, 100) for _ in range(n_items)]
        problem = BinPackingProblemNSA(items, bin_capacity, n_bins)
        sa = NeuralSimulatedAnnealing(problem, n_steps=n_steps, agent=agent)
        best_state, best_energy = sa.solve()
        print(f"[Episode {ep+1}/{n_instances}] Bins utilisés: {best_energy}")

    agent.save("agents/ppo_model100.pth")
    print("Modèle PPO sauvegardé !")

if __name__ == "__main__":
    train()