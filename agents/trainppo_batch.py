# agents/trainppo_all_sizes.py

import random
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.ppo import PPOAgent


# ===== Paramètres =====

N_ITEMS = 50              # comme le papier : BIN50
N_BINS = N_ITEMS
BIN_CAPACITY = 1

EPOCHS = 100              # au lieu de 1000
PROBLEMS_PER_EPOCH = 32   # au lieu de 256
ROLLOUT_STEPS = 100       # K = 100 comme dans l'article


MODEL_DIR = "agents"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ppo_model_norma_50.pth")

random.seed(42)
torch.manual_seed(42)


def train():

    print("=== Training Neural SA (BIN50) ===")

    agent = PPOAgent()

    if os.path.exists(MODEL_PATH):
        print("Chargement modèle existant")
        agent.load(MODEL_PATH)

    for epoch in range(EPOCHS):

        for p in range(PROBLEMS_PER_EPOCH):

            seed = epoch * PROBLEMS_PER_EPOCH + p
            random.seed(seed)
            torch.manual_seed(seed)

            # génération instance
            items = [random.uniform(0,1) for _ in range(N_ITEMS)]

            problem = BinPackingProblemNSA(
                items,
                BIN_CAPACITY,
                N_BINS
            )

            sa = NeuralSimulatedAnnealing(
                problem,
                n_steps=ROLLOUT_STEPS,
                agent=agent
            )

            best_state, best_energy = sa.solve()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")

    agent.save(MODEL_PATH)

    print("Model saved :", MODEL_PATH)


if __name__ == "__main__":
    train()
