import random
import torch
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SimulatedAnnealing import SimulatedAnnealing
from KnapsackProblemNSA import KnapsackProblem as kpnsa
from agents.ppo import PPOAgent


def train():

    batch_size = 16

    n_items = 100
    state_dim = n_items + 3
    action_dim = n_items

    agent = PPOAgent(state_dim, action_dim)

    n_instances = 1000

    for episode in range(n_instances):

        liste_batch = []

        for i in range(batch_size):

            poids = np.random.rand(n_items)
            valeurs = np.random.rand(n_items)

            if n_items == 50 or n_items == 100:
                capacity = n_items/4
            else:
                capacity = n_items/8


            problem = kpnsa(poids, valeurs, capacity)

            sa = SimulatedAnnealing(
                problem,
                initial_temp=1,
                final_temp=0.1,
                n_steps=100,
                agent=agent
            )

            best_state, best_energy, _ = sa.solve()

            liste_batch.append(-best_energy)

        print(f"Episode {episode+1}/{n_instances} | Reward moyen : {sum(liste_batch)/len(liste_batch):.4f}")
        agent.update()

    agent.save("agents/ppo_model_100.pth")

if __name__ == "__main__":
    train()