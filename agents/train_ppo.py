import random
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SimulatedAnnealing import SimulatedAnnealing
from KnapsackProblemNSA import KnapsackProblem as kpnsa
from agents.ppo import PPOAgent


def train():

    n_items = 100
    state_dim = n_items + 3
    action_dim = n_items

    agent = PPOAgent(state_dim, action_dim)

    n_instances = 500

    for episode in range(n_instances):

        poids = [random.randint(1, 20) for _ in range(n_items)]
        valeurs = [random.randint(10, 50) for _ in range(n_items)]
        capacity = int(sum(poids) * 0.4)


        problem = kpnsa(poids, valeurs, capacity)

        sa = SimulatedAnnealing(
            problem,
            initial_temp=100,
            final_temp=0.1,
            n_steps=2000,
            agent=agent
        )

        best_state, best_energy, _ = sa.solve()

        print("episode:", episode, "value:", -best_energy)

    agent.save("agents/ppo_model.pth")

if __name__ == "__main__":
    train()