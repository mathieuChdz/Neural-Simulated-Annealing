import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SimulatedAnnealing import SimulatedAnnealing
from BinPackingProblemNSA import BinPackingProblem
from agents.ppo import PPOAgent


def train():

    n_items = 50

    state_dim = 2 * n_items + 1
    action_dim = n_items * n_items

    agent = PPOAgent(state_dim, action_dim)

    n_instances = 200

    for episode in range(n_instances):

        weights = [random.randint(1, 10) for _ in range(n_items)]

        capacity = 15

        problem = BinPackingProblem(weights, capacity)

        sa = SimulatedAnnealing(
            problem,
            initial_temp=100,
            final_temp=0.1,
            n_steps=2000,
            agent=agent
        )

        best_state, best_energy, _ = sa.solve()

        print("episode:", episode, "bins:", best_energy)

    agent.save("agents/ppo_model.pth")


if __name__ == "__main__":
    train()