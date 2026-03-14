from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.ppo import PPOAgent
import random


def main():

    random.seed(62)

    # ----- paramètres problème -----
    n_items = 50
    n_bins = n_items
    bin_capacity = 1

    # distribution du papier
    items = [random.uniform(0, 1) for _ in range(n_items)]

    print("Items :", items)
    print("Total weight :", sum(items))

    # ----- problème -----
    problem = BinPackingProblemNSA(items, bin_capacity, n_bins)

    # ----- agent PPO -----
    state_dim = n_items * n_bins + 1
    action_dim = n_items * n_bins

    agent = PPOAgent()

    agent.load("agents/ppo_model_50.pth")

    # ----- Neural SA -----
    sa = NeuralSimulatedAnnealing(
        problem,
        n_steps=500,     # comme dans le papier: K = 10N ≈ 500
        agent=agent
    )

    best_state, best_energy = sa.solve()

    print("\n===== RESULTAT =====")
    print("Bins utilisés :", best_energy)
    print("Configuration :", best_state)


if __name__ == "__main__":
    main()
