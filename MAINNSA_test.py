import random
from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.ppo import PPOAgent

def main():
    random.seed(62)
    n_items = 100
    n_bins = 50
    bin_capacity = 100
    items = [random.uniform(0, 1) for _ in range(n_items)]

    problem = BinPackingProblemNSA(items, bin_capacity, n_bins)

    sa = NeuralSimulatedAnnealing(
        problem,
        n_steps=2000,
        agent = PPOAgent()  
    )

    best_state, best_energy = sa.solve()

    print("Bins utilisés :", best_energy)
    print("Configuration :", best_state)

if __name__ == "__main__":
    main()