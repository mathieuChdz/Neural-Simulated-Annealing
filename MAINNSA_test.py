import random
from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing

def main():
    random.seed(62)
    n_items = 50
    n_bins = 20
    bin_capacity = 100
    items = [random.randint(5, 20) for _ in range(n_items)]

    problem = BinPackingProblemNSA(items, bin_capacity, n_bins)

    sa = NeuralSimulatedAnnealing(
        problem,
        n_steps=2000,
        agent=None  # random SA
    )

    best_state, best_energy = sa.solve()

    print("Bins utilisés :", best_energy)
    print("Configuration :", best_state)

if __name__ == "__main__":
    main()