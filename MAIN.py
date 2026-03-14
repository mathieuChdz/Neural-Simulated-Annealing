import random

from BinPackingProblem import BinPackingProblem
from SimulatedAnnealing import SimulatedAnnealing


def main():

    random.seed(62)

    n_items = 50
    capacity = 1

    weights = [random.uniform(0, 1) for _ in range(n_items)]

    print("Weights :", weights)
    print("Total weight :", sum(weights))

    problem = BinPackingProblem(weights, capacity)

    sa = SimulatedAnnealing(
        problem,
        initial_temp=100,
        final_temp=0.01,
        n_steps=5000
    )

    best_state, best_energy, history = sa.solve()

    print("\n========================")
    print("Bins utilisés :", best_energy)
    print("========================")

    # afficher les bins
    bins = {}

    for i, b in enumerate(best_state):

        bins.setdefault(b, [])
        bins[b].append(weights[i])

    for b, items in bins.items():

        load = sum(items)

        print(f"Bin {b} | load={load:.3f} | n_items={len(items)}")


if __name__ == "__main__":
    main()
