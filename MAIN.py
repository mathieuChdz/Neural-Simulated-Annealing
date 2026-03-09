import random
from BinPackingProblem import BinPackingProblem
from SimulatedAnnealing import SimulatedAnnealing


def main():

    random.seed(62)

    n_items = 50

    weights = [random.randint(1, 10) for _ in range(n_items)]

    capacity = 15

    problem = BinPackingProblem(weights, capacity)

    sa = SimulatedAnnealing(problem, n_steps=2000)

    best_state, best_energy, history = sa.solve()

    print("bins utilisés :", best_energy)


if __name__ == "__main__":
    main()