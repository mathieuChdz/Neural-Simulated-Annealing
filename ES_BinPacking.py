import random
import os

from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.es_agent import ESAgent


def main():

    n_items = 50
    n_bins = 50
    capacity = 1

    weights = [random.uniform(0,1) for _ in range(n_items)]

    problem = BinPackingProblemNSA(weights, capacity, n_bins)

    print("\n--- SA CLASSIQUE ---")

    sa_classique = NeuralSimulatedAnnealing(
        problem,
        n_steps=2000,
        agent=None
    )

    best_state_c, best_energy_c = sa_classique.solve()

    print("Bins utilisés :", best_energy_c)


    # ===== Agent ES =====

    agent = ESAgent(n_items, n_bins)

    MODE_ENTRAINEMENT = False


    if MODE_ENTRAINEMENT:

        print("-> ENTRAINEMENT ES")

        if os.path.exists("agents/es_weights_50.npy"):
            agent.load("agents/es_weights_50.npy")

        generations = 200

        for g in range(generations):

            sa = NeuralSimulatedAnnealing(
                problem,
                n_steps=2000,
                agent=agent
            )

            sa.solve()

            agent.update()

            print("Generation", g)

        agent.save("agents/es_weights_50.npy")

    else:

        print("-> CHARGEMENT MODELE ES")

        agent.load("agents/es_weights_50.npy")


    print("\n--- TEST ES ---")

    sa_es = NeuralSimulatedAnnealing(
        problem,
        n_steps=2000,
        agent=agent
    )

    best_state_es, best_energy_es = sa_es.solve()

    print("Bins utilisés :", best_energy_es)


if __name__ == "__main__":
    main()
