import random
import numpy as np

from BinPackingProblem import BinPackingProblem
from BinPackingProblemNSA import BinPackingProblem as BinPackingProblemNSA
from SimulatedAnnealing import SimulatedAnnealing
from agents.ppo import PPOAgent


# tailles testées
sizes = [50,100,200,500,1000,2000]

# paramètres
n_seeds = 5
instances_per_seed = 20

def run_experiment():

    results = []

    for N in sizes:

        print("\nRunning experiments for N =", N)

        vanilla_scores = []
        neural_scores = []

        for seed in range(n_seeds):

            random.seed(seed)

            for _ in range(instances_per_seed):

                weights = [random.randint(1,10) for _ in range(N)]
                capacity = 15


                # ---------- Vanilla SA ----------

                problem = BinPackingProblem(weights, capacity)

                sa = SimulatedAnnealing(
                    problem,
                    initial_temp=100,
                    final_temp=0.1,
                    n_steps=2000
                )

                _, energy_vanilla, _ = sa.solve()

                vanilla_scores.append(energy_vanilla)


                # ---------- Neural SA ----------

                problem_nsa = BinPackingProblemNSA(weights, capacity)

                state_dim = 2*N + 1
                action_dim = N*N

                agent = PPOAgent(state_dim, action_dim)

                agent.load(f"agents/ppo_model_{N}.pth")

                sa_nsa = SimulatedAnnealing(
                    problem_nsa,
                    initial_temp=100,
                    final_temp=0.1,
                    n_steps=2000,
                    agent=agent
                )

                _, energy_neural, _ = sa_nsa.solve()

                neural_scores.append(energy_neural)


        vanilla_mean = np.mean(vanilla_scores)
        neural_mean = np.mean(neural_scores)

        results.append((N, vanilla_mean, neural_mean))


    print("\n===== RESULTS =====")

    print(f"{'N':>6} | {'Vanilla SA':>12} | {'Neural SA':>12}")
    print("-"*36)

    for r in results:

        N,v,n = r
        print(f"{N:>6} | {v:>12.2f} | {n:>12.2f}")


if __name__ == "__main__":

    run_experiment()