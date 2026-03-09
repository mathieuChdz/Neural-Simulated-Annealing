from BinPackingProblemNSA import BinPackingProblemNSA
from NeuralSimulatedAnnealing import NeuralSimulatedAnnealing
from agents.ppo import PPOAgent
import random

def main():
    random.seed(62)

    n_items = 50
    n_bins = 20
    bin_capacity = 100
    items = [random.randint(5, 20) for _ in range(n_items)]

    # === Problem instance ===
    problem = BinPackingProblemNSA(items, bin_capacity, n_bins)

    state_dim = n_items * n_bins + 1
    action_dim = problem.action_space()

    # === PPO agent ===
    agent = PPOAgent(state_dim, action_dim)
    agent.load("agents/ppo_model.pth")  

    # Neural SA avec agent
    sa = NeuralSimulatedAnnealing(
        problem,
        n_steps=2000,
        agent=agent  # <-- juste l'agent, temp_net est inclus dedans
    )

    best_state, best_energy = sa.solve()

    print("Bins utilisés :", best_energy)
    print("Configuration :", best_state)



if __name__ == "__main__":
    main()