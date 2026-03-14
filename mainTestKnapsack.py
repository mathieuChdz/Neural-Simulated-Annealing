import random
import numpy as np
import time
import os
from KnapsackProblemNSA import KnapsackProblem as kpnsa
from KnapsackProblem import KnapsackProblem as kp
from SimulatedAnnealing import SimulatedAnnealing

from agents.ppo import PPOAgent

def generer_probleme(nb_objets, seed=None, nsa=False):
    """Génère un problème de Knapsack consistant pour une seed donnée."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]
    capacite_max = nb_objets / 4.0

    if nsa:
        return kpnsa(poids, valeurs, capacite_max)
    return kp(poids, valeurs, capacite_max)

def run_stats():
    # Configuration
    nb_objets = 50
    nb_seeds = 5
    n_steps = 1000
    path_model = f"agents/ppo_model_{nb_objets}.pth"

    # Initialisation de l'agent PPO
    state_dim = nb_objets + 3
    # On crée un problème temporaire juste pour avoir l'action_dim
    temp_prob = generer_probleme(nb_objets, nsa=True)
    agent_ppo = PPOAgent(state_dim, temp_prob.action_space())
    
    if os.path.exists(path_model):
        agent_ppo.load(path_model)
        print(f"> Modèle PPO chargé : {path_model}")
    else:
        print(f" /!\\ Modèle {path_model} non trouvé. L'agent sera aléatoire.")

    scores_sa = []
    scores_nsa = []

    print("-" * 60)
    print(f"Évaluation sur {nb_seeds} seeds (N={nb_objets}, Steps={n_steps})")
    print(f"{'Seed':<10} | {'SA Classique':<15} | {'NSA (PPO)':<15}")
    print("-" * 60)

    for seed in range(nb_seeds):
        # 1. SA Classique
        prob_std = generer_probleme(nb_objets, seed=seed, nsa=False)
        sa_std = SimulatedAnnealing(prob_std, initial_temp=1.0, final_temp=0.1, n_steps=n_steps)
        _, e_std, _ = sa_std.solve(log_filename="nul") # "nul" pour éviter les logs inutiles
        val_std = -e_std
        scores_sa.append(val_std)

        # 2. NSA (avec Agent PPO)
        prob_nsa = generer_probleme(nb_objets, seed=seed, nsa=True)
        sa_nsa = SimulatedAnnealing(prob_nsa, initial_temp=1.0, final_temp=0.1, n_steps=n_steps, agent=agent_ppo)
        _, e_nsa, _ = sa_nsa.solve(log_filename="nul")
        val_nsa = -e_nsa
        scores_nsa.append(val_nsa)

        print(f"{seed:<10} | {val_std:<15.4f} | {val_nsa:<15.4f}")

    # Calcul des moyennes
    moy_sa = np.mean(scores_sa)
    moy_nsa = np.mean(scores_nsa)
    amelioration = ((moy_nsa - moy_sa) / moy_sa) * 100

    print("-" * 60)
    print(f"MOYENNE    | {moy_sa:<15.4f} | {moy_nsa:<15.4f}")
    print(f"AMÉLIORATION : {amelioration:+.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    run_stats()