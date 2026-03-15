import os
import random
import numpy as np
from KnapsackProblem import KnapsackProblem as kp
from KnapsackProblemNSA import KnapsackProblem as kpnsa
from SimulatedAnnealing import SimulatedAnnealing
from agents.ppo import PPOAgent
from es_agent import ESAgent


SEED = 42
SEEDS = [random.randint(0, 10000) for _ in range(5)]
SEEDS_CUSTOM = [3322, 8092, 973, 4669, 62]
random.seed(SEED)
np.random.seed(SEED)

def generer_probleme(nb_objets, seed=None, nsa=False):
    """Génère un problème de Knapsack consistant pour une seed donnée."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]

    if nb_objets == 50 or nb_objets == 100:
        capacite_max = nb_objets / 4.0
    else:
        capacite_max = nb_objets / 8.0

    if nsa:
        return kpnsa(poids, valeurs, capacite_max)
    return kp(poids, valeurs, capacite_max)

def run_SA(nb_objets):
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]
    capacite_max = nb_objets / 4

    probleme = kp(poids, valeurs, capacite_max)
    res_sum = 0
    meilleur_etat_global = None

    for i in range(5):
        probleme = generer_probleme(nb_objets, seed=SEEDS_CUSTOM[i], nsa=False)
        sa = SimulatedAnnealing(probleme, initial_temp=1.0, final_temp=0.1, n_steps=5000)
        meilleur_etat, meilleure_energie, _ = sa.solve(log_filename=f"journal_sa_run_{i+1}.txt")
        print(f"Run {i+1} | Meilleure énergie : {-meilleure_energie:.4f} | Random Seed: {SEEDS_CUSTOM[i]}")
        res_sum += -meilleure_energie
        meilleur_etat_global = meilleur_etat
    
    res_mean = res_sum / 5
    poids_final = sum(s * p for s, p in zip(meilleur_etat_global, poids))

    print("\n--- RÉSULTATS SA CLASSIQUE ---")
    print(f"Configuration du dernier run : {meilleur_etat_global}")
    print(f"Poids total : {poids_final:.4f} / {capacite_max}")
    print(f"Valeur moyenne (sur 5 runs) : {res_mean:.4f}")


def run_NSA(nb_objets):
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]
    capacite_max = nb_objets / 4

    probleme = kpnsa(poids, valeurs, capacite_max)
    
    state_dim = nb_objets + 3   # sélection + poids_norm + valeur_norm + temp
    action_dim = probleme.action_space()

    agent = PPOAgent(state_dim, action_dim)
    
    fichier_modele = f"agents/ppo_model_{nb_objets}.pth"
    try:
        agent.load(fichier_modele)
    except FileNotFoundError:
        print(f"/!\\ ERREUR : Le fichier '{fichier_modele}' est introuvable. Mathieu doit l'entraîner d'abord !")
        return

    res_sum = 0
    meilleur_etat_global = None

    for i in range(5):
        probleme = generer_probleme(nb_objets, seed=SEEDS_CUSTOM[i], nsa=True)
        sa = SimulatedAnnealing(probleme, initial_temp=1.0, final_temp=0.1, n_steps=1000, agent=agent)
        meilleur_etat, meilleure_energie, _ = sa.solve(log_filename=f"journal_nsa_run_{i+1}.txt")
        print(f"Run {i+1} | Meilleure énergie : {-meilleure_energie:.4f} | Random Seed: {SEEDS_CUSTOM[i]}")
        res_sum += -meilleure_energie
        meilleur_etat_global = meilleur_etat
    
    res_mean = res_sum / 5
    poids_final = sum(s * p for s, p in zip(meilleur_etat_global, poids))
    
    print("\n--- RÉSULTATS NEURAL SA (PPO) ---")
    print(f"Configuration du dernier run : {meilleur_etat_global}")
    print(f"Poids total : {poids_final:.4f} / {capacite_max}")
    print(f"Valeur moyenne (sur 5 runs) : {res_mean:.4f}")


def run_ES(nb_objets):
    poids = [random.uniform(0,1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0,1) for _ in range(nb_objets)]
    capacite_max = nb_objets / 4

    probleme = kp(poids, valeurs, capacite_max)
    n_steps_sa = 2000
    mon_agent_es = ESAgent(n_items=nb_objets, n_steps_per_episode=n_steps_sa, pop_size=10)

    fichier_poids = f"agents/poids_agent_es_{nb_objets}.npy"
    try:
        mon_agent_es.load(fichier_poids)
    except FileNotFoundError:
        print(f"/!\\ ERREUR : Le fichier '{fichier_poids}' est introuvable.")
        print(f"     Vous devez d'abord lancer l'entraînement pour N={nb_objets} objets !")
        return

    res_sum = 0
    meilleur_etat_global = None

    for i in range(5):
        probleme = generer_probleme(nb_objets, seed=SEEDS_CUSTOM[i], nsa=False)
        sa_es = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa, agent=mon_agent_es)
        meilleur_etat_es, meilleure_energie_es, _ = sa_es.solve(log_filename=f"journal_es_run_{i+1}.txt")
        print(f"Run {i+1} | Meilleure énergie : {-meilleure_energie_es:.4f} | Random Seed: {SEEDS_CUSTOM[i]}")
        res_sum += -meilleure_energie_es
        meilleur_etat_global = meilleur_etat_es

    res_mean = res_sum / 5
    poids_final = sum(s * p for s, p in zip(meilleur_etat_global, poids))

    print("\n--- RÉSULTATS NEURAL SA (ES) ---")
    print(f"Configuration du dernier run : {meilleur_etat_global}")
    print(f"Poids total : {poids_final:.4f} / {capacite_max}")
    print(f"Valeur moyenne (sur 5 runs) : {res_mean:.4f}")


if __name__ == "__main__":
    print("=========================================")
    print("   PROJET NEURAL SIMULATED ANNEALING     ")
    print("=========================================")
    print("1. SA  - Recuit Simulé Classique")
    print("2. NSA - Agent PPO ")
    print("3. ES  - Agent Evolution Strategies ")
    print("=========================================")
    
    algo_choisi = input("Choisissez l'algorithme (SA, NSA, ou ES) : ").strip().upper()
    
    print("\nTailles disponibles : 50, 100, 200")
    taille_choisie = input("Choisissez la taille du sac à dos : ").strip()

    if taille_choisie not in ["50", "100", "200"]:
        print("Taille invalide. Par défaut, N=50 a été sélectionné.")
        taille_choisie = 50
    else:
        taille_choisie = int(taille_choisie)
    
    if algo_choisi == "SA" or algo_choisi == "1":
        run_SA(taille_choisie)
    elif algo_choisi == "NSA" or algo_choisi == "2":
        run_NSA(taille_choisie)
    elif algo_choisi == "ES" or algo_choisi == "3":
        run_ES(taille_choisie)
    else:
        print("Sélection d'algorithme invalide. Fin du programme.")