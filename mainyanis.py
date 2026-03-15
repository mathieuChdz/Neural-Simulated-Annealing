import random
import numpy as np
import matplotlib.pyplot as plt
import os

from KnapsackProblem import KnapsackProblem
from SimulatedAnnealing import SimulatedAnnealing
from es_agent import ESAgent


def main():
    nb_objets = 50
    poids = [random.uniform(0,1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0,1) for _ in range(nb_objets)]
    capacite_max = nb_objets/4

    probleme = KnapsackProblem(poids, valeurs, capacite_max)
    
    sa_classique = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=2000)
    
    meilleur_etat_c, meilleure_energie_c, historique_c = sa_classique.solve(log_filename="journal_sa_classique.txt")
    
    print("\n--- RESULTATS CLASSIQUE ---")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat_c, poids))} / {capacite_max}")
    print(f"Valeur totale du sac : {-meilleure_energie_c}")


    n_steps_sa = 2000
    mon_agent_es = ESAgent(n_items=nb_objets, n_steps_per_episode=n_steps_sa, pop_size=10)

    MODE_ENTRAINEMENT = False  

    if MODE_ENTRAINEMENT:
        print("-> DEBUT DE L'ENTRAINEMENT (Veuillez patienter...)")
        if os.path.exists("poids_agent_es_50.npy"):
            print("   (Ancien cerveau détecté : Reprise de l'entraînement !)")
            mon_agent_es.load("poids_agent_es_50.npy")
        else:
            print("   (Aucun cerveau detecte : Entraînement depuis zero.)")
        generations = 500
        total_runs = mon_agent_es.pop_size * generations
        
        for run in range(total_runs):
            sa_train = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa, agent=mon_agent_es)
            sa_train.solve(log_filename="nul")
            if (run + 1) % mon_agent_es.pop_size == 0:
                print(f"   Generation {int((run + 1)/mon_agent_es.pop_size)}/{generations} evaluee et agent mis a jour.")

        mon_agent_es.save("poids_agent_es_50.npy")
    else:
        print("-> CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ")
        mon_agent_es.load("poids_agent_es_50.npy")


    print("\n--- TEST FINAL DE L'AGENT ES ---")
    sa_es = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa, agent=mon_agent_es)
    meilleur_etat_es, meilleure_energie_es, historique_es = sa_es.solve(log_filename="journal_sa_es.txt")

    print("\n--- RÉSULTATS AGENT ES ---")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat_es, poids))} / {capacite_max}")
    print(f"Valeur totale du sac : {-meilleure_energie_es}")



if __name__ == "__main__":
    main()