import random
import numpy as np
import matplotlib.pyplot as plt
import os

from yanis.KnapsackProblem import KnapsackProblem
from yanis.SimulatedAnnealing import SimulatedAnnealing
from yanis.es_agent import ESAgent


def main():
    nb_objets = 100
    poids = [random.uniform(0,1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0,1) for _ in range(nb_objets)]
    capacite_max = nb_objets/4

    probleme = KnapsackProblem(poids, valeurs, capacite_max)
    
    sa_classique = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=2000)
    
    meilleur_etat_c, meilleure_energie_c, historique_c = sa_classique.solve(log_filename="yanis/journal_sa_classique.txt")
    
    print("\n--- RÉSULTATS CLASSIQUE ---")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat_c, poids))} / {capacite_max}")
    print(f"Valeur totale du sac : {-meilleure_energie_c}")


    n_steps_sa = 2000
    mon_agent_es = ESAgent(n_items=nb_objets, n_steps_per_episode=n_steps_sa, pop_size=10)

    MODE_ENTRAINEMENT = False  

    if MODE_ENTRAINEMENT:
        print("-> DÉBUT DE L'ENTRAÎNEMENT (Veuillez patienter...)")
        if os.path.exists("yanis/poids_agent_es.npy"):
            print("   (Ancien cerveau détecté : Reprise de l'entraînement !)")
            mon_agent_es.load("yanis/poids_agent_es.npy")
        else:
            print("   (Aucun cerveau détecté : Entraînement depuis zéro.)")
        generations = 5000
        total_runs = mon_agent_es.pop_size * generations
        
        for run in range(total_runs):
            sa_train = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa, agent=mon_agent_es)
            sa_train.solve(log_filename="nul") # "nul" évite de créer un fichier à chaque fois
            if (run + 1) % mon_agent_es.pop_size == 0:
                print(f"   Génération {int((run + 1)/mon_agent_es.pop_size)}/{generations} évaluée et agent mis à jour.")

        mon_agent_es.save("poids_agent_es.npy")
    else:
        print("-> CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ")
        mon_agent_es.load("poids_agent_es.npy")


    print("\n--- TEST FINAL DE L'AGENT ES ---")
    sa_es = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa, agent=mon_agent_es)
    meilleur_etat_es, meilleure_energie_es, historique_es = sa_es.solve(log_filename="yanis/journal_sa_es.txt")

    print("\n--- RÉSULTATS AGENT ES ---")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat_es, poids))} / {capacite_max}")
    print(f"Valeur totale du sac : {-meilleure_energie_es}")



if __name__ == "__main__":
    main()