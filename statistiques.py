import random
import numpy as np
import time
import os

from yanis.KnapsackProblem import KnapsackProblem
from yanis.SimulatedAnnealing import SimulatedAnnealing
from yanis.es_agent import ESAgent

def generer_probleme(nb_objets, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]

    if nb_objets < 200:
        capacite_max = nb_objets /4.0
    else:
        capacite_max = nb_objets / 8.0

    return KnapsackProblem(poids, valeurs, capacite_max)

def run_stats():
    tailles_problemes = [50, 100, 200]

    nb_seeds = 5
    n_steps_sa = 2000 
    
    print("-" * 50)
    print(f"Lancement de l'évaluation (Moyenne sur {nb_seeds} runs)")
    print(f"{'Taille':<10} | {'Moyenne SA Classique':<20} | {'Moyenne ES Agent':<20}")
    print("-" * 50)

    for nb_objets in tailles_problemes:
        scores_sa = []
        scores_es = []

        agent_es = ESAgent(n_items=nb_objets, n_steps_per_episode=n_steps_sa)
        

        # On va chercher le fichier soit dans 'yanis/', soit à la racine
        fichier_dans_yanis = f"yanis/poids_agent_es_{nb_objets}.npy"
        fichier_racine = f"poids_agent_es_{nb_objets}.npy"
        
        if os.path.exists(fichier_racine):
            fichier_poids = fichier_racine
        elif os.path.exists(fichier_dans_yanis):
            fichier_poids = fichier_dans_yanis
        else:
            fichier_poids = None

        if fichier_poids:
            # On utilise le chemin absolu pour éviter le bug de concaténation de numpy
            chemin_absolu = os.path.abspath(fichier_poids)
            try:
                agent_es.load(chemin_absolu)
            except Exception as e:
                # Si le load interne recrée le double "yanis\yanis", on force le nom court
                agent_es.load(f"poids_agent_es_{nb_objets}.npy")
        else:
            print(f" /!\\ Poids non trouvés pour N={nb_objets}. L'agent jouera avec des poids à 0.")

        for seed in range(nb_seeds):

            probleme = generer_probleme(nb_objets, seed=seed)

            sa_classique = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa)
            _, meilleure_energie_c, _ = sa_classique.solve(log_filename="nul")
            scores_sa.append(-meilleure_energie_c) 
            sa_es = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=n_steps_sa, agent=agent_es)
            _, meilleure_energie_es, _ = sa_es.solve(log_filename="nul")
            scores_es.append(-meilleure_energie_es)

        moyenne_sa = np.mean(scores_sa)
        moyenne_es = np.mean(scores_es)
        amelioration = ((moyenne_es - moyenne_sa) / moyenne_sa) * 100

        print(f"KNAP{nb_objets:<6} | {moyenne_sa:<18.2f} | {moyenne_es:<18.2f} | +{amelioration:.2f}%")

    print("-" * 50)

if __name__ == "__main__":
    run_stats()