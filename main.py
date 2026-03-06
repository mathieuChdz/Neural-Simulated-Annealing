import random
from KnapsackProblem import KnapsackProblem
from SimulatedAnnealing import SimulatedAnnealing

def main():
    random.seed(62) 
    nb_objets = 100
    poids = [random.randint(1, 20) for _ in range(nb_objets)]
    valeurs = [random.randint(10, 50) for _ in range(nb_objets)]
    capacite_max = int(sum(poids) * 0.4)

    probleme = KnapsackProblem(poids, valeurs, capacite_max)
    
    sa = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=2000)
    
    meilleur_etat, meilleure_energie, historique = sa.solve()
    
    print("\n--- RÉSULTATS ---")
    print(f"Meilleure configuration (1=pris, 0=laissé) : {meilleur_etat}")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat, poids))} / {capacite_max}")
    print(f"Valeur totale du sac : {-meilleure_energie}")
    

    print("\n--- DÉTAIL DES OBJETS ---")
    for i in range(nb_objets):
        statut = "Pris" if meilleur_etat[i] == 1 else "Laissé"
        print(f"Objet {i+1:2d} | Poids : {poids[i]:2d} | Valeur : {valeurs[i]:2d} | Statut : {statut}")
    print("-------------------------")


if __name__ == "__main__":
    main()