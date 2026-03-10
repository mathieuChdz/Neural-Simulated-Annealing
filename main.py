import random
from KnapsackProblem import KnapsackProblem
from SimulatedAnnealing import SimulatedAnnealing

def main():
    random.seed(2)
    nb_objets = 100
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]
    # capacite_max = int(sum(poids) * 0.4)
    capacite_max = nb_objets/4

    probleme = KnapsackProblem(poids, valeurs, capacite_max)
    
    res_sum = 0

    for i in range(5):

        sa = SimulatedAnnealing(probleme, initial_temp=1.0, final_temp=0.1, n_steps=5000)
        
        meilleur_etat, meilleure_energie, _ = sa.solve()

        print(f"Run {i+1} | Meilleure énergie : {-meilleure_energie:.4f}")

        res_sum += -meilleure_energie
    
    res_mean = res_sum / 5

    print("\n--- RÉSULTATS ---")
    print(f"Meilleure configuration (1=pris, 0=laissé) : {meilleur_etat}")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat, poids))} / {capacite_max}")
    print(f"Valeur totale du sac : {res_mean:.4f}")
    

    print("\n--- DÉTAIL DES OBJETS ---")
    for i in range(nb_objets):
        statut = "Pris" if meilleur_etat[i] == 1 else "Laissé"
        print(f"Objet {i+1:2d} | Poids : {poids[i]:2.2f} | Valeur : {valeurs[i]:2.2f} | Statut : {statut}")
    print("-------------------------")


if __name__ == "__main__":
    main()