import random
from KnapsackProblemNSA import KnapsackProblem as kpnsa
from KnapsackProblem import KnapsackProblem as kp
from SimulatedAnnealing import SimulatedAnnealing

from agents.ppo import PPOAgent


def main():
    random.seed(62)
    nb_objets = 100
    poids = [random.uniform(0, 1) for _ in range(nb_objets)]
    valeurs = [random.uniform(0, 1) for _ in range(nb_objets)]
    # capacite_max = int(sum(poids) * 0.4)
    capacite_max = nb_objets/4

    probleme = kpnsa(poids, valeurs, capacite_max)
    
    state_dim = nb_objets + 3   # sélection + poids_norm + valeur_norm + temp
    action_dim = probleme.action_space()

    agent = PPOAgent(state_dim, action_dim)

    agent.load("agents/ppo_model_100.pth")

    res_sum = 0

    for i in range(5):

        sa = SimulatedAnnealing(probleme, initial_temp=1.0, final_temp=0.1, n_steps=5000, agent=agent)
        
        meilleur_etat, meilleure_energie, _ = sa.solve()

        print(f"Run {i+1} | Meilleure énergie : {-meilleure_energie:.4f}")

        res_sum += -meilleure_energie
    
    res_mean = res_sum / 5
 
    
    print("\n--- RÉSULTATS NSA ---")
    print(f"Meilleure configuration : {meilleur_etat}")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat, poids))} / {capacite_max}")
    print(f"Valeur totale : {res_mean}")

    print("\n--- DÉTAIL DES OBJETS ---")
    for i in range(nb_objets):
        statut = "Pris" if meilleur_etat[i] == 1 else "Laissé"
        print(f"Objet {i+1:2d} | Poids : {poids[i]:2.2f} | Valeur : {valeurs[i]:2.2f} | {statut}")
    print("-------------------------")


if __name__ == "__main__":
    main()