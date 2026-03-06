import random
from KnapsackProblemNSA import KnapsackProblem as kpnsa
from KnapsackProblem import KnapsackProblem as kp
from SimulatedAnnealing import SimulatedAnnealing

from agents.ppo import PPOAgent


def main():
    random.seed(62) 
    nb_objets = 100
    poids = [random.randint(1, 20) for _ in range(nb_objets)]
    valeurs = [random.randint(10, 50) for _ in range(nb_objets)]
    capacite_max = int(sum(poids) * 0.4)

    probleme = kpnsa(poids, valeurs, capacite_max)
    
    state_dim = nb_objets + 3   # sélection + poids_norm + valeur_norm + temp
    action_dim = probleme.action_space()

    agent = PPOAgent(state_dim, action_dim)

    agent.load("agents/ppo_model.pth")

    sa = SimulatedAnnealing(probleme, initial_temp=100.0, final_temp=0.1, n_steps=2000, agent=agent)
    
    meilleur_etat, meilleure_energie, _ = sa.solve()
 
    
    print("\n--- RÉSULTATS NSA ---")
    print(f"Meilleure configuration : {meilleur_etat}")
    print(f"Poids total : {sum(s * p for s, p in zip(meilleur_etat, poids))} / {capacite_max}")
    print(f"Valeur totale : {-meilleure_energie}")

    print("\n--- DÉTAIL DES OBJETS ---")
    for i in range(nb_objets):
        statut = "Pris" if meilleur_etat[i] == 1 else "Laissé"
        print(f"Objet {i+1:2d} | Poids : {poids[i]:2d} | Valeur : {valeurs[i]:2d} | {statut}")
    print("-------------------------")


if __name__ == "__main__":
    main()