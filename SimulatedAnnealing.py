import math
import random
import matplotlib.pyplot as plt
import numpy as np

class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=100.0, final_temp=0.1, n_steps=1000, agent=None):
        self.problem = problem
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_steps = n_steps
        self.alpha = (final_temp / initial_temp) ** (1 / n_steps) # TK=Ti*a^K => a=(TK/Ti)^(1/K) temperature a letape k = temperature a letat initial * alpha^k par definition du SA
        self.agent = agent

    def solve(self, log_filename="journal_sa.txt"):
        current_state = self.problem.etat_initial()
        current_energy = self.problem.get_energy(current_state)
        
        best_state = current_state
        best_energy = current_energy
        temp = self.initial_temp
        history_energy = [current_energy]

        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("=== DÉBUT DU RECUIT SIMULÉ ===\n")
            f.write(f"Température initiale : {self.initial_temp} | Itérations : {self.n_steps}\n\n")
            
            for k in range(self.n_steps):
                
                if self.agent is not None:
                    voisin, action = self.agent.voisinage(current_state, self.problem)
                else:
                    voisin, action = self.problem.voisinage(current_state)



                voisin_energy = self.problem.get_energy(voisin)
                
                delta_e = voisin_energy - current_energy
                
                if delta_e < 0: #car dans larticle energie en negatif
                    proba_acceptation = 1.0 
                else:
                    proba_acceptation = math.exp(-delta_e / temp)



                tirage_aleatoire = random.random()


                if delta_e < 0: #car dans larticle energie en negatif
                    accepte = True
                else:
                    if tirage_aleatoire < proba_acceptation:
                        accepte = True
                    else:
                        accepte = False
                
                

                f.write(f"--- Itération {k+1} ---\n")
                f.write(f"Température : {temp:.4f}\n")
                f.write(f"Etat x actuel : {current_state} | Énergie : {current_energy}\n")
                f.write(f"Action proposée : Flipper l'objet n°{action}\n")
                f.write(f"Etat x' voisin: {voisin} | Énergie : {voisin_energy}\n")
                f.write(f"Différence d'énergie (Delta E) : {delta_e}\n")
                f.write(f"Probabilité d'accepter : {proba_acceptation:.4f} (Tirage du hasard : {tirage_aleatoire:.4f})\n")
                

                if self.agent is not None:
                    if accepte:
                        recompense = -delta_e
                    else:
                        recompense = -10000

                    self.agent.learn(current_state, action, recompense, voisin)
            

                if accepte:
                    f.write("=> Résultat : ACCEPTÉ \n\n")
                    current_state = voisin
                    current_energy = voisin_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = current_state
                else:
                    f.write("=> Résultat : REFUSÉ \n\n")
                history_energy.append(current_energy)
                temp *= self.alpha 
                
            f.write("=== FIN DE L'OPTIMISATION ===\n")
            f.write(f"Meilleure énergie trouvée : {best_energy}\n")
            
        return best_state, best_energy, history_energy

