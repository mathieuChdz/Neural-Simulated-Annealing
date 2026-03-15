import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os

class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=100.0, final_temp=0.1, n_steps=1000, agent=None):
        self.problem = problem
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_steps = n_steps
        self.alpha = (final_temp / initial_temp) ** (1 / n_steps)
        self.agent = agent

    def solve(self, log_filename="journal_sa.txt"):

        dossier_logs = "logs"
        os.makedirs(dossier_logs, exist_ok=True)
        chemin_fichier = os.path.join(dossier_logs, log_filename)

        current_state = self.problem.etat_initial()
        
        try:
            current_energy = self.problem.get_energy(current_state)
        except (NotImplementedError, AttributeError):
            current_energy = self.problem.energy(current_state)
        
        best_state = current_state
        best_energy = current_energy
        temp = self.initial_temp
        history_energy = [current_energy]

        with open(chemin_fichier, "w", encoding="utf-8") as f:
            f.write("=== DÉBUT DU RECUIT SIMULÉ ===\n")

            if self.agent is not None:
                if hasattr(self.agent, 'voisinage'):
                    f.write("Mode : NEURAL SA (Agent ES - Yanis)\n")
                else:
                    f.write("Mode : NEURAL SA (Agent RL - Mathieu)\n")
            else:
                f.write("Mode : SA CLASSIQUE\n")

            f.write(f"Température initiale : {self.initial_temp} | Itérations : {self.n_steps}\n\n")
            
            for k in range(self.n_steps):
                
                if self.agent is not None:
                    if hasattr(self.agent, 'voisinage'):
                        voisin, action = self.agent.voisinage(current_state, self.problem)
                    else:
                        temp_norm = temp / self.initial_temp
                        state_tensor = self.problem.state_to_tensor(current_state, temp_norm)
                        action, log_prob, value = self.agent.act(state_tensor)
                        voisin = self.problem.apply_action(current_state, action)
                else:
                    voisin, action = self.problem.voisinage(current_state)

                try:
                    voisin_energy = self.problem.get_energy(voisin)
                except (NotImplementedError, AttributeError):
                    voisin_energy = self.problem.energy(voisin)
                
                delta_e = voisin_energy - current_energy
                
                if delta_e < 0:
                    proba_acceptation = 1.0 
                else:
                    proba_acceptation = math.exp(-delta_e / temp)

                tirage_aleatoire = random.random()

                if delta_e < 0:
                    accepte = True
                else:
                    accepte = (tirage_aleatoire < proba_acceptation)
                
                f.write(f"--- Itération {k+1} ---\n")
                f.write(f"Température : {temp:.4f}\n")
                f.write(f"Etat x actuel : {current_state} | Énergie : {current_energy}\n")
                f.write(f"Action proposée : Flipper l'objet n°{action}\n")
                f.write(f"Etat x' voisin: {voisin} | Énergie : {voisin_energy}\n")
                f.write(f"Différence d'énergie (Delta E) : {delta_e}\n")
                f.write(f"Probabilité d'accepter : {proba_acceptation:.4f} (Tirage du hasard : {tirage_aleatoire:.4f})\n")
                
                if self.agent is not None:
                    if accepte:
                        reward = -delta_e 
                        next_state = voisin
                    else:
                        reward = 0
                        next_state = current_state

                    if hasattr(self.agent, 'learn'):
                        self.agent.learn(current_state, action, reward, next_state, self.problem)
                    elif hasattr(self.agent, 'store'):
                        is_done = (k == self.n_steps - 1)
                        self.agent.store(state_tensor, action, log_prob, reward, value, is_done)
        
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
            
            if self.agent is not None and hasattr(self.agent, 'update'):
                 self.agent.update()
                
            f.write("=== FIN DE L'OPTIMISATION ===\n")
            f.write(f"Meilleure énergie trouvée : {best_energy}\n")
            
        return best_state, best_energy, history_energy
    