import random
import numpy as np
import os

class ESAgent:
    def __init__(self, n_items, n_steps_per_episode, pop_size=10, sigma=0.1, lr=0.05):
        self.n_items = n_items
        self.n_steps_per_episode = n_steps_per_episode
        self.pop_size = pop_size  # Nombre de mutants 
        self.sigma = sigma        # bruit
        self.lr = lr              # Taux d'apprentissage
        
        self.theta = np.zeros(4) # Les poids de base
        
        self._reset_population()
        
    def _reset_population(self):

        # On génère du bruit (mutations) pour toute la population
        self.noises = [np.random.randn(4) for _ in range(self.pop_size)]
        self.rewards = []
        self.current_mutant = 0
        self.current_reward = 0
        self.step_counter = 0

    def voisinage(self, state, problem):
        # 1. On récupère les poids du réseau muté actuel
        if self.current_mutant < self.pop_size:
            current_theta = self.theta + self.sigma * self.noises[self.current_mutant]
        else:
            current_theta = self.theta # Mode test (sans bruit)
            
        # 2. Le réseau calcule un score pour chaque objet
        poids_actuel = sum(s * w for s, w in zip(state, problem.poids))
        place_restante = problem.max_capacity - poids_actuel
        
        scores = []
        for i in range(problem.n_items):
            f1 = problem.poids[i] / problem.max_capacity       # Poids relatif
            f2 = problem.valeur[i] / max(problem.valeur)     # Valeur relative
            f3 = place_restante / problem.max_capacity         # Place restante
            f4 = state[i]
            
            # Le réseau donne une note à l'objet (Produit scalaire)
            score = current_theta[0]*f1 + current_theta[1]*f2 + current_theta[2]*f3 + current_theta[3]*f4
            scores.append(score)
            
        action = np.argmax(scores)
        
        # On construit le voisin
        voisin = list(state)
        voisin[action] = 1 - voisin[action]
        return voisin, int(action)

    def learn(self, state, action, reward, next_state, problem):
        self.current_reward += reward
        self.step_counter += 1
        
        # Si le Recuit Simulé est terminé (on a atteint n_steps)
        if self.step_counter >= self.n_steps_per_episode:
            self.rewards.append(self.current_reward)
            self.current_mutant += 1
            
            # On remet les compteurs à zéro pour le prochain mutant
            self.current_reward = 0
            self.step_counter = 0
            
            # Si toute la population de mutants a été évaluée, on met à jour l'intelligence centrale !
            if self.current_mutant == self.pop_size:
                self._update_theta()
                self._reset_population() # Prépare la génération suivante

    def _update_theta(self):

        R = np.array(self.rewards)
        
        # Normalisation des récompenses (Fitness shaping, crucial en ES)
        if np.std(R) != 0:
            A = (R - np.mean(R)) / np.std(R)
        else:
            A = np.zeros_like(R)
            
        # On calcule le gradient estimé
        gradient = np.zeros(4)
        for i in range(self.pop_size):
            gradient += A[i] * self.noises[i]
            
        # On met à jour le cerveau central
        self.theta = self.theta + self.lr / (self.pop_size * self.sigma) * gradient
    
    def save(self, filename="poids_agent_es.npy"):
        chemin = os.path.join("yanis", filename)
        np.save(chemin, self.theta)
        print(f"Poids sauvegardés dans {chemin}")


    def load(self, filename="poids_agent_es.npy"):
        chemin = os.path.join("yanis", filename)
        self.theta = np.load(chemin)
        print(f"Poids chargés depuis {chemin}")