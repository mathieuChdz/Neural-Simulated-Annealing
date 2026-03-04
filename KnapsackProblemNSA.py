
import random

import numpy as np
from NSAProblem import NSAProblem

class KnapsackProblem(NSAProblem):
    def __init__(self, poids, valeur, max_capacity):
        self.poids = poids
        self.valeur = valeur
        self.max_capacity = max_capacity
        self.n_items = len(poids)
        
    def etat_initial(self):
        return [0] * self.n_items 
        
    def energy(self, state):
        total_weight = sum(s * w for s, w in zip(state, self.poids))
        total_value = sum(s * v for s, v in zip(state, self.valeur))
        
        if total_weight > self.max_capacity:
            return 1000 + (total_weight - self.max_capacity) * 10 # pour faire la difference entre les mauvaise solution et les solution encore plus mauvaise
            
        return -total_value

    def apply_action(self, state, action):
        new_state = list(state)
        new_state[action] = 1 - new_state[action]
        return new_state
    
    def state_to_tensor(self, state, temperature):
        
        total_weight = sum(s * w for s, w in zip(state, self.poids))
        total_value = sum(s * v for s, v in zip(state, self.valeur))

        temp_norm = temperature  # idéalement normalisé dans SA

        # vecteur = sélection + poids total + valeur totale + température
        state_vector = (
            list(state)
            + [total_weight / self.max_capacity]
            + [total_value / (sum(self.valeur) + 1e-8)]
            + [temp_norm]
        )

        return np.array(state_vector, dtype=np.float32)

    def action_space(self):
        return self.n_items