
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

        penalty = max(0, total_weight - self.max_capacity)

        return -total_value + 20 * penalty

    def apply_action(self, state, action):
        new_state = list(state)
        new_state[action] = 1 - new_state[action]
        return new_state
    
    def state_to_tensor(self, state, temperature):

        #pour chaque item i : [x_i, w_i, v_i, W, T]
        matrix = []
        for i in range(self.n_items):
            item_features = [
                float(state[i]),
                self.poids[i],
                self.valeur[i],
                self.max_capacity,
                temperature
            ]
            matrix.append(item_features)
        
        return {
            'x': state,
            'w': self.poids,
            'v': self.valeur,
            'W': self.max_capacity,
            'temp': temperature
        }

    def action_space(self):
        return self.n_items