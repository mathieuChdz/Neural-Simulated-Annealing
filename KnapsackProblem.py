
import random
from OptimizationProblem import OptimizationProblem

class KnapsackProblem(OptimizationProblem):
    def __init__(self, poids, valeur, max_capacity):
        self.poids = poids
        self.valeur = valeur
        self.max_capacity = max_capacity
        self.n_items = len(poids)
        
    def etat_initial(self):
        return [0] * self.n_items 
        
    def voisinage(self, state):
        voisin = list(state)
        idx_to_flip = random.randint(0, self.n_items - 1)
        voisin[idx_to_flip] = 1 - voisin[idx_to_flip] # 0 vers 1 et 1 vers 0
        return voisin, idx_to_flip 
        
    def energy(self, state):
        total_weight = sum(s * w for s, w in zip(state, self.poids))
        total_value = sum(s * v for s, v in zip(state, self.valeur))
        
        if total_weight > self.max_capacity:
            return 10000 + (total_weight - self.max_capacity) * 100 # pour faire la difference entre les mauvaise solution et les solution encore plus mauvaise
            
        return -total_value