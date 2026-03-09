import random
from OptimizationProblem import OptimizationProblem


class BinPackingProblem(OptimizationProblem):

    def __init__(self, weights, capacity):

        self.weights = weights
        self.capacity = capacity
        self.n_items = len(weights)

    def etat_initial(self):

        # chaque objet dans son bin
        return list(range(self.n_items))

    def voisinage(self, state):

        voisin = list(state)

        item = random.randint(0, self.n_items - 1)
        new_bin = random.randint(0, self.n_items - 1)

        voisin[item] = new_bin

        return voisin, item

    def energy(self, state):

        bins = {}

        for i, b in enumerate(state):
            bins.setdefault(b, 0)
            bins[b] += self.weights[i]

        for load in bins.values():
            if load > self.capacity:
                return 1000 + (load - self.capacity) * 10

        return len(bins)