import random
from OptimizationProblem import OptimizationProblem


class BinPackingProblem(OptimizationProblem):

    def __init__(self, weights, capacity):

        self.weights = weights
        self.capacity = capacity
        self.n_items = len(weights)
        self.n_bins = self.n_items

    def etat_initial(self):

        # chaque objet dans son bin
        

        bins = []
        state = [-1] * self.n_items

        for i, w in enumerate(self.weights):

            placed = False

            for b, load in enumerate(bins):

                if load + w <= self.capacity:

                    bins[b] += w
                    state[i] = b
                    placed = True
                    break

            if not placed:

                bins.append(w)
                state[i] = len(bins) - 1

        return state

        #return list(range(self.n_items))

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