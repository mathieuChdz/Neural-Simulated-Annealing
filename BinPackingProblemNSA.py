import torch
from NSAProblem import NSAProblem


class BinPackingProblemNSA(NSAProblem):

    def __init__(self, weights, capacity, n_bins):

        self.weights = weights
        self.capacity = capacity
        self.n_items = len(weights)
        self.n_bins = n_bins

    def etat_initial(self):

        # état initial stable : chaque objet dans son bin
        return list(range(self.n_items))

    def energy(self, state):

        bins = {}

        for i, b in enumerate(state):
            bins.setdefault(b, 0)
            bins[b] += self.weights[i]

        for load in bins.values():
            if load > self.capacity:
                return 1000 + (load - self.capacity) * 10

        return len(bins)

    def apply_action(self, state, action):

        i, j = action

        if state[i] == j:
            return state.copy()

        new_state = state.copy()
        new_state[i] = j

        
        loads = self.compute_bin_loads(new_state)

        if loads[j] > self.capacity:
            return state.copy()

        return new_state

    def action_space(self):
        return self.n_items * self.n_bins

    def compute_bin_loads(self, state):

        loads = [0] * self.n_bins

        for i, b in enumerate(state):
            loads[b] += self.weights[i]

        return loads

    def state_to_tensor(self, state, temperature):

        loads = self.compute_bin_loads(state)

        features = []

        for i in range(self.n_items):

            bin_i = state[i]

            wi = self.weights[i] / self.capacity
            free = (self.capacity - loads[bin_i]) / self.capacity
            t = temperature

            features.append([wi, free, t])

        return torch.tensor(features, dtype=torch.float32)
