import numpy as np
import random
from OptimizationProblem import OptimizationProblem

class BinPackingProblemNSA(OptimizationProblem):
    def __init__(self, items, bin_capacity, n_bins):
        self.items = items
        self.bin_capacity = bin_capacity
        self.n_bins = n_bins
        self.n_items = len(items)

    def etat_initial(self):
        # Commence avec chaque item dans un bin aléatoire
        return [random.randint(0, self.n_bins - 1) for _ in range(self.n_items)]

    def action_space(self):
        # Action = item_idx * n_bins + bin_idx
        return self.n_items * self.n_bins

    def state_to_tensor(self, state, temp=None):
        # vectorise le state (1-hot pour chaque item) + optionnellement temperature (ignored)
        tensor = []
        for b in state:
            one_hot = [0] * self.n_bins
            if b >= 0:
                one_hot[b] = 1
            tensor.extend(one_hot)
        return np.array(tensor, dtype=np.float32)

    def apply_action(self, state, action):
        item_idx = action // self.n_bins
        bin_idx = action % self.n_bins
        new_state = list(state)
        new_state[item_idx] = bin_idx
        return new_state

    def energy(self, state):
        bins = [[] for _ in range(self.n_bins)]
        for i, b in enumerate(state):
            bins[b].append(self.items[i])
        energy = 0
        for b in bins:
            total = sum(b)
            if total > self.bin_capacity:
                energy += 1000 + (total - self.bin_capacity) * 10
        used_bins = sum(1 for b in bins if len(b) > 0)
        return energy + used_bins