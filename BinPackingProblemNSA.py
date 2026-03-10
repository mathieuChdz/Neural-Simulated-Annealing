import random
import numpy as np

class BinPackingProblemNSA:
    def __init__(self, items, bin_capacity, n_bins):
        self.items = items
        self.bin_capacity = bin_capacity
        self.n_bins = n_bins
        self.n_items = len(items)

    def etat_initial(self):
        return [random.randint(0, self.n_bins-1) for _ in range(self.n_items)]

    def action_space(self):
        return self.n_items * self.n_bins

    def state_to_tensor(self, state, temp):
        tensor = []
        for b in state:
            one_hot = [0] * self.n_bins
            one_hot[b] = 1
            tensor.extend(one_hot)
        tensor.append(temp / 100.0)  # normaliser la température
        return np.array(tensor, dtype=np.float32)

    def apply_action(self, state, action):
        new_state = state.copy()
        item_idx = action // self.n_bins
        bin_idx = action % self.n_bins
        new_state[item_idx] = bin_idx
        return new_state

    def energy(self, state):
        bins = [[] for _ in range(self.n_bins)]
        for idx, b in enumerate(state):
            bins[b].append(self.items[idx])
        energy = 0
        for b in bins:
            total = sum(b)
            if total > self.bin_capacity:
                energy += 1000 + (total - self.bin_capacity)*10
        used_bins = sum(1 for b in bins if len(b) > 0)
        return energy + used_bins