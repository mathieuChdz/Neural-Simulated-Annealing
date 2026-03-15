import numpy as np
import torch
import random


class ESAgent:

    def __init__(self, n_items, n_bins, sigma=0.1, pop_size=10):

        self.n_items = n_items
        self.n_bins = n_bins

        self.sigma = sigma
        self.pop_size = pop_size

        # simple policy : matrice poids
        self.weights = np.random.randn(n_items, n_bins)

        self.noises = []
        self.rewards = []

    def act(self, state):

        # state shape : [n_items, features]
        state = state.numpy()

        scores = self.weights

        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

        item = random.randint(0, self.n_items - 1)
        bin_choice = np.random.choice(self.n_bins, p=probs[item])

        return (item, bin_choice), torch.tensor(0.0)

    def store(self, state, action, log_prob, reward, done):

        self.rewards.append(reward)

    def update(self):

        if len(self.rewards) == 0:
            return

        reward = np.mean(self.rewards)

        noise = np.random.randn(*self.weights.shape)

        self.weights += self.sigma * reward * noise

        self.rewards = []

    def save(self, path):

        np.save(path, self.weights)

    def load(self, path):

        self.weights = np.load(path)
