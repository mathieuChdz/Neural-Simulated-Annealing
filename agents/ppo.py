import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# === Réseau léger pour sélectionner un item ===
class ItemPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),  # [wi, b(i), T]
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Réseau léger pour sélectionner un bin conditionnel sur l'item ===
class BinPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),  # [wi, cj, T]
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

class PPOAgent:
    def __init__(self, n_items, n_bins, lr=3e-4):
        self.n_items = n_items
        self.n_bins = n_bins
        self.item_net = ItemPolicyNetwork()
        self.bin_net = BinPolicyNetwork()
        self.optimizer = optim.Adam(
            list(self.item_net.parameters()) + list(self.bin_net.parameters()), lr=lr
        )
        self.memory = []

    # Échantillonnage ancestral
    def act(self, state_tensor):
        """
        state_tensor : dict contenant
            - 'weights' : list poids items
            - 'bins' : current bins des items
            - 'temp' : temperature float
        """
        wi_list = state_tensor['weights']
        bi_list = state_tensor['bins']
        T = state_tensor['temp']

        # === Item selection ===
        item_logits = []
        for i, wi in enumerate(wi_list):
            x = torch.tensor([wi, bi_list[i], T], dtype=torch.float32)
            item_logits.append(self.item_net(x))
        item_logits = torch.cat(item_logits)
        item_probs = torch.softmax(item_logits, dim=0)
        item_dist = Categorical(item_probs)
        i = item_dist.sample()
        log_prob_item = item_dist.log_prob(i)

        # === Bin selection conditionnelle ===
        cj_list = state_tensor['bin_remaining']  # list of remaining capacities
        bin_logits = []
        for j, cj in enumerate(cj_list):
            x = torch.tensor([wi_list[i], cj, T], dtype=torch.float32)
            bin_logits.append(self.bin_net(x))
        bin_logits = torch.cat(bin_logits)
        bin_probs = torch.softmax(bin_logits, dim=0)
        bin_dist = Categorical(bin_probs)
        j = bin_dist.sample()
        log_prob_bin = bin_dist.log_prob(j)

        # Stockage pour PPO
        self.last_state = state_tensor
        self.last_action = (i.item(), j.item())
        self.last_log_prob = log_prob_item + log_prob_bin

        return (i.item(), j.item()), self.last_log_prob.detach()

    def store(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))

    def update(self, epochs=4, gamma=0.99):
        if len(self.memory) == 0:
            return

        # Préparer batch
        returns = []
        G = 0
        for _, _, _, reward, _ in reversed(self.memory):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(epochs):
            losses = []
            for idx, (state, action, log_prob, reward, done) in enumerate(self.memory):
                _, new_log_prob = self.act(state)
                ratio = torch.exp(new_log_prob - log_prob)
                advantage = returns[idx]
                loss = -torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
                losses.append(loss)
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def save(self, path):
        torch.save({
            'item_state': self.item_net.state_dict(),
            'bin_state': self.bin_net.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.item_net.load_state_dict(checkpoint['item_state'])
        self.bin_net.load_state_dict(checkpoint['bin_state'])