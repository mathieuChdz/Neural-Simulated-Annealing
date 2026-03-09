import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return Categorical(logits=self.model(x))

class TemperatureNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # T > 0
        )

    def forward(self, x):
        return self.model(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.temp_net = TemperatureNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.temp_net.parameters()), lr=lr
        )
        self.memory = []

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach()

    def act_temperature(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        t = self.temp_net(state)
        return t.item()

    def store(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))

    def update(self, epochs=4):
        if len(self.memory) == 0:
            return

        states = torch.FloatTensor(np.array([m[0].squeeze() if isinstance(m[0], torch.Tensor) else m[0] for m in self.memory]))
        actions = torch.LongTensor([m[1] for m in self.memory])
        old_log_probs = torch.stack([m[2] for m in self.memory])
        rewards = [m[3] for m in self.memory]

        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(epochs):
            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            advantage = returns
            loss = -torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def save(self, path):
        torch.save({
            'policy_state': self.policy.state_dict(),
            'temp_state': self.temp_net.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.temp_net.load_state_dict(checkpoint['temp_state'])