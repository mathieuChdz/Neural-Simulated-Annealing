import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return Categorical(logits=self.model(x))

class TemperatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # assure T>0
        )

    def forward(self, x):
        return self.model(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
    

class PPOAgent:
    def __init__(self, state_dim=100, action_dim=1000, lr=3e-4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.temp_net = TemperatureNetwork()
        self.value = ValueNetwork(state_dim)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) +
            list(self.value.parameters()) +
            list(self.temp_net.parameters()),
            lr=lr
        )
        self.memory = []

    def act(self, state):

        state = torch.FloatTensor(state).unsqueeze(0)

        dist = self.policy(state)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach()

    def act_temperature(self, progress):
        x = torch.FloatTensor([[progress]])
        t = self.temp_net(x)
        return max(0.1, t.item())

    def store(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))


    def update(self, epochs=4):

        if len(self.memory) == 0:
            return

        states = torch.FloatTensor([m[0] for m in self.memory])
        actions = torch.LongTensor([m[1] for m in self.memory])
        old_log_probs = torch.stack([m[2] for m in self.memory])
        rewards = torch.FloatTensor([m[3] for m in self.memory])

        # Value predictions
        values = self.value(states).squeeze()

        # Advantage
        advantages = rewards - values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):

            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = self.value(states).squeeze()
            value_loss = (rewards - value_pred).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def save(self, path):
        torch.save({
            'policy_state': self.policy.state_dict(),
            'value_state': self.value.state_dict(),
            'temp_state': self.temp_net.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.value.load_state_dict(checkpoint['value_state'])
        self.temp_net.load_state_dict(checkpoint['temp_state'])