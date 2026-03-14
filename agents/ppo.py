import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

    def forward(self, state):
        logits = self.model(state)
        return Categorical(logits=logits)
    

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state):
        return self.model(state)
    
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + 
            list(self.value.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.eps_clip = eps_clip

        self.memory = []
    
    def update(self, epochs=4):

    # ====== 1. Récupérer mémoire ======
        states = torch.FloatTensor([m[0] for m in self.memory])
        actions = torch.LongTensor([m[1] for m in self.memory])
        old_log_probs = torch.stack([m[2] for m in self.memory]).detach()
        rewards = [m[3] for m in self.memory]
        values = torch.FloatTensor([m[4] for m in self.memory])

        # ====== 2. Calcul des returns ======
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ====== 3. Update PPO ======
        for _ in range(epochs):

            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            values = self.value(states).squeeze()

            advantages = self.compute_gae(rewards, values, self.gamma, 0.95)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)

            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ====== 4. Reset mémoire ======
        self.memory = []

    
    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):

        advantages = []
        gae = 0

        values = values.tolist()
        values.append(0)

        for t in reversed(range(len(rewards))):

            delta = rewards[t] + gamma * values[t+1] - values[t]

            gae = delta + gamma * lam * gae

            advantages.insert(0, gae)

        return torch.FloatTensor(advantages)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.value(state)

        return action.item(), log_prob.detach(), value.detach().item()
    
    def store(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))


    def save(self, path):
        torch.save({
            'policy_state': self.policy.state_dict(),
            'value_state': self.value.state_dict(),
        }, path)
    

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.value.load_state_dict(checkpoint['value_state'])