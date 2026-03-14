import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class KnapsackNetwork(nn.Module):
    def __init__(self, input_dim=5): # pour chanque objet[xi, wi, vi, W, T]
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)
    
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):

        self.actor = KnapsackNetwork(5)
        self.critic = KnapsackNetwork(5) # Même architecture pour le critic [cite: 526]
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            weight_decay=1e-2
        )

        self.gamma = gamma
        self.eps_clip = eps_clip

        self.memory = []
    
    def update(self, epochs=4):
        if len(self.memory) == 0:
            return

        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.long)
        old_log_probs = torch.stack([m[2] for m in self.memory]).detach()
        rewards = torch.tensor([m[3] for m in self.memory], dtype=torch.float32)
        memory_states = [m[0] for m in self.memory]

        for _ in range(epochs):
            new_log_probs = []
            
            for i, state_dict in enumerate(memory_states):
                x = torch.tensor(state_dict['x'], dtype=torch.float32)
                w = torch.tensor(state_dict['w'], dtype=torch.float32)
                v = torch.tensor(state_dict['v'], dtype=torch.float32)
                n = len(x)
                cap = torch.full((n, 1), state_dict['W'], dtype=torch.float32)
                temp = torch.full((n, 1), state_dict['temp'], dtype=torch.float32)
                
                inputs = torch.cat([torch.stack([x, w, v], dim=1), cap, temp], dim=1)

                logits = self.actor(inputs).squeeze(-1)
                
                current_w = (x * w).sum()
                mask = (x == 0) & (current_w + w > state_dict['W'])
                logits[mask] = -1e10
                
                dist = Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(actions[i]))

            new_log_probs = torch.stack(new_log_probs)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * rewards
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def act(self, state_dict):

        x_bits = torch.FloatTensor(state_dict['x'])
        weights = torch.FloatTensor(state_dict['w'])
        values = torch.FloatTensor(state_dict['v'])
        n = len(x_bits)
        cap = torch.full((n, 1), state_dict['W'])
        temp = torch.full((n, 1), state_dict['temp'])

        inputs = torch.stack([x_bits, weights, values], dim=1)
        inputs = torch.cat([inputs, cap, temp], dim=1)

        logits = self.actor(inputs).squeeze(-1)

        current_weight = (x_bits * weights).sum()
        for i in range(n):
            if x_bits[i] == 0 and (current_weight + weights[i] > state_dict['W']):
                logits[i] = -1e10 #proba tres faible pour les actions qui depassent la cap max
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        state_value = self.critic(inputs.mean(dim=0))

        return action.item(), log_prob.detach(), state_value.detach().item()
    
    
    def store(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))


    def save(self, path):
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])