import torch
import torch.optim as optim

from agents.policy_network import ItemPolicy, BinPolicy
from torch.distributions import Categorical


class PPOAgent:

    def __init__(self, lr=3e-4):

        self.item_policy = ItemPolicy()
        self.bin_policy = BinPolicy()

        self.optimizer = optim.Adam(
            list(self.item_policy.parameters()) +
            list(self.bin_policy.parameters()),
            lr=lr
        )

        self.memory = []

    def act(self, state):

        item_dist = self.item_policy(state)
        item = item_dist.sample()

        log_prob_i = item_dist.log_prob(item)

        bin_dist = self.bin_policy(state)
        bin_choice = bin_dist.sample()

        log_prob_j = bin_dist.log_prob(bin_choice)

        log_prob = log_prob_i + log_prob_j

        return (item.item(), bin_choice.item()), log_prob.detach()

    def store(self, state, action, log_prob, reward, done):

        self.memory.append((state, action, log_prob, reward, done))

    def update(self, epochs=4):

        if len(self.memory) == 0:
            return

        states = torch.stack([m[0] for m in self.memory])
        actions = [m[1] for m in self.memory]
        old_log_probs = torch.stack([m[2] for m in self.memory])
        rewards = torch.tensor([m[3] for m in self.memory], dtype=torch.float32)

        for _ in range(epochs):

            log_probs = []

            for i, state in enumerate(states):

                item, bin_choice = actions[i]

                item_dist = self.item_policy(state)
                bin_dist = self.bin_policy(state)

                log_prob = item_dist.log_prob(
                    torch.tensor(item)
                ) + bin_dist.log_prob(
                    torch.tensor(bin_choice)
                )

                log_probs.append(log_prob)

            log_probs = torch.stack(log_probs)

            # loss PPO : -E[ min(ratio * reward, clip(ratio, 1-eps, 1+eps) * reward) ] 
            epsilon = 0.2

            ratio = torch.exp(log_probs - old_log_probs)

            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

            loss = -torch.min(              ratio * rewards,              clipped_ratio * rewards          ).mean()

            # version sans clipping
            #ratio = torch.exp(log_probs - old_log_probs)

            #loss = -(ratio * rewards).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def save(self, path):

        torch.save({
            'item_policy': self.item_policy.state_dict(),
            'bin_policy': self.bin_policy.state_dict()
        }, path)

    def load(self, path):

        checkpoint = torch.load(path)

        self.item_policy.load_state_dict(checkpoint['item_policy'])
        self.bin_policy.load_state_dict(checkpoint['bin_policy'])