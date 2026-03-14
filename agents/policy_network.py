import torch
import torch.nn as nn
from torch.distributions import Categorical


class ItemPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features):

        logits = self.mlp(features).squeeze(-1)

        return Categorical(logits=logits)


class BinPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features):

        logits = self.mlp(features).squeeze(-1)

        return Categorical(logits=logits)