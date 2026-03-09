import torch
import torch.nn as nn

class TemperatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # ensures T > 0
        )

    def forward(self, progress):
        return self.model(progress)