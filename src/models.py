import torch
import torch.nn as nn

class SpacechipTitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super(SpacechipTitanicMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16,8),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)