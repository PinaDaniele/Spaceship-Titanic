import torch.nn as nn
from torchinfo import summary

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

            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    model = SpacechipTitanicMLP(29)
    summary(model, input_size=(256,29))