import torch
from torch import nn


class DALnn(nn.Module):
    def __init__(self, modality_name):
        super(DALnn, self).__init__()
        self.fc0 = nn.Linear(2000, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 40)

        self.activation = torch.nn.Tanh()

    def forward(self, x):
        # x = self.fc0(x)
        # x = self.activation(x)
        # x = x.squeeze()
        # x = PCA_svd(x, 200, center=True)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x
