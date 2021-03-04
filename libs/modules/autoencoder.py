import torch
from torch import nn
from torch.nn import functional as F


class NaiveAutoEncoder(nn.Module):
    def __init__(self, in_feature, hid_feature):
        super(NaiveAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, hid_feature),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_feature, in_feature),
            # nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)

        return out, emb


class NaiveVAE(nn.Module):
    def __init__(self, in_feature, hid_feature):
        super(NaiveVAE, self).__init__()

        self.fc1 = nn.Linear(in_feature, hid_feature)
        self.fc21 = nn.Linear(hid_feature, hid_feature)
        self.fc22 = nn.Linear(hid_feature, hid_feature)
        self.fc3 = nn.Linear(hid_feature, hid_feature)
        self.fc4 = nn.Linear(hid_feature, in_feature)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z