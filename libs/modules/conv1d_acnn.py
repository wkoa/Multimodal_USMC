import torch
from torch import nn
from torch.nn import functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def statistic_pooling(emb, embedding=True):
    mean = torch.mean(emb, dim=2, keepdim=True)
    k = ((emb - mean + 1e-9) ** 2).sum(dim=2)
    std = k.sqrt()
    mean = mean.squeeze()
    # std = torch.std(emb, dim=2)
    if embedding:
        seg_emb = torch.cat([mean, std], dim=1)
        return seg_emb
    else:
        return mean, std


class ACNNet(nn.Module):
    def __init__(self, modality_name, feature_dim=256, stat_pooling=False):
        super(ACNNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 225, stride=3, dilation=1)
        self.pool1 = nn.MaxPool1d(5)

        self.stage1 = nn.Conv1d(16, 32, 5, dilation=1)
        self.pool2 = nn.MaxPool1d(3)
        self.stage2 = nn.Conv1d(32, 32, 3, dilation=2)
        self.stage3 = nn.Conv1d(32, 32, 3, dilation=3)
        self.stage4 = nn.Conv1d(32, 64, 1)
        # self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        self.norm0 = nn.BatchNorm1d(16)
        self.norm1 = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(32)
        self.norm4 = nn.BatchNorm1d(64)
        # self.norm5 = nn.BatchNorm1d(3 * 512)
        # self.norm6 = nn.BatchNorm1d(feature_dim)

        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.stat_pool = stat_pooling
        if self.stat_pool:
            self.fc = nn.Linear(3 * 512 * 2, feature_dim)

        else:
            if modality_name == "sound":
                self.fc = nn.Linear(61632, feature_dim)
            elif modality_name == "force":
                self.fc = nn.Linear(20288, feature_dim)
            else:
                self.fc = nn.Linear(3200, feature_dim)

    def forward(self, x):
        # x = self.norm0(F.relu(self.conv1(x), inplace=True))
        # x = self.pool1(x)
        #
        # x = self.norm1(F.relu(self.stage1(x), inplace=True))
        # x = self.pool2(x)
        #
        # x = self.norm2(F.relu(self.stage2(x), inplace=True))
        #
        # x = self.norm3(F.relu(self.stage3(x), inplace=True))
        #
        # x = self.norm4(F.relu(self.stage4(x), inplace=True))
        #
        # x = self.norm5(F.relu(self.stage5(x), inplace=True))

        x = self.norm0(F.relu(self.conv1(x), inplace=True))
        x = self.pool1(x)

        x = self.norm1(F.relu(self.stage1(x), inplace=True))
        x = self.pool2(x)

        x = self.norm2(F.relu(self.stage2(x), inplace=True))

        x = self.norm3(F.relu(self.stage3(x), inplace=True))

        x = self.norm4(F.relu(self.stage4(x), inplace=True))

        # x = self.norm5(F.relu(self.stage5(x), inplace=True))

        if self.stat_pool:
            x = statistic_pooling(x)
        else:
            # x = self.avg_pool(x)
            x = x.flatten(1, -1)
        x = self.fc(x)

        return x
