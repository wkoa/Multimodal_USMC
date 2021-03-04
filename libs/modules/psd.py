import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PSD(nn.Module):
    def __init__(self, name_modality, cfg):
        super(PSD, self).__init__()
        self.linear = nn.Linear(2000, cfg.FUSION_HEAD.FEATURE_NUM)

    def forward(self, x):
        feature = self.feature.process(x)
        y = self.linear(feature)
        y = F.relu(y, inplace=True)

        return y