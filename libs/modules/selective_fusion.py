from torch import nn


class SelectiveFusion(nn.Module):
    def __init__(self, cfg):
        super(SelectiveFusion, self).__init__()
        self.cfg = cfg

    def forward(self, modalities):
        pass


