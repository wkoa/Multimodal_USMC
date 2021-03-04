from torch import nn

from libs.modules import *


class SelectiveNet(nn.Module):
    def __init__(self, cfg):
        super(SelectiveNet, self).__init__()

        self.multi_extractor = MultiExtractor(cfg)
        self.selective_fusion = EarlyFusion(cfg)
        if cfg.MODEL.DEVICE == "cuda":
            self.earyly_fusion = self.earyly_fusion.cuda()

    def forward(self, x):
        pass