from torch import nn
from .baseline import BaselineEarlyFusion


class MTUT(nn.Module):
    def __init__(self, cfg):
        super(MTUT, self).__init__()
        pass

    def forward(self, x):
        pass


class Unimodal(nn.Module):
    def __init__(self, modality_name, cfg):
        super(Unimodal, self).__init__()
        requirments = cfg.MODALITY.REQUIRMENTS
        cfg.MODALITY.REQUIRMENTS = [modality_name]
        cfg.MODALITY.NUMS = 1
        self.unimodal = BaselineEarlyFusion(cfg)
        cfg.MODALITY.REQUIRMENTS = requirments
        cfg.MODALITY.NUMS = len(requirments)

    def forward(self, x):
        out, feature, hid_feature = self.unimodal(x, True)
        return out, feature, hid_feature
