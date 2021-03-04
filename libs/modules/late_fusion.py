from torch import nn
from torch.nn import functional as F
from .early_fusion import EarlyFusion


class LateFusion(nn.Module):
    def __init__(self, cfg):
        super(LateFusion, self).__init__()
        self.cfg = cfg
        cfg.MODEL.FEATURE_FUSION = "add"
        self.classifies = nn.ModuleDict()
        for r in cfg.MODALITY.REQUIRMENTS:
            self.classifies[r] = EarlyFusion(cfg)

    def forward(self, x):
        for k, v in x.items():
            try:
                tmpout = self.classifies[k](v)
                out += tmpout
            except UnboundLocalError:
                out = self.classifies[k](v)

        return out/self.cfg.MODALITY.NUMS