from torch import nn
from libs.modules.transfomer import Transformer


class AttentionFusionHead(nn.Module):
    def __init__(self, cfg):
        super(AttentionFusionHead, self).__init__()
        self.cfg = cfg

        self.transformer = Transformer(cfg.FUSION_HEAD.FEATURE_DIMS,
                                       cfg.FUSION_HEAD.ATTENTIONFUSION.DIM_HIDDEN,
                                       cfg.FUSION_HEAD.ATTENTIONFUSION.NUM_HEAD,
                                       cfg.FUSION_HEAD.ATTENTIONFUSION.NUM_LAYER,
                                       cfg.MODALITY.NUMS,
                                       cfg.FUSION_HEAD.ATTENTIONFUSION.DIM_K,
                                       cfg.FUSION_HEAD.ATTENTIONFUSION.DIM_V,
                                       cfg.FUSION_HEAD.DROPOUT
                                       )

    def forward(self, x, return_attns=False):
        return self.transformer(x, return_attns)
