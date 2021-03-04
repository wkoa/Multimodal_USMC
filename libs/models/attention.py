from ..modules.attention_fusion_head import *
from ..modules.early_fusion import EarlyFusion
from ..modules.multi_extractor import MultiExtractor

from torch import nn
import torch


def statistic_pooling(emb):
    mean = torch.mean(emb, dim=1, keepdim=True)
    k = ((emb - mean + 1e-9) ** 2).sum(dim=1)
    std = k.sqrt()
    mean = mean.squeeze()
    # std = torch.std(emb, dim=2)
    seg_emb = torch.cat([mean, std], dim=1)
    return seg_emb


class AttentionFusion(nn.Module):
    def __init__(self, cfg, stat_pooling=False):
        super(AttentionFusion, self).__init__()

        self.multi_extractor = MultiExtractor(cfg)
        self.transformer = AttentionFusionHead(cfg)
        if cfg.FUSION_HEAD.ATTENTIONFUSION.ADD:
            cfg.MODEL.FEATURE_FUSION = "add"
        else:
            cfg.MODEL.FEATURE_FUSION = "concat"
        self.early_fusion = EarlyFusion(cfg)

        if stat_pooling:
            self.max_pooling = nn.AdaptiveMaxPool1d(1)
            self.early_fusion.fc1 = nn.Linear(cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.transformer = self.transformer.cuda()
            self.early_fusion = self.early_fusion.cuda()

        self.cfg = cfg
        self.stat_pooling = stat_pooling

    def forward(self, x):
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "stack"
        out = self.multi_extractor(x)

        # out's shape: batch_size, modality_num, feature_dims
        out = self.transformer(out)

        if self.stat_pooling:
            out = self.max_pooling(out.transpose(1, 2))
            out = self.early_fusion(out.squeeze())
            return out

        if self.cfg.FUSION_HEAD.ATTENTIONFUSION.ADD:
            out = torch.sum(out, dim=1)
        else:
            out = out.flatten(1, -1)
        out = self.early_fusion(out)

        return out
