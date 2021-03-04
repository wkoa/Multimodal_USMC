from torch import nn

from ..modules.tensor_fusion_head import *
from ..modules.multi_extractor import MultiExtractor
from ..modules.early_fusion import *
from ..modules.attention_fusion_head import *


class TFN(nn.Module):
    def __init__(self, cfg):
        super(TFN, self).__init__()
        cfg.FUSION_HEAD.FEATURE_DIMS = cfg.FUSION_HEAD.TENSORFUSION.DIMS
        self.multi_extractor = MultiExtractor(cfg)
        self.tensor_fusion = TensorFusionHead()
        cfg.MODEL.FEATURE_FUSION = "concat"
        self.early_fusion = EarlyFusion(cfg)
        self.early_fusion.fc1 = nn.Linear((cfg.FUSION_HEAD.FEATURE_DIMS+1)**cfg.MODALITY.NUMS, cfg.FUSION_HEAD.HIDDEN_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.early_fusion = self.early_fusion.cuda()

    def forward(self, x):
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "stack"
        x = self.multi_extractor(x)
        x = self.tensor_fusion(x)
        out = self.early_fusion(x)

        out = F.softmax(out, dim=1)
        return out


class TFNAttention(nn.Module):
    def __init__(self, cfg):
        super(TFNAttention, self).__init__()
        self.multi_extractor = MultiExtractor(cfg)
        self.transformer = AttentionFusionHead(cfg)

        self.fc = nn.Linear(cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.TENSORFUSION.DIMS, bias=False)

        self.tensor_fusion = TensorFusionHead()
        cfg.MODEL.FEATURE_FUSION = "concat"
        self.early_fusion = EarlyFusion(cfg)
        self.early_fusion.fc1 = nn.Linear((cfg.FUSION_HEAD.TENSORFUSION.DIMS + 1) ** cfg.MODALITY.NUMS,
                                          cfg.FUSION_HEAD.HIDDEN_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.early_fusion = self.early_fusion.cuda()
            self.transformer = self.transformer.cuda()

    def forward(self, x):
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "stack"
        x = self.multi_extractor(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = self.tensor_fusion(x)
        out = self.early_fusion(x)

        out = F.softmax(out, dim=1)
        return out