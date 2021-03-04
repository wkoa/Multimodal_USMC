from .baseline import BaselineEarlyFusion, BaselineEarlyFusionv2, \
    BaselineLateFusion, BaselineEarylyFusionLSTM, BaselineEarlyFusionStat, \
    BaselineEarlyFusionSE, BaselineEarlyFusionStackSE, DALEarlyFusion
from .re_weighting import ReWeightingFusion
from .attention import AttentionFusion
from .tensor_fusion import TFN, TFNAttention
from .mtut import Unimodal
from torch import nn


def build(cfg):
    if cfg.FUSION_HEAD.METHOD == "Early Fusion":
        cfg.MODEL.FEATURE_FUSION = "concat"
        net = BaselineEarlyFusion(cfg)
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion Stat":
        cfg.MODEL.FEATURE_FUSION = "concat"
        net = BaselineEarlyFusionStat(cfg)
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion DAL":
        cfg.MODEL.FEATURE_FUSION = "concat"
        cfg.MODEL.TS_NET = "dal"
        net = DALEarlyFusion(cfg)
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion SE":
        net = BaselineEarlyFusionSE(cfg, max_pool=False)
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion StackSE":
        net = BaselineEarlyFusionStackSE(cfg, max_pool=False, n_layer=6)

    elif cfg.FUSION_HEAD.METHOD == "Early Fusion CorrSE":
        net = BaselineEarlyFusionSE(cfg, max_pool=False, corr_matrix=True)
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion MaxSE":
        net = BaselineEarlyFusionSE(cfg, max_pool=True)
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion CorrMaxSE":
        net = BaselineEarlyFusionSE(cfg, max_pool=True, corr_matrix=True)
    elif cfg.FUSION_HEAD.METHOD == "Re-weighting Fusion":
        cfg.MODEL.FEATURE_FUSION = "concat"
        net = ReWeightingFusion(cfg)
        if cfg.MODEL.DEVICE == "cuda":
            net = net.cuda()
    elif cfg.FUSION_HEAD.METHOD == "Early Fusion v2":
        cfg.MODEL.FEATURE_FUSION = "add"
        net = BaselineEarlyFusionv2(cfg)
    elif cfg.FUSION_HEAD.METHOD == "Attention":
        net = AttentionFusion(cfg, stat_pooling=False)
    elif cfg.FUSION_HEAD.METHOD == "AttentionStat":
        net = AttentionFusion(cfg, stat_pooling=True)
    elif cfg.FUSION_HEAD.METHOD == "Tensor Fusion":
        net = TFN(cfg)
    elif cfg.FUSION_HEAD.METHOD == "Late Fusion":
        net = BaselineLateFusion(cfg)
    elif cfg.FUSION_HEAD.METHOD == "Tensor Fusion Attention":
        net = TFNAttention(cfg)
    elif cfg.FUSION_HEAD.METHOD == "MTUT":
        net = nn.ModuleDict()
        for r in cfg.MODALITY.REQUIRMENTS:
            net[r] = Unimodal(r, cfg)
    elif cfg.FUSION_HEAD.METHOD == "LSTM":
        net = BaselineEarylyFusionLSTM(cfg, False)
    elif cfg.FUSION_HEAD.METHOD == "X-LSTM":
        net = BaselineEarylyFusionLSTM(cfg, True)
    elif cfg.FUSION_HEAD.METHOD == "ResLSTM":
        net = BaselineEarylyFusionLSTM(cfg, res=True)
    else:
        raise ValueError
    return net
