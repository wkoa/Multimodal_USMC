from torch import nn

from libs.modules import *


def statistic_pooling(emb):
    mean = torch.mean(emb, dim=1, keepdim=True)
    k = ((emb - mean + 1e-9) ** 2).sum(dim=1)
    std = k.sqrt()
    mean = mean.squeeze()
    # std = torch.std(emb, dim=2)
    seg_emb = torch.cat([mean, std], dim=1)
    return seg_emb


class BaselineEarlyFusion(nn.Module):
    def __init__(self, cfg):
        """

        :param num_class:
        :param requirements: Should not include class part.
        """
        super(BaselineEarlyFusion, self).__init__()
        cfg.MODEL.FEATURE_FUSION = "concat"
        self.multi_extractor = MultiExtractor(cfg)
        self.earyly_fusion = EarlyFusion(cfg)
        if cfg.MODEL.DEVICE == "cuda":
            self.earyly_fusion = self.earyly_fusion.cuda()

    def forward(self, x, return_feature=False):
        """

        :param x: A list of tensor
        :return:
        """
        # self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "none"
        x = self.multi_extractor(x)
        out, hid_feature = self.earyly_fusion(x, True)

        if return_feature:
            return out, x, hid_feature
        return out


class DALEarlyFusion(nn.Module):
    def __init__(self, cfg):
        """

        :param num_class:
        :param requirements: Should not include class part.
        """
        super(DALEarlyFusion, self).__init__()
        cfg.MODEL.FEATURE_FUSION = "concat"
        self.multi_extractor = MultiExtractor(cfg)
        self.earyly_fusion = EarlyFusionDAL(cfg)
        if cfg.MODEL.DEVICE == "cuda":
            self.earyly_fusion = self.earyly_fusion.cuda()

    def forward(self, x, return_feature=False):
        """

        :param x: A list of tensor
        :return:
        """
        # self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "none"
        x = self.multi_extractor(x)
        out, hid_feature = self.earyly_fusion(x, True)

        if return_feature:
            return out, x, hid_feature
        return out


class BaselineEarlyFusionStat(nn.Module):
    def __init__(self, cfg):
        """

        :param num_class:
        :param requirements: Should not include class part.
        """
        super(BaselineEarlyFusionStat, self).__init__()

        self.multi_extractor = MultiExtractor(cfg)
        self.early_fusion = EarlyFusion(cfg)
        self.early_fusion.fc1 = nn.Linear(2 * cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
        if cfg.MODEL.DEVICE == "cuda":
            self.early_fusion = self.early_fusion.cuda()

    def forward(self, x, return_feature=False):
        """

        :param x: A list of tensor
        :return:
        """
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "stack"
        x = self.multi_extractor(x)
        x = statistic_pooling(x)
        out, hid_feature = self.early_fusion(x, True)

        if return_feature:
            return out, x, hid_feature
        return out


class BaselineEarlyFusionSE(nn.Module):
    def __init__(self, cfg, max_pool=False, corr_matrix=False):
        """

        :param num_class:
        :param requirements: Should not include class part.
        """
        super(BaselineEarlyFusionSE, self).__init__()

        self.multi_extractor = MultiExtractor(cfg)
        conv_style = "Conv1d"
        if corr_matrix:
            conv_style = "Conv2d"

        if max_pool:
            self.se_layer = MaxSELayer(channel=cfg.MODALITY.NUMS,
                                       reduction=1,
                                       conv_style=conv_style
                                       )
        else:
            self.se_layer = SELayer(channel=cfg.MODALITY.NUMS,
                                    reduction=1,
                                    conv_style=conv_style
                                    )

        if cfg.MODEL.FEATURE_FUSION == "add":
            self.conv1d = nn.Conv1d(cfg.MODALITY.NUMS, 1, 1)
        else:
            pass
        cfg.MODEL.FEATURE_FUSION = "concat"
        self.early_fusion = EarlyFusion(cfg)
        # self.early_fusion.fc1 = nn.Linear(2 * cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
        if cfg.MODEL.DEVICE == "cuda":
            self.early_fusion = self.early_fusion.cuda()

        self.cfg = cfg
        self.corr_matrix = corr_matrix

    def forward(self, x, return_feature=False):
        """

        :param x: A list of tensor
        :return:
        """
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "stack"
        x = self.multi_extractor(x)
        if self.corr_matrix:
            x = torch.mm(x.unsqueeze(3), x.unsqueeze(2))
        x = self.se_layer(x)
        if self.cfg.MODEL.FEATURE_FUSION == "add":
            x = self.conv1d(x)
        x = x.flatten(1, -1)
        out, hid_feature = self.early_fusion(x, True)

        if return_feature:
            return out, x, hid_feature
        return out


class BaselineEarlyFusionStackSE(nn.Module):
    def __init__(self, cfg, max_pool=False, n_layer=1):
        """

        :param num_class:
        :param requirements: Should not include class part.
        """
        super(BaselineEarlyFusionStackSE, self).__init__()

        self.multi_extractor = MultiExtractor(cfg)
        conv_style = "Conv1d"

        self.se_layer = nn.ModuleList([SEEncoderLayer(channel=cfg.MODALITY.NUMS,
                                       reduction=1,
                                       conv_style=conv_style,
                                       max_pool=max_pool
                                       ) for _ in range(n_layer)])

        if cfg.MODEL.FEATURE_FUSION == "add":
            self.conv1d = nn.Conv1d(cfg.MODALITY.NUMS, 1, 1)
        else:
            pass
        cfg.MODEL.FEATURE_FUSION = "concat"
        self.early_fusion = EarlyFusion(cfg)
        # self.early_fusion.fc1 = nn.Linear(2 * cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
        if cfg.MODEL.DEVICE == "cuda":
            self.early_fusion = self.early_fusion.cuda()

        self.cfg = cfg

    def forward(self, x, return_feature=False):
        """

        :param x: A list of tensor
        :return:
        """
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "stack"
        enc_output = self.multi_extractor(x)
        for enc_layer in self.se_layer:
            enc_output = enc_layer(enc_output)
        if self.cfg.MODEL.FEATURE_FUSION == "add":
            x = self.conv1d(x)
        enc_output = enc_output.flatten(1, -1)
        out, hid_feature = self.early_fusion(enc_output, True)

        if return_feature:
            return out, x, hid_feature
        return out


class BaselineEarlyFusionv2(nn.Module):
    def __init__(self, cfg):
        super(BaselineEarlyFusionv2, self).__init__()

        self.multi_extractor = MultiExtractor(cfg)
        cfg.MODEL.FEATURE_FUSION = "add"
        self.earyly_fusion = EarlyFusion(cfg)
        if cfg.MODEL.DEVICE == "cuda":
            self.earyly_fusion = self.earyly_fusion.cuda()

    def forward(self, x):
        x = self.multi_extractor(x)
        out = self.earyly_fusion(x)

        return out


class BaselineLateFusion(nn.Module):
    def __init__(self, cfg):
        super(BaselineLateFusion, self).__init__()
        self.multi_extractor = MultiExtractor(cfg)
        self.late_fusion = LateFusion(cfg)
        if cfg.MODEL.DEVICE == "cuda":
            self.late_fusion = self.late_fusion.cuda()

    def forward(self, x):
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "none"
        x = self.multi_extractor(x)
        out = self.late_fusion(x)

        return out
    

class BaselineEarylyFusionLSTM(nn.Module):
    def __init__(self, cfg, statistic_pooling=False, res=False):
        super(BaselineEarylyFusionLSTM, self).__init__()
        self.multi_extractor = MultiExtractor(cfg)
        if res:
            self.lstm_classify = EarlyFusionResLSTM(cfg)
        else:
            self.lstm_classify = EarlyFusionLSTM(cfg, statistic_pooling)

        if cfg.MODEL.DEVICE == "cuda":
            self.lstm_classify = self.lstm_classify.cuda()
            self.multi_extractor = self.multi_extractor.cuda()
        self.cfg = cfg
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(85)

    def forward(self, x):
        self.multi_extractor.cfg.MODEL.FEATURE_FUSION = "none"
        x = self.multi_extractor(x)
        feature = None
        for i, r in enumerate(self.cfg.MODALITY.REQUIRMENTS):
            if "Force" in r:
                # For accel and sound len is 85, for *Force is 485
                x[r] = self.adaptive_avg_pool(x[r].transpose(1, 2))
                x[r] = x[r].transpose(1, 2)
            elif "Image" in r:
                x[r] = x[r].expand(self.cfg.TRAIN.BATCH_SIZE, 85, self.cfg.FUSION_HEAD.FEATURE_DIMS)
            if i == 0:
                feature = x[r]
            else:
                tmpout = x[r]
                feature = torch.cat([feature, tmpout], dim=2)

        out = self.lstm_classify(feature)

        return out