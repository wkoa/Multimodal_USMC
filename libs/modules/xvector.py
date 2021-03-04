import torch
from torch import nn
from .se_layer import SELayer
from .attention_fusion_head import AttentionFusionHead
from .transfomer import Transformer


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    # eps = torch.randn_like(std)
    return mu + std


def statistic_pooling(emb, return_detail=False):
    mean = torch.mean(emb, dim=2, keepdim=True)
    k = ((emb - mean + 1e-9) ** 2).sum(dim=2)
    std = k.sqrt()
    mean = mean.squeeze()
    # std = torch.std(emb, dim=2)

    # cov = std/(mean + 1e-12)

    seg_emb = torch.cat([mean, std], dim=1)
    if return_detail:
        return seg_emb, mean, std
    else:
        return seg_emb


class SEXVectorNet(nn.Module):
    """

    """
    def __init__(self, input_dim, feature_dim, activation):
        """

        :param contexts: A list of context.
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param feature_dim: The final desired dims.
        """
        super(SEXVectorNet, self).__init__()
        # Adopted from sidekit nnet/xvector.py
        # in https://git-lium.univ-lemans.fr/Larcher/sidekit/blob/master/nnet/xvector.py
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.se_layer1 = SELayer(512, reduction=16, conv_style="Conv1d")
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.se_layer2 = SELayer(512, reduction=16, conv_style="Conv1d")
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.se_layer3 = SELayer(512, reduction=16, conv_style="Conv1d")
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.se_layer4 = SELayer(512, reduction=16, conv_style="Conv1d")
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)
        self.se_layer5 = SELayer(512 * 3, reduction=16, conv_style="Conv1d")

        self.fc1 = nn.Linear(3 * 512 * 2, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(3*512)
        self.norm6 = nn.BatchNorm1d(feature_dim)
        # self.norm7 = nn.BatchNorm1d(512)

        if activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(0.2)
        elif activation == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif activation == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = torch.nn.ReLU6()
        elif activation == 'SELU':
            self.activation = torch.nn.SELU()
        else:
            raise ValueError("Activation function is not implemented")

        self.layer_out1, self.layer_out2, self.layer_out3, self.layer_out4, self.layer_out5 = \
            None, None, None, None, None

    def forward(self, x):
        self.layer_out1 = emb1 = self.norm1(self.activation(self.se_layer1(self.stage1(x))))
        self.layer_out2 = emb2 = self.norm2(self.activation(self.se_layer2(self.stage2(emb1))))
        self.layer_out3 = emb3 = self.norm3(self.activation(self.se_layer3(self.stage3(emb2))))
        self.layer_out4 = emb4 = self.norm4(self.activation(self.se_layer4(self.stage4(emb3))))
        self.layer_out5 = emb5 = self.norm5(self.activation(self.se_layer5(self.stage5(emb4))))

        # Statistic Pooling
        seg_emb = statistic_pooling(emb5)

        emb = self.norm6(self.activation(self.fc1(seg_emb)))

        return emb


class XVectorNet(nn.Module):
    """

    """
    def __init__(self, input_dim, feature_dim, activation):
        """

        :param contexts: A list of context.
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param feature_dim: The final desired dims.
        """
        super(XVectorNet, self).__init__()
        # Adopted from sidekit nnet/xvector.py
        # in https://git-lium.univ-lemans.fr/Larcher/sidekit/blob/master/nnet/xvector.py
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=1)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=1)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        # self.fc_mean = nn.Linear(3 * 512, 3 * 512)
        # self.fc_var = nn.Linear(3 * 512, 3 * 512)

        self.fc = nn.Linear(3 * 512, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(3*512)
        self.norm6 = nn.BatchNorm1d(feature_dim)
        # self.norm7 = nn.BatchNorm1d(512)

        if activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(0.2)
        elif activation == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif activation == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = torch.nn.ReLU6()
        elif activation == 'SELU':
            self.activation = torch.nn.SELU()
        else:
            raise ValueError("Activation function is not implemented")

        self.layer_out1, self.layer_out2, self.layer_out3, self.layer_out4, self.layer_out5 = \
            None, None, None, None, None

    def forward(self, x):
        self.layer_out1 = emb1 = self.norm1(self.activation(self.stage1(x)))
        self.layer_out2 = emb2 = self.norm2(self.activation(self.stage2(emb1)))
        self.layer_out3 = emb3 = self.norm3(self.activation(self.stage3(emb2)))
        self.layer_out4 = emb4 = self.norm4(self.activation(self.stage4(emb3)))
        self.layer_out5 = emb5 = self.norm5(self.activation(self.stage5(emb4)))

        # Statistic Pooling
        seg_emb = statistic_pooling(emb5, return_detail=False)

        # mean = self.activation(self.fc_mean(mu))
        # var = self.activation(self.fc_var(var))

        emb = self.norm6(self.activation(self.fc(seg_emb)))

        return emb


class MultiHeadXvector(nn.Module):
    def __init__(self, input_dim, feature_dim, activation, cfg, head_num):
        super(MultiHeadXvector, self).__init__()
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=1)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=1)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 512, 1)

        self.fc = nn.Linear(3 * 512, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(512)
        self.norm6 = nn.BatchNorm1d(feature_dim)
        # self.norm7 = nn.BatchNorm1d(512)

        if activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(0.2)
        elif activation == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif activation == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = torch.nn.ReLU6()
        elif activation == 'SELU':
            self.activation = torch.nn.SELU()
        else:
            raise ValueError("Activation function is not implemented")

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        emb1 = self.norm1(self.activation(self.stage1(x)))
        emb2 = self.norm2(self.activation(self.stage2(emb1)))
        emb3 = self.norm3(self.activation(self.stage3(emb2)))
        emb4 = self.norm4(self.activation(self.stage4(emb3)))
        emb5 = self.norm5(self.activation(self.stage5(emb4)))

        # Statistic Pooling
        # seg_emb = statistic_pooling(emb5, return_detail=False)
        mu = self.avg_pool(emb5)
        k = ((emb5 - mu + 1e-9) ** 2).sum(dim=2)
        std = k.sqrt()
        max = self.max_pool(emb5).squeeze()
        mu = mu.squeeze()
        seg_emb = torch.cat([mu, std, max], dim=1)

        # mean = self.activation(self.fc_mean(mu))
        # var = self.activation(self.fc_var(var))

        emb = self.norm6(self.activation(self.fc(seg_emb)))

        return emb


class AttentionXvector(nn.Module):
    def __init__(self, input_dim, feature_dim, activation, cfg, modality_type):
        # feature_dim = int(feature_dim/2)

        super(AttentionXvector, self).__init__()
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        self.fc1 = nn.Linear(3 * 512 * 3, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(3 * 512)
        self.norm6 = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)
        # self.norm7 = nn.BatchNorm1d(512)

        if activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(0.2)
        elif activation == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif activation == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = torch.nn.ReLU6()
        elif activation == 'SELU':
            self.activation = torch.nn.SELU()
        else:
            raise ValueError("Activation function is not implemented")

        # self.layer_out1, self.layer_out2, self.layer_out3, self.layer_out4, self.layer_out5 = \
        #     None, None, None, None, None
        if modality_type == "sound":
            n_position = 85
        elif modality_type == "force":
            n_position = 485
        else:
            n_position = 485

        self.transformer = Transformer(3 * 512,
                                       cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN,
                                       cfg.MODEL.ATTNXVECTOR.NUM_HEAD,
                                       cfg.MODEL.ATTNXVECTOR.NUM_LAYER,
                                       n_position,
                                       cfg.MODEL.ATTNXVECTOR.DIM_K,
                                       cfg.MODEL.ATTNXVECTOR.DIM_V,
                                       cfg.FUSION_HEAD.DROPOUT)

        # self.fc2 = nn.Linear(cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.FEATURE_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.transformer = self.transformer.cuda()

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(
            3*512,
            3*512,
            batch_first=True
        )

    def forward(self, x):
        emb1 = self.norm1(self.activation(self.stage1(x)))
        emb2 = self.norm2(self.activation(self.stage2(emb1)))
        emb3 = self.norm3(self.activation(self.stage3(emb2)))
        emb4 = self.norm4(self.activation(self.stage4(emb3)))
        emb5 = self.norm5(self.activation(self.stage5(emb4)))  # emb5 has shape batch_size, feature_dims, len

        emb6_1 = self.transformer(emb5.transpose(1, 2))  # emb6_1 has shape batch_size, lens, feature_dims
        emb6_1, (h_n, h_c) = self.lstm(emb6_1, None)
        emb6_1 = emb6_1[:, -1, :]
        emb6_2 = statistic_pooling(emb5)

        # seg_emb_2 = statistic_pooling(emb5)
        seg_emb = torch.cat([emb6_1, emb6_2], dim=1)
        emb = self.norm6(self.activation(self.fc1(seg_emb)))

        return emb


class VAEXVectorNet(nn.Module):
    """

    """
    def __init__(self, input_dim, feature_dim, activation):
        """

        :param contexts: A list of context.
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param feature_dim: The final desired dims.
        """
        super(VAEXVectorNet, self).__init__()
        # Adopted from sidekit nnet/xvector.py
        # in https://git-lium.univ-lemans.fr/Larcher/sidekit/blob/master/nnet/xvector.py
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        self.fc11 = nn.Linear(3 * 512, 512)
        self.fc12 = nn.Linear(3 * 512, 512)
        self.fc13 = nn.Linear(3 * 512, 512)
        self.fc2 = nn.Linear(3 * 512, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(3*512)
        self.norm6 = nn.BatchNorm1d(feature_dim)
        self.norm7 = nn.BatchNorm1d(512)

        if activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(0.2)
        elif activation == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif activation == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = torch.nn.ReLU6()
        elif activation == 'SELU':
            self.activation = torch.nn.SELU()
        else:
            raise ValueError("Activation function is not implemented")

        self.layer_out1, self.layer_out2, self.layer_out3, self.layer_out4, self.layer_out5 = \
            None, None, None, None, None

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        self.layer_out1 = emb1 = self.norm1(self.activation(self.stage1(x)))
        self.layer_out2 = emb2 = self.norm2(self.activation(self.stage2(emb1)))
        self.layer_out3 = emb3 = self.norm3(self.activation(self.stage3(emb2)))
        self.layer_out4 = emb4 = self.norm4(self.activation(self.stage4(emb3)))
        self.layer_out5 = emb5 = self.norm5(self.activation(self.stage5(emb4)))

        # emb1 = self.activation(self.stage1(x))
        # emb2 = self.activation(self.stage2(emb1))
        # emb3 = self.activation(self.stage3(emb2))
        # emb4 = self.activation(self.stage4(emb3))
        # emb5 = self.activation(self.stage5(emb4))

        emb6 = self.avgpool(emb5)
        batch_size = emb6.shape[0]
        emb6 = emb6.squeeze()
        if batch_size == 1:
            emb6 = emb6.unsqueeze(0)
        mu = self.activation(self.fc11(emb6))
        var = self.activation(self.fc12(emb6))
        var1 = self.activation(self.fc13(emb6))
        seg_emb = self.fc2(torch.cat([mu, var, var1], dim=1))
        # seg_emb = self.fc2(mu)

        emb = self.norm6(self.activation(seg_emb))
        # emb = self.activation(seg_emb)
        return emb

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class MaxXVectorNet(nn.Module):
    """

    """
    def __init__(self, input_dim, feature_dim, activation):
        """

        :param contexts: A list of context.
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param feature_dim: The final desired dims.
        """
        super(MaxXVectorNet, self).__init__()
        # Adopted from sidekit nnet/xvector.py
        # in https://git-lium.univ-lemans.fr/Larcher/sidekit/blob/master/nnet/xvector.py
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        self.fc11 = nn.Linear(3 * 512, 3*512)
        self.fc12 = nn.Linear(3*512, 3*512)
        self.fc1 = nn.Linear(3 * 512 * 2, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(3*512)
        self.norm6 = nn.BatchNorm1d(feature_dim)
        # self.norm7 = nn.BatchNorm1d(512)

        if activation == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(0.2)
        elif activation == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif activation == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = torch.nn.ReLU6()
        elif activation == 'SELU':
            self.activation = torch.nn.SELU()
        else:
            raise ValueError("Activation function is not implemented")

        self.layer_out1, self.layer_out2, self.layer_out3, self.layer_out4, self.layer_out5 = \
            None, None, None, None, None

        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        self.layer_out1 = emb1 = self.norm1(self.activation(self.stage1(x)))
        self.layer_out2 = emb2 = self.norm2(self.activation(self.stage2(emb1)))
        self.layer_out3 = emb3 = self.norm3(self.activation(self.stage3(emb2)))
        self.layer_out4 = emb4 = self.norm4(self.activation(self.stage4(emb3)))
        self.layer_out5 = emb5 = self.norm5(self.activation(self.stage5(emb4)))

        # Statistic Pooling
        seg_emb_1 = self.avg_pooling(emb5)
        # seg_emb_2 = self.max_pooling(emb5)

        # seg_emb = torch.cat([seg_emb_1, seg_emb_2], dim=2)
        seg_emb_1 = seg_emb_1.flatten(1, 2)
        seg_emb_1 = self.fc11(seg_emb_1)
        # seg_emb_2 = seg_emb_2.flatten(1, 2)
        # seg_emb_2 = self.fc12(seg_emb_2)

        # seg_emb = torch.cat([seg_emb_1, seg_emb_2], dim=1)

        emb = self.norm6(self.activation(self.fc1(seg_emb_1)))

        return emb