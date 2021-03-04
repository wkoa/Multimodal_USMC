import torch
from torch import nn


class TiCNN(nn.Module):
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
        super(TiCNN, self).__init__()
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


class NoDilatedTiCNN(nn.Module):
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
        super(NoDilatedTiCNN, self).__init__()
        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=1)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=1)
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


class NoMultiHeadTiCNN(nn.Module):
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
        super(NoMultiHeadTiCNN, self).__init__()

        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        self.fc11 = nn.Linear(3 * 512, 512)
        # self.fc12 = nn.Linear(3 * 512, 512)
        # self.fc13 = nn.Linear(3 * 512, 512)
        self.fc2 = nn.Linear(512, feature_dim)

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
        # var = self.activation(self.fc12(emb6))
        # var1 = self.activation(self.fc13(emb6))
        seg_emb = self.fc2(mu)
        # seg_emb = self.fc2(mu)

        emb = self.norm6(self.activation(seg_emb))
        # emb = self.activation(seg_emb)
        return emb


class NoTimePoolingTiCNN(nn.Module):
    """

    """
    def __init__(self, input_dim, feature_dim, activation, cfg):
        """

        :param contexts: A list of context.
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param feature_dim: The final desired dims.
        """
        super(NoTimePoolingTiCNN, self).__init__()

        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, cfg.FUSION_HEAD.FEATURE_DIMS, 1)

        # self.fc11 = nn.Linear(3 * 512, 512)
        # self.fc12 = nn.Linear(3 * 512, 512)
        # self.fc13 = nn.Linear(3 * 512, 512)
        # self.fc2 = nn.Linear(3 * 512, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)
        # self.norm6 = nn.BatchNorm1d(feature_dim)
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

        self.lstm = nn.LSTM(
            input_size=cfg.FUSION_HEAD.FEATURE_DIMS,
            hidden_size=cfg.FUSION_HEAD.FEATURE_DIMS,
            num_layers=cfg.MODEL.LSTM.NUM_LAYERS,
            batch_first=True
        )

        # self.avgpool = nn.AdaptiveAvgPool1d(1)

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

        # emb6 = self.avgpool(emb5)
        # batch_size = emb6.shape[0]
        # emb6 = emb6.squeeze()
        # if batch_size == 1:
        #     emb6 = emb6.unsqueeze(0)
        # mu = self.activation(self.fc11(emb6))
        # var = self.activation(self.fc12(emb6))
        # var1 = self.activation(self.fc13(emb6))
        # seg_emb = self.fc2(torch.cat([mu, var, var1], dim=1))
        # seg_emb = self.fc2(mu)

        # emb = self.norm6(self.activation(seg_emb))
        # emb = self.activation(seg_emb)

        emb = emb5.transpose(1, 2)

        r_out, (h_n, h_c) = self.lstm(emb, None)

        return r_out[:, -1, :]


class NoMFCCTiCNN(nn.Module):
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
        super(NoMFCCTiCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 24, 225, stride=3, dilation=1)
        self.pool1 = nn.MaxPool1d(5)

        self.stage1 = nn.Conv1d(24, 512, 5, stride=3, dilation=1)
        self.pool2 = nn.MaxPool1d(3)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, 3 * 512, 1)

        self.fc11 = nn.Linear(3 * 512, 512)
        self.fc12 = nn.Linear(3 * 512, 512)
        self.fc13 = nn.Linear(3 * 512, 512)
        self.fc2 = nn.Linear(3 * 512, feature_dim)

        self.norm0 = nn.BatchNorm1d(24)
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
        x = self.norm0(self.activation(self.conv1(x)))
        x = self.pool1(x)
        self.layer_out1 = emb1 = self.norm1(self.activation(self.stage1(x)))
        emb1 = self.pool2(emb1)

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