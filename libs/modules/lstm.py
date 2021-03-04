from torch import nn
import torch
from .conv1d_acnn import ACNNet


def statistic_pooling(emb):
    mean = torch.mean(emb, dim=1, keepdim=True)
    k = ((emb - mean + 1e-9) ** 2).sum(dim=1)
    std = k.sqrt()
    mean = mean.squeeze()
    # std = torch.std(emb, dim=2)
    seg_emb = torch.cat([mean, std], dim=1)
    return seg_emb


class LSTMExtractor(nn.Module):
    def __init__(self, modality_name, input_dim, activation, cfg, raw_data=False):
        super(LSTMExtractor, self).__init__()

        if raw_data:
            self.conv1 = nn.Conv1d(1, 16, 225, stride=3, dilation=1)
            self.pool1 = nn.MaxPool1d(5)

            self.stage1 = nn.Conv1d(16, 32, 5, dilation=1)
            self.pool2 = nn.MaxPool1d(3)
            self.stage2 = nn.Conv1d(32, 32, 3, dilation=2)
            self.stage3 = nn.Conv1d(32, 32, 3, dilation=3)
            self.stage4 = nn.Conv1d(32, 64, 1)
            # self.stage5 = nn.Conv1d(512, 3 * 512, 1)

            # self.norm0 = nn.BatchNorm1d(16)
            # self.norm1 = nn.BatchNorm1d(32)
            # self.norm2 = nn.BatchNorm1d(32)
            # self.norm3 = nn.BatchNorm1d(32)
            # self.norm4 = nn.BatchNorm1d(64)
        else:
            self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)

            self.stage2 = nn.Conv1d(512, 512, 3, dilation=1)
            self.stage3 = nn.Conv1d(512, 512, 3, dilation=1)
            self.stage4 = nn.Conv1d(512, 512, 1)
            self.stage5 = nn.Conv1d(512, cfg.FUSION_HEAD.FEATURE_DIMS, 1)

        # self.fc1 = nn.Linear(3 * 512 * 2, feature_dim)

        # self.norm1 = nn.BatchNorm1d(512)
        # self.norm4 = nn.BatchNorm1d(512)
        # self.norm5 = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)

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

        self.raw_data = raw_data

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=cfg.FUSION_HEAD.FEATURE_DIMS,
            num_layers=cfg.MODEL.LSTM.NUM_LAYERS,
            batch_first=True
        )

    def forward(self, x):
        if self.raw_data:
            # emb0 = self.norm0(self.activation(self.conv1(x)))
            # emb0 = self.activation(self.conv1(x))
            # emb0 = self.pool1(emb0)
            # # emb1 = self.norm1(self.activation(self.stage1(emb0)))
            # emb1 = self.activation(self.stage1(emb0))
            # emb1 = self.pool2(emb1)
            # x = self.norm0(self.activation(self.conv1(x)))
            # x = self.pool1(x)
            #
            # x = self.norm1(self.activation(self.stage1(x)))
            # x = self.pool2(x)
            #
            # x = self.norm2(self.activation(self.stage2(x)))
            #
            # x = self.norm3(self.activation(self.stage3(x)))
            #
            # emb = self.norm4(self.activation(self.stage4(x)))
            x = self.activation(self.conv1(x))
            x = self.pool1(x)

            x = self.activation(self.stage1(x))
            x = self.pool2(x)

            x = self.activation(self.stage2(x))

            x = self.activation(self.stage3(x))

            emb = self.activation(self.stage4(x))
        else:
            emb1 = self.norm1(self.activation(self.stage1(x)))

        # emb4 = self.norm4(self.activation(self.stage4(emb1)))
        # emb = self.norm5(self.activation(self.stage5(emb4)))  # emb5 has shape batch_size, feature_dims, len
        # emb4 = self.activation(self.stage4(emb1))
        # emb = self.activation(self.stage5(emb4))
        emb = emb.transpose(1, 2)

        r_out, (h_n, h_c) = self.lstm(emb, None)

        return r_out[:, -1, :]


class XLSTMExtractor(nn.Module):
    def __init__(self, input_dim, activation, cfg):
        super(XLSTMExtractor, self).__init__()

        self.stage1 = nn.Conv1d(input_dim, 512, 5, dilation=1)
        self.stage2 = nn.Conv1d(512, 512, 3, dilation=2)
        self.stage3 = nn.Conv1d(512, 512, 3, dilation=3)
        self.stage4 = nn.Conv1d(512, 512, 1)
        self.stage5 = nn.Conv1d(512, cfg.FUSION_HEAD.FEATURE_DIMS, 1)

        # self.fc1 = nn.Linear(3 * 512 * 2, feature_dim)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(512)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)

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

        self.lstm = nn.LSTM(
            input_size=cfg.FUSION_HEAD.FEATURE_DIMS,
            hidden_size=cfg.MODEL.LSTM.DIM_HIDDEN,
            num_layers=cfg.MODEL.LSTM.NUM_LAYERS,
            batch_first=True
        )
        self.fc = nn.Linear(2 * cfg.MODEL.LSTM.DIM_HIDDEN, cfg.FUSION_HEAD.HIDDEN_DIMS)

    def forward(self, x):
        emb1 = self.norm1(self.activation(self.stage1(x)))
        emb2 = self.norm2(self.activation(self.stage2(emb1)))
        emb3 = self.norm3(self.activation(self.stage3(emb2)))
        emb4 = self.norm4(self.activation(self.stage4(emb3)))
        emb = self.norm5(self.activation(self.stage5(emb4)))  # emb5 has shape batch_size, feature_dims, len
        emb = emb.transpose(1, 2)

        r_out, (h_n, h_c) = self.lstm(x, None)
        feature = statistic_pooling(r_out)

        return feature
