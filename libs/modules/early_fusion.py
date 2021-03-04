from torch import nn
from torch.nn import functional as F
import torch


def statistic_pooling(emb):
    mean = torch.mean(emb, dim=1, keepdim=True)
    k = ((emb - mean + 1e-9) ** 2).sum(dim=1)
    std = k.sqrt()
    mean = mean.squeeze()
    # std = torch.std(emb, dim=2)
    seg_emb = torch.cat([mean, std], dim=1)
    return seg_emb


class EarlyFusionDAL(nn.Module):
    def __init__(self, cfg):
        super(EarlyFusionDAL, self).__init__()
        self.fc1 = nn.Linear(cfg.MODALITY.NUMS * 40, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, cfg.MODEL.NUM_CLASSES)
        self.activation = nn.Tanh()
        self.last_activation = nn.Softmax()

    def forward(self, x, hid_feature=False):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        h_feature = x = self.activation(x)
        y = self.fc3(x)
        y = self.last_activation(y)

        if hid_feature:
            return y, h_feature

        return y


class EarlyFusion(nn.Module):
    """

    """
    def __init__(self, cfg):
        super(EarlyFusion, self).__init__()
        if cfg.MODEL.FEATURE_FUSION == "concat":
            self.fc1 = nn.Linear(cfg.MODALITY.NUMS*cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
        elif cfg.MODEL.FEATURE_FUSION == "add":
            self.fc1 = nn.Linear(cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
        else:
            raise ValueError
        self.fc2 = nn.Linear(cfg.FUSION_HEAD.HIDDEN_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
        self.fc3 = nn.Linear(cfg.FUSION_HEAD.HIDDEN_DIMS, cfg.MODEL.NUM_CLASSES)

        self.dropout1 = nn.Dropout(cfg.FUSION_HEAD.DROPOUT)
        self.dropout2 = nn.Dropout(cfg.FUSION_HEAD.DROPOUT)

        activation = cfg.FUSION_HEAD.ACTIVATION

        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ReLU6':
            self.activation = nn.ReLU6(inplace=True)
        elif activation == 'SELU':
            self.activation = nn.SELU(inplace=True)
        elif activation == "ELU":
            self.activation = nn.ELU(alpha=1.3, inplace=True)
        elif activation == "CELU":
            self.activation = nn.CELU(alpha=1.3, inplace=True)
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "GLU":
            self.activation = nn.GLU()
        elif activation == "None":
            self.activation = nn.Identity()
        else:
            raise ValueError("Activation function is not implemented")

        self.cfg = cfg

        if self.cfg.FUSION_HEAD.COSINE:
            # self.weight_base = nn.Parameter(self.fc3.parameters())
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(cfg.FUSION_HEAD.SCALECLS), requires_grad=True)

    def forward(self, x, hid_feature=False):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation(x)
        h_feature = x = self.dropout2(x)

        if self.cfg.FUSION_HEAD.COSINE:
            weight_ = self.fc3.weight
            weight_normal = F.normalize(weight_, p=2, dim=weight_.dim()-1, eps=1e-12)
            x = F.normalize(x, p=2, dim=x.dim()-1, eps=1e-12)
            self.fc3.weight = nn.Parameter(weight_normal)

        y = self.fc3(x)

        if self.cfg.FUSION_HEAD.COSINE:
            y = self.scale_cls * y

        if hid_feature:
            return y, h_feature

        return y


class EarlyFusionLSTM(nn.Module):
    def __init__(self, cfg, statistic_pooling=False):
        super(EarlyFusionLSTM, self).__init__()
        self.statistic_pooling = statistic_pooling

        self.lstm = nn.LSTM(
            input_size=cfg.FUSION_HEAD.FEATURE_DIMS*cfg.MODALITY.NUMS,
            hidden_size=cfg.MODEL.LSTM.DIM_HIDDEN,
            num_layers=cfg.MODEL.LSTM.NUM_LAYERS,
            batch_first=True
        )
        if self.statistic_pooling:
            self.fc = nn.Linear(2 * cfg.MODEL.LSTM.DIM_HIDDEN, cfg.MODEL.NUM_CLASSES)
        else:
            self.fc = nn.Linear(cfg.MODEL.LSTM.DIM_HIDDEN, cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        if self.statistic_pooling:
            feature = statistic_pooling(r_out)
        else:
            feature = r_out[:, -1, :]

        out = self.fc(feature)

        return out


class EarlyFusionResLSTM(nn.Module):
    def __init__(self, cfg):
        super(EarlyFusionResLSTM, self).__init__()
        self.statistic_pooling = statistic_pooling

        self.lstm = nn.LSTM(
            input_size=cfg.FUSION_HEAD.FEATURE_DIMS*cfg.MODALITY.NUMS,
            hidden_size=cfg.MODEL.LSTM.DIM_HIDDEN,
            num_layers=cfg.MODEL.LSTM.NUM_LAYERS,
            batch_first=True
        )
        self.fc_1 = nn.Linear(2 * cfg.FUSION_HEAD.FEATURE_DIMS*cfg.MODALITY.NUMS, cfg.MODEL.LSTM.DIM_HIDDEN)
        self.fc = nn.Linear(cfg.MODEL.LSTM.DIM_HIDDEN, cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        feature = F.relu(self.fc_1(statistic_pooling(x)))
        feature += r_out[:, -1, :]

        out = self.fc(feature)

        return out