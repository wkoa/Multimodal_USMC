from torch import nn


class ReWeightingFusionHead(nn.Module):
    """

        """

    def __init__(self, cfg):
        super(ReWeightingFusionHead, self).__init__()
        self.fc1 = nn.Linear(cfg.FUSION_HEAD.FEATURE_DIMS, cfg.FUSION_HEAD.HIDDEN_DIMS)
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
        else:
            raise ValueError("Activation function is not implemented")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        y = self.fc3(x)

        return y
