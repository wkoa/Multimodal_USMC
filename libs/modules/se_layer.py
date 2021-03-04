import torch
from torch import nn


class MaxSELayer(nn.Module):
    def __init__(self, channel, reduction=16, conv_style='Conv2d'):
        super(MaxSELayer, self).__init__()
        self.conv_style = conv_style

        if conv_style == 'Conv2d':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(2 * channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.conv_style == 'Conv2d':
            b, c, _, _ = x.size()
            y_avg = self.avg_pool(x).view(b, c)
            y_max = self.max_pool(x).view(b, c)

            y = torch.cat([y_avg, y_max], dim=1)
            y = self.fc(y).view(b, c, 1, 1)
        else:
            b, c, _ = x.size()
            y_avg = self.avg_pool(x).view(b, c)
            y_max = self.max_pool(x).view(b, c)
            y = torch.cat([y_avg, y_max], dim=1)
            y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, conv_style='Conv2d'):
        super(SELayer, self).__init__()
        self.conv_style = conv_style

        if conv_style == 'Conv2d':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.conv_style == 'Conv2d':
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
        else:
            b, c, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)


class SEEncoderLayer(nn.Module):
    def __init__(self, channel, reduction=16, conv_style='Conv2d', max_pool=False):
        super(SEEncoderLayer, self).__init__()
        if max_pool:
            self.se_layer = MaxSELayer(channel, reduction, conv_style)
        else:
            self.se_layer = SELayer(channel, reduction, conv_style)

    def forward(self, x):
        residual = x
        return residual + self.se_layer(x)


class MultimodalSELayer(nn.Module):
    def __init__(self):
        super(MultimodalSELayer, self).__init__()

    def forward(self, modalities):
        pass