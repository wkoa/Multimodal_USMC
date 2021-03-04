import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torch.hub import load_state_dict_from_url
from PIL import Image
from libs.utils import init_model
from libs.modules.se_layer import SELayer


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, cfg, pretrained=True, is_freeze=True):
        name = cfg.MODEL.RESNET_NAME
        assert name in ['resnet18', "se_resnet50", "resnet50", "resnet101", "resnet152"], "Invalid CNN type: {:s}".format(name)
        super(ResNet, self).__init__()
        self.name = name
        if name == "se_resnet50":
            if isinstance(pretrained, bool):
                self.model = se_resnet50(pretrained=pretrained)
                if not pretrained:
                    self.model = init_model(self.model, cfg.MODEL.INIT_METHOD)
            else:
                self.model = se_resnet50(pretrained=False)
                self.model.load_state_dict(torch.load(pretrained))
        else:
            if isinstance(pretrained, bool):
                # Load pretrained model from Internet.
                self.model = models.__dict__[name](pretrained=pretrained)
                if not pretrained:
                    self.model = init_model(self.model, cfg.MODEL.INIT_METHOD)
            else:
                # Load pretrained model in local dirs.
                self.model = models.__dict__[name](pretrained=False)
                self.model.load_state_dict(torch.load(pretrained))
        if name == "resnet18":
            self.fc = nn.Linear(512, cfg.FUSION_HEAD.FEATURE_DIMS)
        else:
            self.fc = nn.Linear(2048, cfg.FUSION_HEAD.FEATURE_DIMS)
        delattr(self.model, "fc")
        # delattr(self.model, "avgpool")

        if is_freeze:
            print("Freezing %s ..." % name)
            for param in self.model.parameters():
                param.requires_grad_(requires_grad=False)

        self.layer_out1, self.layer_out2, self.layer_out3, self.layer_out4 = None, None, None, None

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        self.layer_out1 = x = self.model.layer1(x)
        self.layer_out2 = x = self.model.layer2(x)
        self.layer_out3 = x = self.model.layer3(x)
        self.layer_out4 = x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))

        return x

def se_resnet18(num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model