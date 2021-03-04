from torch import nn
from torch.nn import functional as F

from ..modules.multi_extractor import MultiExtractor
from ..modules.re_weighting_head import ReWeightingFusionHead


class ReWeightingFusion(nn.Module):
    def __init__(self, cfg):
        super(ReWeightingFusion, self).__init__()
        assert cfg.MODALITY.NUMS > 1, "cfg.MODALITY.NUMS must be more than 1."
        self.cfg = cfg
        reduction = cfg.FUSION_HEAD.REWEIGHTINGFUSION.DIM_REDUCTION
        self.multi_extractor = MultiExtractor(cfg)
        self.modality_list = cfg.MODALITY.REQUIRMENTS

        self.bottleneck_conv1 = nn.Conv1d(512, 256, 1)
        self.bottleneck_conv2 = nn.Conv1d(512, 512, 1)
        self.bottleneck_conv3 = nn.Conv1d(512, 1024, 1)
        self.bottleneck_conv4 = nn.Conv1d(512 * 3, 2048, 1)

        self.linear1 = nn.Linear(256, 512, bias=False)
        self.linear2 = nn.Linear(512, 1024, bias=False)
        self.linear3 = nn.Linear(1024, 2048, bias=False)

        output_channel = 2048

        if 'h4' in self.cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY:
            output_channel = 2048
        elif 'h3' in self.cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY:
            output_channel = 1024
        elif 'h2' in self.cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY:
            output_channel = 512
        elif 'h1' in self.cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY:
            output_channel = 256

        self.fc = nn.Sequential(
            nn.Linear(output_channel, output_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel // reduction, output_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, cfg.MODALITY.NUMS, bias=False),
        )

        self.head = ReWeightingFusionHead(cfg)

        if cfg.MODEL.DEVICE == "cuda":
            self.head = self.head.cuda()

    def forward(self, x):
        """

        :param x: A dict of tensor
        :return:
        """
        out = {}
        for i, m in enumerate(self.modality_list):
            out[m] = self.multi_extractor.extractor[m](x[m])
            if "Image" in m:
                h1_, h2_, h3_, h4_ = self.multi_extractor.extractor[m].extractor.layer_out1, \
                                     self.multi_extractor.extractor[m].extractor.layer_out2, \
                                     self.multi_extractor.extractor[m].extractor.layer_out3, \
                                     self.multi_extractor.extractor[m].extractor.layer_out4

                b, c, _, _ = h1_.size()
                h1_ = F.adaptive_avg_pool2d(h1_, 1).view(b, c)
                _, c, _, _ = h2_.size()
                h2_ = F.adaptive_avg_pool2d(h2_, 1).view(b, c)
                _, c, _, _ = h3_.size()
                h3_ = F.adaptive_avg_pool2d(h3_, 1).view(b, c)
                _, c, _, _ = h4_.size()
                h4_ = F.adaptive_avg_pool2d(h4_, 1).view(b, c)

            else:
                _, h1_, h2_, h3_, h4_ = self.multi_extractor.extractor[m].extractor.layer_out1, \
                                        self.multi_extractor.extractor[m].extractor.layer_out2, \
                                        self.multi_extractor.extractor[m].extractor.layer_out3, \
                                        self.multi_extractor.extractor[m].extractor.layer_out4, \
                                        self.multi_extractor.extractor[m].extractor.layer_out5
                h1_ = self.bottleneck_conv1(h1_)
                h2_ = self.bottleneck_conv2(h2_)
                h3_ = self.bottleneck_conv3(h3_)
                h4_ = self.bottleneck_conv4(h4_)

                b, c, _ = h1_.size()
                h1_ = F.adaptive_avg_pool1d(h1_, 1).view(b, c)
                b, c, _ = h2_.size()
                h2_ = F.adaptive_avg_pool1d(h2_, 1).view(b, c)
                b, c, _ = h3_.size()
                h3_ = F.adaptive_avg_pool1d(h3_, 1).view(b, c)
                b, c, _ = h4_.size()
                h4_ = F.adaptive_avg_pool1d(h4_, 1).view(b, c)

            if i == 0:
                h1, h2, h3, h4 = h1_, h2_, h3_, h4_
            else:
                h1 += h1_
                h2 += h2_
                h3 += h3_
                h4 += h4_

        h_dict = {'h1': h1, 'h2': h2, "h3": h3, "h4": h4}

        reweighting = self.reweighting_attention(h_dict, self.cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY)

        reweighting = F.softmax(reweighting, dim=1)
        result = None
        for i, m in enumerate(self.modality_list):
            if i == 0:
                result = reweighting[:, i].view(b, 1).expand_as(out[m]) * out[m]
            else:
                result += reweighting[:, i].view(b, 1).expand_as(out[m]) * out[m]

        y = self.head(result)
        return y

    def reweighting_attention(self, x_dict, level_list):
        level_list.sort()
        fc_dict = {'h1': self.linear1, 'h2': self.linear2, 'h3': self.linear3}
        out = None
        for i, clevel in enumerate(level_list):
            if i == 0:
                out = x_dict[clevel]
            else:
                out = fc_dict[level_list[i - 1]](out) + x_dict[clevel]

        assert out is not None

        return out


