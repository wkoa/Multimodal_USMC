import torch
from torch import nn

from libs.utils import init_model
from .resnet import ResNet
from .xvector import XVectorNet, SEXVectorNet, AttentionXvector, VAEXVectorNet, MaxXVectorNet, MultiHeadXvector
from .transfomer import TransformerExtrctor
from .lstm import LSTMExtractor
from .conv1d_acnn import ACNNet
from .TiCNN import TiCNN, NoDilatedTiCNN, NoMultiHeadTiCNN, NoMFCCTiCNN, NoTimePoolingTiCNN
from .dal import DALnn


class MultiExtractor(nn.Module):
    def __init__(self, cfg):
        super(MultiExtractor, self).__init__()
        self.extractor = nn.ModuleDict()
        for r in cfg.MODALITY.REQUIRMENTS:
            self.extractor[r] = _ModalityExtractor(r, cfg)
            if cfg.MODEL.EXTRACTOR_FREEZE:
                print("Freezing %s ..." % r)
                for param in self.extractor[r].parameters():
                    param.requires_grad_(requires_grad=False)
        self.cfg = cfg
        self.requirments = cfg.MODALITY.REQUIRMENTS

    def forward(self, x):
        """

        :param x: A dict of tensor.
        :return:
        """
        if self.cfg.MODEL.FEATURE_FUSION == "concat":
            out = None
            for i, r in enumerate(self.requirments):
                if i == 0:
                    out = self.extractor[r](x[r])
                else:
                    tmpout = self.extractor[r](x[r])
                    out = torch.cat([out, tmpout], dim=1)
        elif self.cfg.MODEL.FEATURE_FUSION == "add":
            out = None
            for i, r in enumerate(self.requirments):
               if i == 0:
                   out = self.extractor[r](x[r])
               else:
                   out += self.extractor[r](x[r])
        elif self.cfg.MODEL.FEATURE_FUSION == "stack":
            out = None
            for i, r in enumerate(self.requirments):
               if i == 0:
                   out = self.extractor[r](x[r]).unsqueeze(1)
               else:
                   tmpout = self.extractor[r](x[r]).unsqueeze(1)
                   out = torch.cat([out, tmpout], dim=1)

        elif self.cfg.MODEL.FEATURE_FUSION == "none":
            out = {}
            for r in self.requirments:
                out[r] = self.extractor[r](x[r])
        else:
            raise ValueError

        return out


class _ModalityExtractor(nn.Module):
    def __init__(self, name_modality, cfg):
        super(_ModalityExtractor, self).__init__()
        cuda = True if cfg.MODEL.DEVICE == "cuda" else False
        if "Image" in name_modality:
            self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED, is_freeze=cfg.MODEL.RESNET_FREEZE,)
            # self.extractor.fc = nn.Linear(2048, cfg.FUSION_HEAD.FEATURE_DIMS)
        elif "sound" in name_modality:
            if cfg.MODEL.TS_NET == "xvector":
                self.extractor = XVectorNet(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                            cfg.FUSION_HEAD.FEATURE_DIMS,
                                            cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                            )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "se_xvector":
                self.extractor = SEXVectorNet(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                              cfg.FUSION_HEAD.FEATURE_DIMS,
                                              cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                              )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "vae_xvector":
                self.extractor = VAEXVectorNet(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "ticnn":
                self.extractor = TiCNN(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_dilated_ticnn":
                self.extractor = NoDilatedTiCNN(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_multi_ticnn":
                self.extractor = NoMultiHeadTiCNN(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_tp_ticnn":
                self.extractor = NoTimePoolingTiCNN(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                                cfg
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "no_mfcc_ticnn":
                self.extractor = NoMFCCTiCNN(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "max_xvector":
                self.extractor = MaxXVectorNet(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "attn_xvector":
                self.extractor = AttentionXvector(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                                  cfg.FUSION_HEAD.FEATURE_DIMS,
                                                  cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                                  cfg, 'sound')
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "multi_xvector":
                self.extractor = MultiHeadXvector(cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                                  cfg.FUSION_HEAD.FEATURE_DIMS,
                                                  cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                                  cfg, cfg.MODEL.ATTNXVECTOR.NUM_HEAD)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "attention":
                self.extractor = TransformerExtrctor(cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                                     cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                                     cfg.FUSION_HEAD.FEATURE_DIMS,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN,
                                                     cfg.MODEL.ATTNXVECTOR.NUM_LAYER,
                                                     cfg.MODEL.ATTNXVECTOR.NUM_HEAD,
                                                     200,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_K,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_V,
                                                     cfg.FUSION_HEAD.DROPOUT
                                                     )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "LSTM":
                self.extractor = LSTMExtractor('sound',
                                               cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               cfg)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "RawLSTM":
                self.extractor = LSTMExtractor('sound',
                                               cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                                               cfg.MODALITY.SOUND.XVECTOR.ACTIVATION,
                                               cfg, True)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "acnn":
                self.extractor = ACNNet('sound', cfg.FUSION_HEAD.FEATURE_DIMS)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "resnet":
                self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED,
                                        is_freeze=cfg.MODEL.RESNET_FREEZE, )
            elif cfg.MODEL.TS_NET == "resnet_2":
                self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED,
                                        is_freeze=cfg.MODEL.RESNET_FREEZE, )
                self.extractor.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                                 bias=False)
            elif cfg.MODEL.TS_NET == "statacnn":
                self.extractor = ACNNet('sound', cfg.FUSION_HEAD.FEATURE_DIMS, stat_pooling=True)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "dal":
                self.extractor = DALnn('sound')
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            else:
                raise NotImplementedError

        elif "accel" in name_modality:
            if cfg.MODEL.TS_NET == "xvector":
                self.extractor = XVectorNet(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                            cfg.FUSION_HEAD.FEATURE_DIMS,
                                            cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                            )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "se_xvector":
                self.extractor = SEXVectorNet(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                              cfg.FUSION_HEAD.FEATURE_DIMS,
                                              cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                              )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "vae_xvector":
                self.extractor = VAEXVectorNet(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "ticnn":
                self.extractor = TiCNN(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                       cfg.FUSION_HEAD.FEATURE_DIMS,
                                        cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                        )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_tp_ticnn":
                self.extractor = NoTimePoolingTiCNN(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                       cfg.FUSION_HEAD.FEATURE_DIMS,
                                        cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                        cfg)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_mfcc_ticnn":
                self.extractor = NoMFCCTiCNN(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                       cfg.FUSION_HEAD.FEATURE_DIMS,
                                        cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                        )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_dilated_ticnn":
                self.extractor = NoDilatedTiCNN(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_multi_ticnn":
                self.extractor = NoMultiHeadTiCNN(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "max_xvector":
                self.extractor = MaxXVectorNet(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "attn_xvector":
                self.extractor = AttentionXvector(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                                  cfg.FUSION_HEAD.FEATURE_DIMS,
                                                  cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                                  cfg, 'accel')
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "multi_xvector":
                self.extractor = MultiHeadXvector(cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                                  cfg.FUSION_HEAD.FEATURE_DIMS,
                                                  cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                                  cfg, cfg.MODEL.ATTNXVECTOR.NUM_HEAD)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "attention":
                self.extractor = TransformerExtrctor(cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                                     cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                                     cfg.FUSION_HEAD.FEATURE_DIMS,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN,
                                                     cfg.MODEL.ATTNXVECTOR.NUM_LAYER,
                                                     cfg.MODEL.ATTNXVECTOR.NUM_HEAD,
                                                     200,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_K,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_V,
                                                     cfg.FUSION_HEAD.DROPOUT
                                                     )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)

            elif cfg.MODEL.TS_NET == "LSTM":
                self.extractor = LSTMExtractor('accel',
                                               cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                               cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                               cfg)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "RawLSTM":
                self.extractor = LSTMExtractor('accel',
                                               cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM,
                                               cfg.MODALITY.ACCEL.XVECTOR.ACTIVATION,
                                               cfg, raw_data=True)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "acnn":
                self.extractor = ACNNet('accel', cfg.FUSION_HEAD.FEATURE_DIMS)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "resnet":
                self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED,
                                        is_freeze=cfg.MODEL.RESNET_FREEZE,)
            elif cfg.MODEL.TS_NET == "resnet_2":
                self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED,
                                        is_freeze=cfg.MODEL.RESNET_FREEZE, )
                self.extractor.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                                 bias=False)
            elif cfg.MODEL.TS_NET == "statacnn":
                self.extractor = ACNNet('accel', cfg.FUSION_HEAD.FEATURE_DIMS, stat_pooling=True)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "dal":
                self.extractor = DALnn('accel')
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            else:
                raise NotImplementedError

        elif "Force" in name_modality:
            if cfg.MODEL.TS_NET == "xvector":
                self.extractor = XVectorNet(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                            cfg.FUSION_HEAD.FEATURE_DIMS,
                                            cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                            )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "se_xvector":
                self.extractor = SEXVectorNet(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                              cfg.FUSION_HEAD.FEATURE_DIMS,
                                              cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                              )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "vae_xvector":
                self.extractor = VAEXVectorNet(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "ticnn":
                self.extractor = TiCNN(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_tp_ticnn":
                self.extractor = NoTimePoolingTiCNN(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               cfg)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_mfcc_ticnn":
                self.extractor = NoMFCCTiCNN(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_dilated_ticnn":
                self.extractor = NoDilatedTiCNN(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "no_multi_ticnn":
                self.extractor = NoMultiHeadTiCNN(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "max_xvector":
                self.extractor = MaxXVectorNet(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.FUSION_HEAD.FEATURE_DIMS,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "attn_xvector":
                self.extractor = AttentionXvector(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                                  cfg.FUSION_HEAD.FEATURE_DIMS,
                                                  cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                                  cfg, 'force')
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "multi_xvector":
                self.extractor = MultiHeadXvector(cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                                  cfg.FUSION_HEAD.FEATURE_DIMS,
                                                  cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                                  cfg, cfg.MODEL.ATTNXVECTOR.NUM_HEAD)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "attention":
                self.extractor = TransformerExtrctor(cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                                     cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                                     cfg.FUSION_HEAD.FEATURE_DIMS,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN,
                                                     cfg.MODEL.ATTNXVECTOR.NUM_LAYER,
                                                     cfg.MODEL.ATTNXVECTOR.NUM_HEAD,
                                                     500,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_K,
                                                     cfg.MODEL.ATTNXVECTOR.DIM_V,
                                                     cfg.FUSION_HEAD.DROPOUT
                                                     )
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "LSTM":
                self.extractor = LSTMExtractor('force',
                                               cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               cfg)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "RawLSTM":
                self.extractor = LSTMExtractor('force',
                                               cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                                               cfg.MODALITY.FORCE.XVECTOR.ACTIVATION,
                                               cfg, True)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "acnn":
                self.extractor = ACNNet('force', cfg.FUSION_HEAD.FEATURE_DIMS)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "statacnn":
                self.extractor = ACNNet('force', cfg.FUSION_HEAD.FEATURE_DIMS, stat_pooling=True)
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            elif cfg.MODEL.TS_NET == "resnet":
                self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED,
                                        is_freeze=cfg.MODEL.RESNET_FREEZE, )
            elif cfg.MODEL.TS_NET == "resnet_2":
                self.extractor = ResNet(cfg, pretrained=cfg.MODEL.RESNET_PRETRAINED,
                                        is_freeze=cfg.MODEL.RESNET_FREEZE, )
                self.extractor.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                                       bias=False)
            elif cfg.MODEL.TS_NET == "dal":
                self.extractor = DALnn('force')
                self.extractor = init_model(self.extractor, cfg.MODEL.INIT_METHOD)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        if cuda:
            self.extractor = self.extractor.cuda()

    def forward(self, x):
        x = self.extractor(x)
        return x


def _time_series_extrctor(modality_spec, ts_net):
    pass
