import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from libs.configs.defaults import _C as cfg
from libs.datasets.CBM_dataset import CBMDataset, CBMDatasetPickle
from libs.engien.train_loop import TrainerMTUT
from libs.models.baseline import BaselineEarlyFusion
from libs.logger import Logger
from libs.utils import *
from libs.models.build import build
from libs.modules.center_loss import CenterLoss


MULTIGPU = False
cfg.MODALITY.TO_PICKLE = False

PICKLE_DIR = '/onekeyai_shared/CBM_PickleData'
cfg.MODALITY.PICKLE_FILE = "CBMDataset_normalForce_macroImage_class_-1.data"
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_sound_normalForce_frictionForce_class_-1.data"
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_macroImage_sound_frictionForce_normalForce_accelDFT_class_-1.data"

cfg.MODEL.RESNET_NAME = "resnet50"
cfg.MODEL.TS_NET = "xvector"

if cfg.MODEL.RESNET_NAME == 'resnet50':
    cfg.MODEL.RESNET_PRETRAINED = "resnet50-19c8e357.pth"
elif cfg.MODEL.RESNET_NAME == "se_resnet50":
    cfg.MODEL.RESNET_PRETRAINED = "seresnet50-60a8950a85b2b.pkl"

cfg.FUSION_HEAD.METHOD = "MTUT"

cfg.MODEL.RESNET_FREEZE = True
cfg.MODEL.EXTRACTOR_FREEZE = False

cfg.MODEL.UNTIL_CONVERGE = False
cfg.MODEL.UNTIL_CONVERGE_THRESHOLD = 0.9

cfg.MODEL.AUX_LOSS = True
cfg.MODEL.AUX_LOSS_GAMMA = 0.0005
cfg.MODEL.AUX_LOSS_LR = 0.05

cfg.TRAIN.EPOCHES = 100

cfg.TRAIN.NUM_WORKERS = 4
cfg.TRAIN.BATCH_SIZE = 16
cfg.SOLVER.LR = 1e-5

cfg.FUSION_HEAD.COSINE = False
cfg.FUSION_HEAD.SCALECLS = 10
cfg.FUSION_HEAD.ACTIVATION = "ReLU"
cfg.FUSION_HEAD.FEATURE_DIMS = 2048  # Per modality feature vector dims
cfg.FUSION_HEAD.HIDDEN_DIMS = 128

cfg.FUSION_HEAD.MTUT.BETA = 0.5
cfg.FUSION_HEAD.MTUT.LAMDA = 0.5
cfg.FUSION_HEAD.MTUT.LAMDA_AE = 0.1
cfg.FUSION_HEAD.MTUT.AUXLOSS = "AUX_OUT_SSALoss"
cfg.FUSION_HEAD.MTUT.THRESOLD = 1e-12
cfg.FUSION_HEAD.MTUT.REG_METHOD = "log"
cfg.FUSION_HEAD.MTUT.T = 40

start_idx = 0


def train():
    datasets = CBMDatasetPickle(os.path.join(PICKLE_DIR, cfg.MODALITY.PICKLE_FILE))

    datasets.cfg.MODALITY.PICKLE_FILE = os.path.join(PICKLE_DIR, cfg.MODALITY.PICKLE_FILE)
    cfg.MODALITY = datasets.cfg.MODALITY
    cfg.MODEL.NUM_CLASSES = datasets.cfg.MODEL.NUM_CLASSES

    cfg.RECORD.LOG = log_filename(cfg, freeze=True)
    log = Logger(cfg.RECORD.LOG, when='D')

    print("========" * 5)
    log.logger.info(cfg)

    avg_best_test_acc = {}
    for r in cfg.MODALITY.REQUIRMENTS:
        avg_best_test_acc[r] = 0

    cbm_datasets = CBMDataset("/onekeyai_shared/CBM_FinalDatabase", cfg)

    for n in range(start_idx, 5):
        if n == 0:
            train_datasets, test_datasets = dataset_split(datasets, method='novel', desired_idx=n,
                                                          orignal_datasets=cbm_datasets, log=log)
        else:
            train_datasets, test_datasets = dataset_split(datasets, method='novel', desired_idx=n,
                                                          orignal_datasets=cbm_datasets, log=None)

        cfg.RECORD.LOG = log_filename(cfg, n, freeze=True)

        log_ = Logger(cfg.RECORD.LOG, when='D')

        log.logger.info("Total numbers of data: {:}, training data: {:} and testing data: {:} in valid {:}".
                        format(len(datasets), len(train_datasets), len(test_datasets), n))

        dataloader = DataLoader(train_datasets, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS,
                                collate_fn=datasets.CBM_collate_fn)
        test_dataloader = DataLoader(test_datasets, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                     num_workers=cfg.TRAIN.NUM_WORKERS,
                                     collate_fn=datasets.CBM_collate_fn, drop_last=True)
        loss_func = CrossEntropyLoss()

        net = build(cfg)
        if MULTIGPU:
            net = nn.DataParallel(net).cuda()
        optimizer = {}
        for r in cfg.MODALITY.REQUIRMENTS:
            optimizer[r] = Adam(filter(lambda p: p.requires_grad, net[r].parameters(recurse=True)),
                                lr=cfg.SOLVER.WEIGHT_DECAY,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        if cfg.MODEL.AUX_LOSS:
            center_loss_func = {}
            center_loss_optimizer = {}
            for r in cfg.MODALITY.REQUIRMENTS:
                center_loss_func[r] = CenterLoss(num_classes=cfg.MODEL.NUM_CLASSES,
                                                 feat_dim=cfg.FUSION_HEAD.FEATURE_DIMS)
                if cfg.MODEL.DEVICE == "cuda":
                    center_loss_func[r] = center_loss_func[r].cuda()

                center_loss_optimizer[r] = Adam(center_loss_func[r].parameters(), lr=cfg.MODEL.AUX_LOSS_LR)

            trainer = TrainerMTUT(net, dataloader, loss_func, optimizer, cfg,
                                  aux_loss=center_loss_func,
                                  aux_optimizer=center_loss_optimizer)
        else:
            trainer = TrainerMTUT(net, dataloader, loss_func, optimizer, cfg)

        if n == 0:
            log.logger.info(net)
            log.logger.info(trainer.ssa_loss)

        print("========" * 5)
        print("Start training!")
        best_test_acc = {}
        for r in cfg.MODALITY.REQUIRMENTS:
            best_test_acc[r] = 0.0

        for i in range(cfg.TRAIN.EPOCHES):
            if cfg.MODEL.AUX_LOSS:
                losses, ssa_losses, acc, nums, aux_losses = trainer.train()
                teacher_m, _ = min_in_dict(ssa_losses)
                log_.logger.info(
                    "EPOCH: {:}, Total Loss: {:}, SSA Loss: {:}, AUX Loss: {:}, ACC: {:}".format(i, losses, ssa_losses,
                                                                                                 aux_losses, acc))
                losses, ssa_losses, acc, nums, aux_losses = trainer.train_with_teacher(teacher_m)
                print(
                    "Teacher {:} EPOCH: {:}, Total Loss: {:}, SSA Loss: {:}, AUX Loss: {:}, ACC: {:}"
                        .format(teacher_m, i, losses, ssa_losses, aux_losses, acc))

            else:
                losses, ssa_losses, acc, nums = trainer.train()
                log_.logger.info(
                    "EPOCH: {:}, Total Loss: {:}, SSA Loss: {:}, ACC: {:}".format(i, losses, ssa_losses, acc))

            test_acc = trainer.test(test_dataloader)
            for r in cfg.MODALITY.REQUIRMENTS:
                if test_acc[r] > best_test_acc[r]:
                    save_model(trainer.model[r], cfg, n=n, specific_modality=[r], suffix="best")
                    best_test_acc[r] = test_acc[r]
            log_.logger.info("TEST EPOCH: {:}, ACC: {:}".format(i, test_acc))

        log.logger.critical("Best test acc in {:} is {:}".format(n, best_test_acc))

        for r in cfg.MODALITY.REQUIRMENTS:
            avg_best_test_acc[r] += best_test_acc[r]
        visualize_log_mtut(cfg.RECORD.LOG, cfg.MODALITY.REQUIRMENTS, aux_loss=cfg.MODEL.AUX_LOSS)
    for r in cfg.MODALITY.REQUIRMENTS:
        avg_best_test_acc[r] /= 5
    log.logger.critical("Average best test acc is {:}".format(avg_best_test_acc))

    print(cfg.RECORD.LOG)


train()