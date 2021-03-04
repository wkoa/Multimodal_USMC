import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from libs.configs.defaults import _C as cfg
from libs.datasets.XELA_dataset import XELADataset
from libs.engien.train_loop import Trainer
from libs.models.baseline import BaselineEarlyFusion
from libs.logger import Logger
from libs.utils import *
from libs.models.build import build
from libs.modules.center_loss import CenterLoss
classes = ["Wood", "Biodegradable", "Polymers", "Ceramics", "Glasses", "Stones", "Metals", "Composites"]


cfg.MODALITY.TO_PICKLE = False

# PICKLE_DIR = '/onekeyai_shared/CBM_PickleData'
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_accelDFT_class-novel_-1_raw.data"
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_frictionForce_class-novel_-1_mfccdelta.data"
# cfg.MODALITY.ACCEL.SPECTROGRAM = True
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_frictionForce_normalForce_class-novel_-1_logmel.data"
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_sound_accelDFT_frictionForce_normalForce_class-novel_-1_logmel.data"


cfg.MODEL.RESNET_NAME = "resnet18"
# cfg.MODEL.TS_NET = "xvector"
# cfg.MODEL.TS_NET = "vae_xvector"
# cfg.MODEL.TS_NET = "multi_xvector"
# cfg.MODEL.TS_NET = "acnn"
# cfg.MODEL.TS_NET = "RawLSTM"
# cfg.MODEL.TS_NET = "resnet_2"
cfg.MODEL.TS_NET = "ticnn"

if cfg.MODEL.RESNET_NAME == 'resnet50':
    cfg.MODEL.RESNET_PRETRAINED = "resnet50-19c8e357.pth"
elif cfg.MODEL.RESNET_NAME == "resnet18":
    cfg.MODEL.RESNET_PRETRAINED = "resnet18-5c106cde.pth"
elif cfg.MODEL.RESNET_NAME == "se_resnet50":
    cfg.MODEL.RESNET_PRETRAINED = "seresnet50-60a8950a85b2b.pkl"

# cfg.FUSION_HEAD.METHOD = "Re-weighting Fusion"
# cfg.FUSION_HEAD.METHOD = "Early Fusion v2"
# cfg.FUSION_HEAD.METHOD = "Attention"
# cfg.FUSION_HEAD.METHOD = "AttentionStat"
# cfg.FUSION_HEAD.METHOD = "Early Fusion Stat"
# cfg.FUSION_HEAD.METHOD = "Early Fusion SE"
# cfg.FUSION_HEAD.METHOD = "Early Fusion StackSE"
# cfg.FUSION_HEAD.METHOD = "Late Fusion"
# cfg.FUSION_HEAD.METHOD = "Tensor Fusion"
# cfg.FUSION_HEAD.METHOD = "Tensor Fusion Attention"
# cfg.FUSION_HEAD.METHOD = "LSTM"


# cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY = ['h1']

cfg.MODEL.RESNET_FREEZE = False
cfg.MODEL.EXTRACTOR_FREEZE = False

cfg.MODEL.AUX_LOSS = False
cfg.MODEL.AUX_LOSS_GAMMA = 0.0003
cfg.MODEL.AUX_LOSS_LR = 0.05

cfg.TRAIN.EPOCHES = 50

cfg.TRAIN.NUM_WORKERS = 4
cfg.TRAIN.BATCH_SIZE = 4
cfg.SOLVER.LR = 1e-5

cfg.FUSION_HEAD.COSINE = False
cfg.FUSION_HEAD.SCALECLS = 10

cfg.FUSION_HEAD.ACTIVATION = "ReLU"
cfg.FUSION_HEAD.FEATURE_DIMS = 256  # Per modality feature vector dims
cfg.FUSION_HEAD.HIDDEN_DIMS = 128

#
cfg.FUSION_HEAD.ATTENTIONFUSION.NUM_POSITION = 4
cfg.FUSION_HEAD.ATTENTIONFUSION.NUM_LAYER = 4
cfg.FUSION_HEAD.ATTENTIONFUSION.NUM_HEAD = 4
cfg.FUSION_HEAD.ATTENTIONFUSION.DIM_K = 64
cfg.FUSION_HEAD.ATTENTIONFUSION.DIM_V = 64
cfg.FUSION_HEAD.ATTENTIONFUSION.DIM_HIDDEN = 2048
cfg.FUSION_HEAD.ATTENTIONFUSION.ADD = False

#
# cfg.FUSION_HEAD.TENSORFUSION.DIMS = 64
#
cfg.MODEL.ATTNXVECTOR.NUM_LAYER = 6
cfg.MODEL.ATTNXVECTOR.NUM_HEAD = 4
cfg.MODEL.ATTNXVECTOR.DIM_K = 64
cfg.MODEL.ATTNXVECTOR.DIM_V = 64
cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN = 2048

cfg.MODEL.LSTM.DIM_HIDDEN = 128
cfg.MODEL.LSTM.NUM_LAYERS = 2

start_idx = 0

PRETRAINED_MODEL = "./CBMDataset_sound_normalForce_frictionForce_resnet18_ticnn_ReLU_256_Best_0.pkl"


def train():
    # datasets = CBMDatasetPickle(os.path.join(PICKLE_DIR, cfg.MODALITY.PICKLE_FILE))

    # datasets.cfg.MODALITY.PICKLE_FILE = os.path.join(PICKLE_DIR, cfg.MODALITY.PICKLE_FILE)
    # cfg.MODALITY = datasets.cfg.MODALITY

    cfg.MODEL.NUM_CLASSES = 8

    cfg.MODALITY.REQUIRMENTS = ['sound', 'normalForce', 'frictionForce']
    cfg.MODALITY.NUMS = 3
    cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM = 24
    cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM = 24
    # cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM = 24

    global_log = cfg.RECORD.LOG = log_filename(cfg)
    log = Logger(cfg.RECORD.LOG, when='D')

    print("========" * 5)
    log.logger.info(cfg)

    avg_best_test_acc = 0
    avg_confuse_matrix = np.zeros((cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES))
    train_datasets = XELADataset("./XELA_SOUND/train/", cfg)

    test_datasets = XELADataset("./XELA_SOUND/test/", cfg)


    cfg.RECORD.LOG = log_filename(cfg, 0)

    log_ = Logger(cfg.RECORD.LOG, when='D')

    log.logger.info("Total numbers of data: {:}, training data: {:} and testing data: {:} in valid {:}".
                    format(len(train_datasets) + len(test_datasets), len(train_datasets), len(test_datasets), 0))

    dataloader = DataLoader(train_datasets, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.TRAIN.NUM_WORKERS,
                            collate_fn=train_datasets.CBM_collate_fn)
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=True,
                                 num_workers=cfg.TRAIN.NUM_WORKERS,
                                 collate_fn=test_datasets.CBM_collate_fn, drop_last=False)
    loss_func = CrossEntropyLoss()
    net = build(cfg)
    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters(recurse=True)), lr=cfg.SOLVER.WEIGHT_DECAY,
                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    log.logger.info(net)
    params = cal_model_params_num(net.parameters(recurse=True))
    # net_flops = cal_model_parm_flops(net, {'sound': train_datasets.__getitem__(0)[0].unsqueeze(0).cuda()})

    log.logger.info("Model Params: {:}".format(params))

    if cfg.MODEL.AUX_LOSS:
        center_loss_func = CenterLoss(num_classes=cfg.MODEL.NUM_CLASSES,
                                      feat_dim=cfg.MODALITY.NUMS * cfg.FUSION_HEAD.FEATURE_DIMS)
        center_loss_optimizer = Adam(center_loss_func.parameters(), lr=cfg.MODEL.AUX_LOSS_LR)
        trainer = Trainer(net, dataloader, loss_func, optimizer, cfg,
                          aux_loss=center_loss_func,
                          aux_optimizer=center_loss_optimizer)
    else:
        trainer = Trainer(net, dataloader, loss_func, optimizer, cfg)

    print("========" * 5)
    print("Start training!")
    best_test_acc = 0.0
    best_confuse_matrix = None
    trainer.model.load_state_dict(torch.load(PRETRAINED_MODEL))

    for i in range(cfg.TRAIN.EPOCHES):
        if cfg.MODEL.AUX_LOSS:
            losses, aux_losses, acc, nums = trainer.train()
            log_.logger.info("EPOCH: {:}, Loss: {:}, Aux Loss: {:}, ACC: {:}".format(i, losses, aux_losses, acc))
        else:
            losses, acc, nums = trainer.train()
            log_.logger.info("EPOCH: {:}, Loss: {:}, ACC: {:}".format(i, losses, acc))

        test_acc, confuse_matrix = trainer.test(test_dataloader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_confuse_matrix = confuse_matrix
            save_model(trainer.model, cfg, n=0, suffix="Best", total_model=True)
        log_.logger.info("TEST EPOCH: {:}, ACC: {:}".format(i, test_acc))
    log.logger.critical("Best test acc in {:} is {:}".format(0, best_test_acc))
    # avg_best_test_acc += best_test_acc
    # tmp_confuse_matrix = best_confuse_matrix/best_confuse_matrix.sum(axis=1)
    avg_confuse_matrix += best_confuse_matrix
    visualize_log(cfg.RECORD.LOG, aux_loss=cfg.MODEL.AUX_LOSS)
    plot_confuse_matrix(best_confuse_matrix, classes, cfg.RECORD.LOG)

    confuse_sum = avg_confuse_matrix.sum(axis=1).reshape([8, 1])
    confuse_sum[1, :] = 1e-9
    confuse_sum[3, :] = 1e-9
    confuse_sum[4, :] = 1e-9
    confuse_sum[5, :] = 1e-9
    confuse_sum[7, :] = 1e-9
    avg_confuse_matrix = avg_confuse_matrix / confuse_sum
    # avg_confuse_matrix[:, 4] = np.zeros(8)
    print(cfg.RECORD.LOG)
    plot_confuse_matrix(avg_confuse_matrix, classes, global_log)
    return avg_best_test_acc, avg_confuse_matrix

# params_search()
avg_confuse_matrix = None

for i in range(10):
    test_acc, confuse_matrix = train()
    if avg_confuse_matrix is None:
        avg_confuse_matrix = confuse_matrix
    else:
        avg_confuse_matrix += confuse_matrix

print(avg_confuse_matrix)