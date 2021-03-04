import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from libs.configs.defaults import _C as cfg
from libs.datasets.CBM_dataset import CBMDataset, CBMDatasetPickle
from libs.engien.train_loop import Trainer
from libs.models.baseline import BaselineEarlyFusion
from libs.logger import Logger
from libs.utils import *
from libs.models.build import build
from libs.modules.center_loss import CenterLoss
classes = ["Wood", "Biodegradable", "Polymers", "Ceramics", "Glasses", "Stones", "Metals", "Composites"]


cfg.MODALITY.TO_PICKLE = False

PICKLE_DIR = '/home/wjh/LMTHMDataset/CBM_PickleData'
cfg.MODALITY.PICKLE_FILE = "CBMDataset_accelDFT_class-novel_-1_psd_svd.data"

# cfg.MODALITY.PICKLE_FILE = "CBMDataset_frictionForce_class-novel_-1_mfccdelta.data"
# cfg.MODALITY.ACCEL.SPECTROGRAM = True
cfg.MODALITY.PICKLE_FILE = "CBMDataset_sound_frictionForce_class-novel_-1_psd_svd.data"
# cfg.MODALITY.PICKLE_FILE = "CBMDataset_sound_accelDFT_frictionForce_normalForce_accelDFT_class-novel_-1_psd.data"


cfg.MODEL.RESNET_NAME = "resnet18"
# cfg.MODEL.TS_NET = "xvector"
# cfg.MODEL.TS_NET = "vae_xvector"
# cfg.MODEL.TS_NET = "multi_xvector"
# cfg.MODEL.TS_NET = "acnn"
# cfg.MODEL.TS_NET = "RawLSTM"
# cfg.MODEL.TS_NET = "resnet"
# cfg.MODEL.TS_NET = "ticnn"
cfg.MODEL.TS_NET = "dal"

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
cfg.FUSION_HEAD.METHOD = "Early Fusion DAL"

# cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY = ['h1']

cfg.MODEL.RESNET_FREEZE = False
cfg.MODEL.EXTRACTOR_FREEZE = False

cfg.MODEL.AUX_LOSS = False
cfg.MODEL.AUX_LOSS_GAMMA = 0.0003
cfg.MODEL.AUX_LOSS_LR = 0.05

cfg.TRAIN.EPOCHES = 200

cfg.TRAIN.NUM_WORKERS = 4
cfg.TRAIN.BATCH_SIZE = 32
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
cfg.MODEL.ATTNXVECTOR.NUM_HEAD = 8
cfg.MODEL.ATTNXVECTOR.DIM_K = 64
cfg.MODEL.ATTNXVECTOR.DIM_V = 64
cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN = 2048

cfg.MODEL.LSTM.DIM_HIDDEN = 128
cfg.MODEL.LSTM.NUM_LAYERS = 2

start_idx = 0


def train():
    datasets = CBMDatasetPickle(os.path.join(PICKLE_DIR, cfg.MODALITY.PICKLE_FILE))

    datasets.cfg.MODALITY.PICKLE_FILE = os.path.join(PICKLE_DIR, cfg.MODALITY.PICKLE_FILE)
    cfg.MODALITY = datasets.cfg.MODALITY

    cfg.MODEL.NUM_CLASSES = datasets.cfg.MODEL.NUM_CLASSES

    global_log = cfg.RECORD.LOG = log_filename(cfg)
    log = Logger(cfg.RECORD.LOG, when='D')

    print("========" * 5)
    log.logger.info(cfg)

    avg_best_test_acc = 0
    avg_confuse_matrix = np.zeros((cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES))
    cbm_datasets = CBMDataset("/home/wjh/LMTHMDataset/CBM_FinalDatabase", cfg)

    for n in range(start_idx, 5):
        if n == 0:
            train_datasets, test_datasets = dataset_split(datasets, method='novel', desired_idx=n,
                                                          orignal_datasets=cbm_datasets, log=log)
        else:
            train_datasets, test_datasets = dataset_split(datasets, method='novel', desired_idx=n,
                                                          orignal_datasets=cbm_datasets, log=None)

        cfg.RECORD.LOG = log_filename(cfg, n)

        log_ = Logger(cfg.RECORD.LOG, when='D')

        log.logger.info("Total numbers of data: {:}, training data: {:} and testing data: {:} in valid {:}".
                        format(len(datasets), len(train_datasets), len(test_datasets), n))

        dataloader = DataLoader(train_datasets, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS,
                                collate_fn=datasets.CBM_collate_fn)
        test_dataloader = DataLoader(test_datasets, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                     num_workers=cfg.TRAIN.NUM_WORKERS,
                                     collate_fn=datasets.CBM_collate_fn, drop_last=False)
        loss_func = CrossEntropyLoss()
        net = build(cfg)

        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters(recurse=True)), lr=cfg.SOLVER.WEIGHT_DECAY,
                         weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        if n == 0:
            log.logger.info(net)
            params = cal_model_params_num(net.parameters(recurse=True))
            log.logger.info("Model Params: {:}".format(params))

            # net_flops = cal_model_parm_flops(net, {'sound': torch.FloatTensor(train_datasets.__getitem__(0)[0]).unsqueeze(0).cuda()})
            # log.logger.info("FLOPs: {:}".format(net_flops))

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
                save_model(trainer.model, cfg, n=n, suffix="Best")
            log_.logger.info("TEST EPOCH: {:}, ACC: {:}".format(i, test_acc))
        log.logger.critical("Best test acc in {:} is {:}".format(n, best_test_acc))
        avg_best_test_acc += best_test_acc
        # tmp_confuse_matrix = best_confuse_matrix/best_confuse_matrix.sum(axis=1)
        avg_confuse_matrix += best_confuse_matrix
        visualize_log(cfg.RECORD.LOG, aux_loss=cfg.MODEL.AUX_LOSS)
        plot_confuse_matrix(best_confuse_matrix, classes, cfg.RECORD.LOG)

    avg_best_test_acc /= 5
    avg_confuse_matrix /= 5
    confuse_sum = avg_confuse_matrix.sum(axis=1).reshape([8, 1])
    confuse_sum[4, :] = 1e-9
    avg_confuse_matrix = avg_confuse_matrix/confuse_sum
    # avg_confuse_matrix[:, 4] = np.zeros(8)
    log.logger.critical("Average best test acc is {:}".format(avg_best_test_acc))
    log.logger.critical("Average best test confuse matrix is {:}".format(avg_confuse_matrix))
    print(cfg.RECORD.LOG)
    plot_confuse_matrix(avg_confuse_matrix, classes, global_log)
    return avg_best_test_acc


def params_search():
    best_acc = 0
    best_params = None

    params_log = Logger(os.path.join('log', 'search_params.log'), when='D')
    for dim_k in [16, 32, 64, 128]:
        for dim_hidden in [512, 1024, 2048]:
            for num_layer in [1, 2, 3, 4, 5, 6, 7, 8]:
                for num_head in [1, 2, 3, 4, 5, 6, 7, 8]:
                    cfg.MODEL.ATTNXVECTOR.NUM_LAYER = num_layer
                    cfg.MODEL.ATTNXVECTOR.NUM_HEAD = num_head
                    cfg.MODEL.ATTNXVECTOR.DIM_K = dim_k
                    cfg.MODEL.ATTNXVECTOR.DIM_V = dim_k
                    cfg.MODEL.ATTNXVECTOR.DIM_HIDDEN = dim_hidden

                    avg_best_acc = train()
                    if avg_best_acc > best_acc:
                        best_acc = avg_best_acc
                        best_params = [dim_k, dim_hidden, num_layer, num_head]
                    params_log.logger.info('Current Result: {:} in params {:}'.format(avg_best_acc, [dim_k, dim_hidden, num_layer, num_head]))
                    params_log.logger.info('Current Best Result: {:} in params {:}'.format(best_acc, best_params))

    params_log.logger.info('Best Result: {:} in params {:}'.format(best_acc, best_params))


# params_search()
train()
