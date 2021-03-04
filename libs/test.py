from libs.configs.defaults import _C as cfg
from libs.datasets.CBM_dataset import *
from libs.models.baseline import BaselineEarlyFusion
from libs.utils import visualize_log_mtut

# cfg.MODALITY.TO_PICKLE = False
# cfg.MODALITY.PICKLE_FILE = "CBM_Dataset.data"
# cfg.MODEL.NUM_CLASSES = 4
# cfg.MODEL.RESNET_PRETRAINED = "../resnet50-19c8e357.pth"
#
#
# test_ = CBMDataset("../CBM_Dataset", cfg)
# net = BaselineEarlyFusion(cfg)

f_log = open("../log/MTUT/CBMDataset_sound_accelDFT_resnet50_xvector_ReLU_2048_JSDSSALoss_1.log")
visualize_log_mtut(f_log, ['sound', 'accelDFT'])

pass
