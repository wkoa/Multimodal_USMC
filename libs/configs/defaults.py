# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import *

from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 8
_C.MODEL.CLASSES = []
# Path (possibly with schema like catalog:// or detectron2://) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""
_C.MODEL.INIT_METHOD = "kaiming"
_C.MODEL.RESNET_PRETRAINED = False # "resnet50-19c8e357.pth" # Can be the path of resnet weights.
_C.MODEL.RESNET_FREEZE = False
_C.MODEL.RESNET_NAME = "resnet50"  # "se_resnet50", "resnet50"
_C.MODEL.FEATURE_FUSION = "concat"  # "concat", "add", "none"
_C.MODEL.TS_NET = "xvector"  # "se_vector", "xvector", "attn_xvector"
_C.MODEL.EXTRACTOR_FREEZE = False
_C.MODEL.UNTIL_CONVERGE = False
_C.MODEL.UNTIL_CONVERGE_THRESHOLD = 0.9

_C.MODEL.AUX_LOSS = False
_C.MODEL.AUX_LOSS_GAMMA = 0.0005
_C.MODEL.AUX_LOSS_LR = 0.05

_C.MODEL.ATTNXVECTOR = CN()
_C.MODEL.ATTNXVECTOR.NUM_LAYER = 6
_C.MODEL.ATTNXVECTOR.NUM_HEAD = 8
_C.MODEL.ATTNXVECTOR.DIM_K = 64
_C.MODEL.ATTNXVECTOR.DIM_V = 64
_C.MODEL.ATTNXVECTOR.DIM_HIDDEN = 2048

# LSTM params
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.DIM_HIDDEN = 128
_C.MODEL.LSTM.NUM_LAYERS = 2
# -----------------------------------------------------------------------------
# Modality process params
# -----------------------------------------------------------------------------
_C.MODALITY = CN()
_C.MODALITY.REQUIRMENTS = ['macroImage', 'sound', 'frictionForce']
_C.MODALITY.NUMS = len(_C.MODALITY.REQUIRMENTS)
_C.MODALITY.TS_METHOD = "xvector" # "PSD", "MFCC", "xvector", "se_xvector"
_C.MODALITY.TO_PICKLE = False
_C.MODALITY.PICKLE_FILE = "CBMDataset_" + "_".join(_C.MODALITY.REQUIRMENTS) + ".data"
_C.MODALITY.LABEL_LEVEL = "class" # "class", "subclass", or "material"
_C.MODALITY.TS_NOISE_SNR = -1

# -----------------------------------------------------------------------------
# Image process params
# -----------------------------------------------------------------------------
_C.MODALITY.IMAGE = CN()
_C.MODALITY.IMAGE.PIXEL_MEAN = [103.530, 116.280, 123.675]
_C.MODALITY.IMAGE.PIXEL_STD = [1.0, 1.0, 1.0]
_C.MODALITY.IMAGE.SIZE = 256

# _C.MODALITY.IMAGE.TRANSFORMER = transforms.Compose([transforms.ToTensor()])

# -----------------------------------------------------------------------------
# Sound process params
# -----------------------------------------------------------------------------
_C.MODALITY.SOUND = CN()
_C.MODALITY.SOUND.SAMPLERATE = 44100
_C.MODALITY.SOUND.SPECTROGRAM = False
# PSD params
_C.MODALITY.SOUND.PSD = CN()
_C.MODALITY.SOUND.PSD.FS = 1
_C.MODALITY.SOUND.PSD.NFFT = 1103

# MFCC params
_C.MODALITY.SOUND.MFCC = CN()

# XVector params
_C.MODALITY.SOUND.XVECTOR = CN()
_C.MODALITY.SOUND.XVECTOR.INPUT_DIM = 13
_C.MODALITY.SOUND.XVECTOR.NFFT = 1103
_C.MODALITY.SOUND.XVECTOR.ACTIVATION = "ReLU"

# -----------------------------------------------------------------------------
# Force process params
# -----------------------------------------------------------------------------
_C.MODALITY.FORCE = CN()
_C.MODALITY.FORCE.SAMPLERATE = 3000
_C.MODALITY.FORCE.SPECTROGRAM = False

# PSD params
_C.MODALITY.FORCE.PSD = CN()
# _C.MODALITY.FORCE.PSD.FS = 10000
_C.MODALITY.FORCE.PSD.NFFT = 1000
# MFCC params
_C.MODALITY.FORCE.MFCC = CN()

# XVector params
_C.MODALITY.FORCE.XVECTOR = CN()
_C.MODALITY.FORCE.XVECTOR.INPUT_DIM = 13
_C.MODALITY.FORCE.XVECTOR.NFFT = 1000
_C.MODALITY.FORCE.XVECTOR.ACTIVATION = "ReLU"

# -----------------------------------------------------------------------------
# Accel process params
# -----------------------------------------------------------------------------
_C.MODALITY.ACCEL = CN()
_C.MODALITY.ACCEL.SAMPLERATE = 3000
_C.MODALITY.ACCEL.SPECTROGRAM = False
# PSD params
_C.MODALITY.ACCEL.PSD = CN()
# _C.MODALITY.ACCEL.PSD.FS = 10000
_C.MODALITY.ACCEL.PSD.NFFT = 1000
# MFCC params
_C.MODALITY.ACCEL.MFCC = CN()

# XVector params
_C.MODALITY.ACCEL.XVECTOR = CN()
_C.MODALITY.ACCEL.XVECTOR.INPUT_DIM = 13
_C.MODALITY.ACCEL.XVECTOR.NFFT = 1000
_C.MODALITY.ACCEL.XVECTOR.ACTIVATION = "ReLU"

# -----------------------------------------------------------------------------
# Fusion head
# -----------------------------------------------------------------------------
_C.FUSION_HEAD = CN()
_C.FUSION_HEAD.METHOD = "Early Fusion" # "Re-weighting Fusion"
# _C.FUSION_HEAD.METHOD = "Re-weighting Fusion" # "Re-weighting Fusion"

_C.FUSION_HEAD.FEATURE_DIMS = 512 # Per modality feature vector dims
_C.FUSION_HEAD.HIDDEN_DIMS = 128
_C.FUSION_HEAD.ACTIVATION = "ReLU"
_C.FUSION_HEAD.DROPOUT = 0.2
_C.FUSION_HEAD.COSINE = False
_C.FUSION_HEAD.SCALECLS = 10
_C.FUSION_HEAD.REWEIGHTINGFUSION = CN()
_C.FUSION_HEAD.REWEIGHTINGFUSION.DIM_REDUCTION = 16

_C.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY = ['h1', 'h2', 'h3', 'h4']

_C.FUSION_HEAD.ATTENTIONFUSION = CN()
_C.FUSION_HEAD.ATTENTIONFUSION.NUM_POSITION = 200
_C.FUSION_HEAD.ATTENTIONFUSION.NUM_LAYER = 6
_C.FUSION_HEAD.ATTENTIONFUSION.NUM_HEAD = 8
_C.FUSION_HEAD.ATTENTIONFUSION.DIM_K = 64
_C.FUSION_HEAD.ATTENTIONFUSION.DIM_V = 64
_C.FUSION_HEAD.ATTENTIONFUSION.DIM_HIDDEN = 2048
_C.FUSION_HEAD.ATTENTIONFUSION.POSITION_ENC = True
_C.FUSION_HEAD.ATTENTIONFUSION.ADD = False

_C.FUSION_HEAD.TENSORFUSION = CN()
_C.FUSION_HEAD.TENSORFUSION.DIMS = 64

_C.FUSION_HEAD.MTUT = CN()
_C.FUSION_HEAD.MTUT.BETA = 2.0
_C.FUSION_HEAD.MTUT.LAMDA = 0.05
_C.FUSION_HEAD.MTUT.LAMDA_AE = 0.5
_C.FUSION_HEAD.MTUT.THRESOLD = 1e-5
_C.FUSION_HEAD.MTUT.REG_METHOD = "exp"
_C.FUSION_HEAD.MTUT.AUXLOSS = "SSALoss"
_C.FUSION_HEAD.MTUT.NORM = True
_C.FUSION_HEAD.MTUT.LAST_FEATURE = False
_C.FUSION_HEAD.MTUT.T = 40
# -----------------------------------------------------------------------------
# Train params
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.EPOCHES = 50

_C.TRAIN.NUM_WORKERS = 8

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-5
_C.SOLVER.WEIGHT_DECAY = 1e-4

_C.RECORD = CN()
_C.RECORD.TENSORBOARD = True
_C.RECORD.LOG = "CBMDataset_" + "_".join(_C.MODALITY.REQUIRMENTS) + ".log"
_C.RECORD.MEMO = ""