import os
import numpy as np
import torch

from PIL import Image
from python_speech_features import mfcc
from scipy import signal
from scipy.io import wavfile
from torchvision.transforms import transforms

from libs.configs.defaults import _C

LMT_DEFAULT_CFG = _C


def modality_preprocessor(name, modality, cfg):
    x = modality.copy()
    if cfg.MODALITY.TS_NOISE_SNR <= 0:
        pass
    else:
        if "image" in name:
            pass
        else:
            x = _add_noise(x, cfg.MODALITY.TS_NOISE_SNR)

    if "image" in name:
        return _image_processor(x, cfg)
    elif "sound" in name:
        return _sound_processor(x, cfg)
    elif "accel" in name:
        return _accel_processor(x, cfg)
    elif "Force" in name:
        return _force_processor(x, cfg)
    else:
        raise NotImplementedError


def _image_processor(x, cfg):
    x = Image.fromarray(x)
    return transforms.Compose([transforms.Resize((cfg.MODALITY.IMAGE.SIZE, cfg.MODALITY.IMAGE.SIZE)),
                               transforms.ToTensor()])(x)


def _sound_processor(wavfile_name, cfg):
    x = wavfile.read(wavfile_name)

    if cfg.MODALITY.TS_METHOD == "PSD":
        _, ts_result = signal.periodogram(x, fs=cfg.MODALITY.SOUND.PSD.FS, nfft=cfg.MODALITY.SOUND.PSD.NFFT, axis=0)
        return ts_result
    elif cfg.MODALITY.TS_METHOD == "MFCC":
        ts_result = mfcc(x, cfg.MODALITY.SOUND.SAMPLERATE,
                         numcep=cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM, )
        return torch.FloatTensor(ts_result.T)
    elif cfg.MODALITY.TS_METHOD == "xvector":
        ts_result = mfcc(x, cfg.MODALITY.SOUND.SAMPLERATE,
                         numcep=cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.SOUND.XVECTOR.NFFT)
        return torch.FloatTensor(ts_result.T)
    else:
        raise NotImplementedError


def _force_processor(x, cfg):
    x = open(x, 'r')
    x = np.array(float(x.readlines()))

    x = x.reshape((x.shape, 1))

    if cfg.MODALITY.TS_METHOD == "PSD":
        _, ts_result = signal.periodogram(x, fs=cfg.MODALITY.FORCE.PSD.FS, nfft=cfg.MODALITY.FORCE.PSD.NFFT, axis=0)
        return torch.FloatTensor(ts_result)
    elif cfg.MODALITY.TS_METHOD == "MFCC":
        ts_result = mfcc(x, cfg.MODALITY.FORCE.SAMPLERATE, numcep=cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM)
        return torch.FloatTensor(ts_result.T)
    elif cfg.MODALITY.TS_METHOD == "xvector":
        ts_result = mfcc(x, cfg.MODALITY.FORCE.SAMPLERATE,
                         numcep=cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM,
                         nfft=cfg.MODALITY.FORCE.XVECTOR.NFFT)
        return torch.FloatTensor(ts_result.T)
    else:
        raise NotImplementedError

def _accel_processor(x, cfg):
    pass


def _add_noise(signal, SNR=5):
    noise = np.random.randn(*signal.shape)
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))

    noise = (np.sqrt(noise_variance)/np.std(noise))*noise

    return noise + signal


LMT_DEFAULT_CFG.MODALITY.REQUIRMENTS = ['sound', 'accelDFT', 'force', 'image']

LMT_DEFAULT_CFG.MODALITY.NUMS = len(LMT_DEFAULT_CFG.MODALITY.REQUIRMENTS)
LMT_DEFAULT_CFG.MODALITY.TS_METHOD = "xvector" # "PSD", "MFCC", "xvector", "se_xvector"
LMT_DEFAULT_CFG.MODALITY.TO_PICKLE = False
LMT_DEFAULT_CFG.MODALITY.PICKLE_FILE = "CBMDataset_" + "_".join(_C.MODALITY.REQUIRMENTS) + ".data"
LMT_DEFAULT_CFG.MODALITY.LABEL_LEVEL = "class" # "class", "subclass", or "material"
LMT_DEFAULT_CFG.MODALITY.TS_NOISE_SNR = -1
# -----------------------------------------------------------------------------
# Image process params
# -----------------------------------------------------------------------------
LMT_DEFAULT_CFG.MODALITY.IMAGE.PIXEL_MEAN = [103.530, 116.280, 123.675]
LMT_DEFAULT_CFG.MODALITY.IMAGE.PIXEL_STD = [1.0, 1.0, 1.0]
LMT_DEFAULT_CFG.MODALITY.IMAGE.SIZE = 256

# _C.MODALITY.IMAGE.TRANSFORMER = transforms.Compose([transforms.ToTensor()])

# -----------------------------------------------------------------------------
# Sound process params
# -----------------------------------------------------------------------------
LMT_DEFAULT_CFG.MODALITY.SOUND.SAMPLERATE = 44100

# PSD params
LMT_DEFAULT_CFG.MODALITY.SOUND.PSD.FS = 1
LMT_DEFAULT_CFG.MODALITY.SOUND.PSD.NFFT = 1103

# MFCC params

# XVector params
LMT_DEFAULT_CFG.MODALITY.SOUND.XVECTOR.INPUT_DIM = 13
LMT_DEFAULT_CFG.MODALITY.SOUND.XVECTOR.NFFT = 1103
LMT_DEFAULT_CFG.MODALITY.SOUND.XVECTOR.ACTIVATION = "ReLU"


# -----------------------------------------------------------------------------
# Force process params
# -----------------------------------------------------------------------------
LMT_DEFAULT_CFG.MODALITY.FORCE.SAMPLERATE = 10000

# PSD params
LMT_DEFAULT_CFG.MODALITY.FORCE.PSD.NFFT = 1024
# MFCC params

# XVector params
LMT_DEFAULT_CFG.MODALITY.FORCE.XVECTOR.INPUT_DIM = 13
LMT_DEFAULT_CFG.MODALITY.FORCE.XVECTOR.NFFT = 1000
LMT_DEFAULT_CFG.MODALITY.FORCE.XVECTOR.ACTIVATION = "ReLU"

# -----------------------------------------------------------------------------
# Accel process params
# -----------------------------------------------------------------------------
LMT_DEFAULT_CFG.MODALITY.ACCEL.SAMPLERATE = 3000
# PSD params

# _C.MODALITY.ACCEL.PSD.FS = 10000
LMT_DEFAULT_CFG.MODALITY.ACCEL.PSD.NFFT = 1000
# MFCC params

# XVector params
LMT_DEFAULT_CFG.MODALITY.ACCEL.XVECTOR.INPUT_DIM = 13
LMT_DEFAULT_CFG.MODALITY.ACCEL.XVECTOR.NFFT = 1000
LMT_DEFAULT_CFG.MODALITY.ACCEL.XVECTOR.ACTIVATION = "ReLU"