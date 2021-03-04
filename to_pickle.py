import pickle
import os
import torch
from libs.datasets.CBM_dataset import CBMDataset
from libs.configs.defaults import _C as cfg
from libs.utils import multiprocess_to_pickle, PCA_svd


cfg.MODALITY.TS_NOISE_SNR = -1
cfg.MODALITY.TS_METHOD = "PSD"
cfg.MODALITY.REQUIRMENTS = ['sound', 'normalForce', 'frictionForce', 'accelDFT']

cfg.MODALITY.NUMS = len(cfg.MODALITY.REQUIRMENTS)
cfg.MODALITY.SOUND.XVECTOR.INPUT_DIM = 24
cfg.MODALITY.FORCE.XVECTOR.INPUT_DIM = 24
cfg.MODALITY.ACCEL.XVECTOR.INPUT_DIM = 24

cfg.MODALITY.LABEL_LEVEL = "class-novel"
if cfg.MODALITY.LABEL_LEVEL == "class":
    cfg.MODEL.NUM_CLASSES = 8
elif cfg.MODALITY.LABEL_LEVEL == "subclass":
    cfg.MODEL.NUM_CLASSES = 33
elif cfg.MODALITY.LABEL_LEVEL == "class-novel":
    cfg.MODEL.NUM_CLASSES = 8
elif cfg.MODALITY.LABEL_LEVEL == "subclass-novel":
    cfg.MODEL.NUM_CLASSES = 33

data_root = "/home/wjh/LMTHMDataset/CBM_FinalDatabase"
pickle_file = "CBMDataset_" + "_".join(cfg.MODALITY.REQUIRMENTS) + "_" + cfg.MODALITY.LABEL_LEVEL \
              + '_{:}'.format(cfg.MODALITY.TS_NOISE_SNR) + "_psd.data"
pickle_file = os.path.join('/home/wjh/LMTHMDataset/CBM_PickleData', pickle_file)
cbm_dataset = CBMDataset(data_root, cfg, spectrogram=False, mel_spectrogram=False, raw_data=False)
print("Total data")
print(len(cbm_dataset.data_path))

pickle_file = open(pickle_file, 'rb')
data = pickle.load(pickle_file)

data = multiprocess_to_pickle(cbm_dataset, None, num_thread=5)

# for n in range(len(data[0]) - 1):
#     tmp_data = [data[i][n].squeeze() for i in range(len(data) - 1)]
#     tmp_tensor_data = torch.stack(tmp_data)
#     print(tmp_tensor_data.shape)
#     (U, S, V) = torch.pca_lowrank(tmp_tensor_data, 200, center=True)
#     tmp_tensor_data = torch.matmul(tmp_tensor_data, V[:, :200])
#     print(tmp_tensor_data.shape)
#     for i in range(len(data) - 1):
#         data[i][n] = tmp_tensor_data[i]
#
# pickle_file.close()
#
# pickle_file = "CBMDataset_" + "_".join(cfg.MODALITY.REQUIRMENTS) + "_" + cfg.MODALITY.LABEL_LEVEL \
#               + '_{:}'.format(cfg.MODALITY.TS_NOISE_SNR) + "_psd_svd.data"
# pickle_file = os.path.join('/home/wjh/LMTHMDataset/CBM_PickleData', pickle_file)

# pickle_file = open(pickle_file, 'wb')
pickle.dump(data, pickle_file)
pickle_file.close()

print(pickle_file)



