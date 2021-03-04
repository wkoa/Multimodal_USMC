import os
import torch
from libs.datasets.CBM_dataset import CBMDatasetPickle, CBMDataset
import matplotlib.pyplot as plt

pickle_file = "CBMDataset_sound_class_-1.data"
pickle_file = os.path.join('/onekeyai_shared/CBM_PickleData', pickle_file)

dataset = CBMDatasetPickle(pickle_file)
cbm_datasets = CBMDataset("/onekeyai_shared/CBM_FinalDatabase", dataset.cfg)

i = 10
data = dataset.__getitem__(10)[0]
print(cbm_datasets.data_path[10])
numpy_data = data.numpy()

plt.imsave("test.png", numpy_data)