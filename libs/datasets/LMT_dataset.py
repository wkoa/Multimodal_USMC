import os
import pickle
import pickletools

import torch
from scipy.io import wavfile
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class LMTDataset(Dataset):
    def __init__(self, data_root, cfg, train=True):
        files_name = [os.path.splitext(fn)[0] for fn in os.listdir(data_root) if fn.lower().endswith('.jpg')]
        files_name.sort()
        classes_to_idx = self._find_classes(files_name)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def _find_classes(self, files_name):
        class_to_idx = {files_name[i]: i for i in range(len(files_name))}
        return class_to_idx