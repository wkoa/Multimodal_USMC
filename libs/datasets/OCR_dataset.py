import os

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from libs.utils import chunks

ACTIONS = ["crush", "grasp", "hold", "lift_slow", "low_drop", "poke", "push", "shake", "tap"]

ACTIONS_FRAME_MAP = dict(zip(ACTIONS, [47, 19, 11, 42, 21, 32, 52, 60, 23]))


class OCRActionDataset(Dataset):
    """
        Process raw data of OCR with specific action
    """
    def __init__(self, root_dir, action="push", as_image=True):
        img_root = os.path.join(root_dir, "visual_data")

        # Label part
        object_names = os.listdir(img_root)
        class_names = [o_n.split("_")[0] for o_n in object_names]
        class_names = list(set(class_names))  # To delete repeat part.
        class_names.sort()
        self.classes = class_names
        self.classes_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._action = action

        action_img_dirs = dict(zip(ACTIONS, [[] for _ in ACTIONS]))
        self.action_img = dict(zip(ACTIONS, [[] for _ in ACTIONS]))
        self.action_haptic = dict(zip(ACTIONS, [[] for _ in ACTIONS]))
        self.action_hearing = dict(zip(ACTIONS, [[] for _ in ACTIONS]))

        action_imgs_num = dict(zip(ACTIONS, [[] for _ in ACTIONS]))

        for root, dirs, files in os.walk(img_root):
            for d in dirs:
                pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def _interpolate(self):
        pass


def _test_main():
    root_dir = "/onekeyai_shared/OCRRawDataset"
    test = OCRActionDataset(root_dir)
    item = test.__getitem__(0)


if __name__ == "__main__":
    import pandas as pd

    # _test_main()
    ticks = open("/onekeyai_shared/OCRRawDataset/rc_data/ball_base/trial_1/exec_1/crush/hearing/1301700954114670.ticks", 'rb')
    content = pickle.load(ticks)
    for line in content:
        print(line)