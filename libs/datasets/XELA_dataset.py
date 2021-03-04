import os
from torch.utils.data import Dataset
import numpy as np
import wave
from python_speech_features import mfcc
import torch
from libs.configs.defaults import _C as cfg
from libs.utils import visualize_xela_sound

class XELADataset(Dataset):
    def __init__(self, data_root, cfg):
        """

        :param data_root:
        :param cfg:
        """
        self.sound_path = []
        self.normal_force_path = []
        self.friction_force_path = []

        self.cfg = cfg
        self.requirements = cfg.MODALITY.REQUIRMENTS

        for root, dirs, files in os.walk(data_root, topdown=True):
            for f in files:
                if f.endswith(".wav"):
                    self.sound_path.append(os.path.join(root, f))
                    self.normal_force_path.append(os.path.join(root, 'tactile_z.npy'))
                    self.friction_force_path.append(os.path.join(root, 'tactile_y.npy'))

        self.sound_path.sort()
        self.normal_force_path.sort()
        self.friction_force_path.sort()

    def __len__(self):
        return len(self.sound_path)

    def __getitem__(self, item):
        sound_file = wave.open(self.sound_path[item], "rb")
        params = sound_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = sound_file.readframes(nframes)
        sound_file.close()
        sound = np.fromstring(str_data, dtype=np.short)
        sound.shape = -1, 2
        sound = sound.T[0]

        sound = sound/8000
        normal_force = np.load(self.normal_force_path[item])
        friction_force = np.load(self.friction_force_path[item])
        normal_force = self.remove_zero(normal_force)
        normal_force = normal_force/50000
        friction_force = self.remove_zero(friction_force)
        friction_force = friction_force/20000

        if len(normal_force) > 300:
            normal_force = normal_force[-200:]
            friction_force = friction_force[-200:]

        if "Wood" in self.sound_path[item]:
            classes_idx = 0
        elif "Polymers" in self.sound_path[item]:
            classes_idx = 2
        elif "Metals" in self.sound_path[item]:
            classes_idx = 6
        else:
            raise RuntimeError

        sound_ = mfcc(sound, 44100, numcep=24, nfft=1103)
        sound_tensor = torch.FloatTensor(sound_.T[:, :150])

        normal_force_ = mfcc(normal_force, 100, numcep=24, nfft=1000)
        normal_force_tensor = torch.FloatTensor(normal_force_.T[:, :175])

        friction_force_ = mfcc(friction_force, 100, numcep=24, nfft=1000)
        friction_force_tensor = torch.FloatTensor(friction_force_.T[:, :175])

        return [sound_tensor, normal_force_tensor, friction_force_tensor, torch.tensor(classes_idx)]

    def CBM_collate_fn(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        output = []
        for i in range(self.cfg.MODALITY.NUMS):
            tmp_batch = [elem[i] for elem in batch]
            tmp_batch = torch.stack(tmp_batch)
            output.append(tmp_batch)
        target = [elem[-1] for elem in batch]
        target = torch.stack(target)
        return output, target

    def remove_zero(self, data):
        data = data.copy()
        if len(np.where(data == 0)[0]) > 0:
            for idx_zero in np.where(data == 0)[0]:
                if idx_zero == 0:
                    data[0] = data[1]/2
                elif idx_zero == len(data) - 1:
                    data[idx_zero] = data[-2]/2
                else:
                    data[idx_zero] = (data[idx_zero+1] + data[idx_zero-1])/2

        return data

if __name__ == "__main__":
    test_dataset = XELADataset('../../XELA_SOUND/train', cfg)

    root_path = "../../XELA_SOUND/test/Polymers/1/0/"
    sound_path = os.path.join(root_path, 'audio.wav')
    normal_path = os.path.join(root_path, 'tactile_z.npy')
    frcition_path = os.path.join(root_path, 'tactile_y.npy')

    sound_file = wave.open(sound_path, "rb")
    params = sound_file.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = sound_file.readframes(nframes)
    sound_file.close()
    sound = np.fromstring(str_data, dtype=np.short)
    sound.shape = -1, 2
    sound = sound.T[0]

    normal_force = np.load(normal_path)
    friction_force = np.load(frcition_path)
    normal_force = test_dataset.remove_zero(normal_force)
    friction_force = test_dataset.remove_zero(friction_force)

    visualize_xela_sound(sound, is_sound=True, save_dir="../../test.mp4")
    pass
