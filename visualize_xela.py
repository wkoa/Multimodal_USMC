from libs.datasets.XELA_dataset import XELADataset
import os
from libs.configs.defaults import _C as cfg
import wave
import numpy as np
from libs.utils import visualize_xela_sound

test_dataset = XELADataset('XELA_SOUND/train', cfg)
root_path = "XELA_SOUND/test/Wood/0/0/"
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

visualize_xela_sound(sound, is_sound=True, save_dir="sound.mp4")
visualize_xela_sound(normal_force, is_sound=False, save_dir="normal.mp4", tacitle_type='normal')
visualize_xela_sound(friction_force, is_sound=False, save_dir="friction.mp4", tacitle_type='friction')

