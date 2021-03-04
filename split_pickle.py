from libs.utils import split_pickle
import pickle
import os

pickle_file = "CBMDataset_sound_normalForce_frictionForce_accelDFT_class-novel_-1_psd_svd.data"
pickle_file = os.path.join('/home/wjh/LMTHMDataset/CBM_PickleData', pickle_file)

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
    split_pickle(data, '/home/wjh/LMTHMDataset/CBM_PickleData', raw_data=False)
