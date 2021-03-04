from libs.utils import concat_pickle
import pickle
import os

pickle_dir = '/home/wjh/LMTHMDataset/CBM_PickleData'

modality_file_list = ["CBMDataset_sound_class-novel_-1_psd_svd.data",
                      # "CBMDataset_accelDFT_class-novel_-1_psd.data",
                      "CBMDataset_frictionForce_class-novel_-1_psd_svd.data",
                      # "CBMDataset_normalForce_class-novel_-1_psd.data",
                      ]

m_list = []
for mf in modality_file_list:
    mf = os.path.join(pickle_dir, mf)
    f = open(mf, 'rb')
    print("Load pickle modality: {:}".format(mf))

    m = pickle.load(f)
    m_list.append(m)
    f.close()

concat_pickle(m_list, pickle_dir, spectrogram=False, log_mel=False, raw_data=False)
