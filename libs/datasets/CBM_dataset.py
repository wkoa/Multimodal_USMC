import pickle

from torch.utils.data import Dataset

from libs.datasets._CBM_dataset_aux import *

CBM_DATASET_ATTR = CBM_DATASET_ATTR


class CBMDataset(Dataset):
    def __init__(self, data_root, cfg, spectrogram=False, mel_spectrogram=False, raw_data=False):
        """

        :param data_root: root dir of CBM_Dataset
        :param seq_len:
        :param requirements: the required modalities name, default: ['macroImage', 'frictionForce']
        In the *.mat, we can also get: 'labelCoarse', 'labelMedium', 'labelFine', 'materialNameEnglish', 'density',
                                       'subclass', 'class', 'material'， 'instance', # Class part
                                       'displayImage', 'temperatureData', 'metalData', 'reflectance',
                                       'accelDFT', 'accelDFTSteelTooltip', 'sound', 'soundSteelTooltip', 'accelTap',
                                       'accelTapSteelTooltip', 'soundTap', 'soundTapSteelTooltip', 'pressure',
                                       'pressureFolding', 'foldingAngle', 'normalForce', 'frictionForce', 'materialID',
                                       'illuminatedImage', 'labelFineP', 'labelFinePX', 'processing', 'extra',
                                       'magneticData', 'pressureRaw', 'pressureFoldingRaw', 'foldingAngleRaw',
                                       'ambientTemperature'
        """
        self.data_path = []
        self.data_dirs = []
        self.requirements = cfg.MODALITY.REQUIRMENTS
        self.cfg = cfg
        self.spectrogram = spectrogram
        self.mel_spectrogram = mel_spectrogram
        self.raw_data = raw_data
        # self.seq_len = seq_len

        for root, dirs, files in os.walk(data_root):
            for f in files:
                if f.endswith(".mat"):
                    self.data_path.append(os.path.join(root, f))
                    if root in self.data_dirs:
                        continue
                    else:
                        self.data_dirs.append(root)

        classes, class_to_idx = self._find_classes(self.data_dirs)
        assert len(classes) == cfg.MODEL.NUM_CLASSES

        self.data_path = cleanup_data(self.data_path)
        self.data_path.sort()
        if len(self.data_path) == 0:
            raise FileNotFoundError("No *.mat files in {:}".format(data_root))

        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        # Load data
        try:
            data_mat = loadmat(self.data_path[item])
        except TypeError:
            print(self.data_path[item])
        except ValueError:
            print(self.data_path[item])

        data_mat = data_mat['finalMaterialRecording']
        data_mat = data_mat[0]
        data_mat = data_mat[0]
        # Get label
        class_name = self._label_process(data_mat['materialID'][0])
        class_idx = self.class_to_idx[class_name]
        # class_idx = torch.LongTensor(class_idx)
        # Multi-modality, save the required modalities in a list with the order of self.requirements
        modalities = []
        # print(self.data_path[item], )
        for m in self.requirements:
            if m in CBM_DATASET_ATTR[:4]:
                continue
            try:
                modality = modality_preprocessor(m,
                                                 data_mat[m],
                                                 self.cfg,
                                                 self.spectrogram,
                                                 self.mel_spectrogram,
                                                 self.raw_data,
                                                 self.data_path[item])
            except ValueError:
                print(self.data_path[item], m)
                assert False
            if "Image" in m:
                if modality.shape[0] == 1:
                    modality = torch.stack([modality]*3)
                if len(modality.shape) == 4:
                    print(self.data_path[item])
                    modality = modality.unsqueeze(dim=1)
            modalities.append(modality)

        modalities.append(torch.tensor(class_idx))

        return modalities

    def modality_visualize(self, item, name_modality):
        modality = loadmat(self.data_dirs[item])['finalMaterialRecording'][0][0][name_modality]
        modality_visualize(name_modality, modality)

    def _label_process(self, materialID):
        if self.cfg.MODALITY.LABEL_LEVEL == 'class':
            label_split = materialID.split("_")[0]
        elif self.cfg.MODALITY.LABEL_LEVEL == "class-novel":
            label_split = materialID.split("_")[0]
        elif self.cfg.MODALITY.LABEL_LEVEL == 'subclass':
            label_split = '_'.join(materialID.split("_")[:2])
        elif self.cfg.MODALITY.LABEL_LEVEL == "subclass-novel":
            label_split = '_'.join(materialID.split("_")[:2])

        elif self.cfg.MODALITY.LABEL_LEVEL == 'material':
            label_split = '_'.join(materialID.split("_")[:3])
        else:
            raise ValueError
        # label = '_'.join(label_split)

        return label_split

    def _find_classes(self, dirs):
        """

        :param dirs: A list of directory path, like ROOT/C1/S1/M1
        :return:
        """
        # classes = ['_'.join(d.split("/")[-3:]) for d in dirs]
        if self.cfg.MODALITY.LABEL_LEVEL == 'class':
            classes = [d.split("/")[-3] for d in dirs]
        elif self.cfg.MODALITY.LABEL_LEVEL == "class-novel":
            classes = [d.split("/")[-3] for d in dirs]
        elif self.cfg.MODALITY.LABEL_LEVEL == 'subclass':
            classes = ['_'.join(d.split("/")[-3:-1]) for d in dirs]
        elif self.cfg.MODALITY.LABEL_LEVEL == "subclass-novel":
            classes = ['_'.join(d.split("/")[-3:-1]) for d in dirs]
        elif self.cfg.MODALITY.LABEL_LEVEL == 'material':
            classes = ['_'.join(d.split("/")[-3:]) for d in dirs]
        else:
            raise ValueError
        classes = list(set(classes))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def CBM_collate_fn(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        output = []
        for i in range(self.cfg.MODALITY.NUMS):
            tmp_batch = [elem[i] for elem in batch]
            try:
                tmp_batch = torch.stack(tmp_batch)
            except RuntimeError:
                for elem in batch:
                    print(i, elem[i].shape)
            output.append(tmp_batch)
        target = [elem[-1] for elem in batch]
        target = torch.stack(target)

        return output, target


class CBMDatasetPickle(Dataset):
    """
    For .data format(Python Pickle)
    """
    def __init__(self, pickle_path):
        f = open(pickle_path, 'rb')
        self.data = pickle.load(f)
        self.cfg = self.data['cfg']

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data.keys()) - 1 # 1 for cfg

    def CBM_collate_fn(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        output = []
        for i in range(self.cfg.MODALITY.NUMS):
            tmp_batch = [elem[i] for elem in batch]
            try:
                tmp_batch = torch.stack(tmp_batch)
            except RuntimeError:
                for elem in batch:
                    print(i, elem[i].shape)
            output.append(tmp_batch)
        target = [elem[-1] for elem in batch]
        target = torch.stack(target)

        return output, target