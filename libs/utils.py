import os
import math
import itertools

from .datasets.CBM_dataset import CBMDataset, CBMDatasetPickle, CBM_DATASET_ATTR, \
    plot_spectrogram, create_spectrogram
from .logger import Logger

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn
from collections.abc import Iterable
from multiprocessing import Process, Manager
from torch.utils.data import Subset


def requirements_filter(requirements):
    """
    To remove the class part in the requirements.
    :param requirements:
    :return:
    """
    r = [m for m in requirements if m not in CBM_DATASET_ATTR[:4]]
    return r


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks_with_desired_nums(l, n):
    """Return successive n children chunks from l."""
    children_ = [[] for _ in range(n)]
    for i, e in enumerate(l):
        children_[i%n].append(e)
    # children_ = np.array([np.array(c) for c in children_])

    return children_


def cal_model_params_num(params):
    total = sum([param.nelement() for param in params])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total


def cal_model_parm_flops(model, input):

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    return out, total_flops / 1e9


def init_model(model, method="xavier"):
    """

    :param model: nn.Module
    :param method: xavier, kaiming or ortho
    :return: model
    """
    if method == "xavier":
        return _xavier_init(model)
    elif method == "kaiming":
        return _kaiming_init(model)
    elif method == "ortho":
        return _ortho_init(model)
    elif method is None:
        return model
    else:
        raise NotImplementedError("method should be one of xavier, kaiming or ortho")


def _xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return model


def _kaiming_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
    return model


def _ortho_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal(m.weight)
    return model


def data_list_to_dict(data, cfg):
    return dict(zip(cfg.MODALITY.REQUIRMENTS, data))


def _datapath_split_by_method(data_path, method="last_1"):
    """
    To split dataset by desired way.
    :param data_path: A list includes all valid data path
    :param method: define how to split the dataset.
        Can be "last_1", "last_2", "last_3", "first_1", "first_2"
    :return: train_data_path, test_data_path
    """
    dirs = [os.path.split(d)[0] for d in data_path]
    dirs = list(set(dirs))

    train_data_path, test_data_path = [], []
    for d in dirs:
        file_num = len(os.listdir(d))
        for f in os.listdir(d):
            if 'tmp' in f:
                file_num -= 1
        if file_num == 5:
            files_path = [f for f in data_path if d+'/' in f]
            files_path.sort()
            if method == 'last_1':
                train_data_path += files_path[:-1]
                test_data_path.append(files_path[-1])
            elif method == 'last_2':
                train_data_path += files_path[:-2]
                test_data_path += files_path[-2:]
            elif method == 'last_3':
                train_data_path += files_path[:-3]
                test_data_path += files_path[-3:]
            elif method == 'first_1':
                train_data_path += files_path[1:]
                test_data_path.append(files_path[0])
            elif method == 'first_2':
                train_data_path += files_path[2:]
                test_data_path += files_path[:2]
            else:
                raise NotImplementedError('the params method should be one of '
                                          '"last_1", "last_2", "last_3", "first_1", "first_2"')
        else:
            print(file_num, d)

    train_data_path.sort()
    test_data_path.sort()

    return train_data_path, test_data_path


def _dataset_split_by_method(dataset, method='last_1', desired_idx=None, orignal_datasets=None, log=None):
    """

    :param dataset: CBM_dataset or CBMDatasetPickle class
    :param method: can be 'last_*' or 'first_*' means the last * or first * instance used to test.
                    Also can be 'novel' with desired_idx is not None, to return novel materials to test.
    :param desired_idx: for 'novel' method
    :param orignal_data_path: for 'novel' method and when dataset is CBMDatasetPickle, this param should be not None
    :return:
    """
    if method == 'novel':
        return _dataset_split_for_novel(dataset, desired_idx, orignal_datasets, log=log)

    pickle = isinstance(dataset, CBMDatasetPickle)
    if pickle:
        data_idx = [i for i in range(len(dataset))]
        data_ = np.array([idx for idx in chunks(data_idx, 5)])

        if method == 'last_1':
            train_idx = data_[:, :4]
            test_idx = data_[:, 4]

        elif method == 'last_2':
            train_idx = data_[:, :3]
            test_idx = data_[:, 3:]
        elif method == 'last_3':
            train_idx = data_[:, :2]
            test_idx = data_[:, 2:]
        elif method == 'first_1':
            train_idx = data_[:, 1:]
            test_idx = data_[:, 0]
        elif method == 'first_2':
            train_idx = data_[:, 2:]
            test_idx = data_[:, :2]
        else:
            raise NotImplementedError('the params method should be one of '
                                      '"last_1", "last_2", "last_3", "first_1", "first_2"')

        train_idx, test_idx = train_idx.flatten(), test_idx.flatten()
        train_dataset = Subset(dataset, train_idx.tolist())
        test_dataset = Subset(dataset, test_idx.tolist())

    else:
        assert isinstance(dataset, CBMDataset)
        train_dataset = Subset(dataset, [_ for _ in range(len(dataset))])
        test_dataset = Subset(dataset, [_ for _ in range(len(dataset))])

        train_datapath, test_datapath = _datapath_split_by_method(dataset.data_path, method)
        train_dataset.dataset.data_path, test_dataset.dataset.data_path = train_datapath, test_datapath
        train_dataset, test_dataset = train_dataset.dataset, test_dataset

    return train_dataset, test_dataset


def _dataset_split_for_novel(dataset, desired_idx=1, orignal_datasets=None, log=None):
    """

    :param dataset:
    :param desired_idx:
    :param orignal_data_path:
    :return:
    """
    pickle = isinstance(dataset, CBMDatasetPickle)
    if pickle:
        assert orignal_datasets is not None
        data_path = orignal_datasets.data_path
    else:
        assert isinstance(dataset, CBMDataset)
        data_path = dataset.data_path

    cfg = dataset.cfg
    try:
        classes = cfg.MODEL.CLASSES
    except AttributeError:
        if pickle:
            classes = orignal_datasets.classes
        else:
            classes = dataset.classes
    num_classes = cfg.MODEL.NUM_CLASSES
    class_by_class = {}
    for c in classes:
        class_by_class[c] = []

    for i, dp in enumerate(data_path):
        ddir, _ = os.path.split(dp)
        if num_classes == 33:
            dclass = '_'.join(ddir.split("/")[-3:-1])
        elif num_classes == 8:
            dclass = ddir.split("/")[-3]
        else:
            raise ValueError
        class_by_class[dclass].append(i)

    desired_idx = [desired_idx]
    undesired_idx = [_ for _ in range(5) if _ not in desired_idx]

    train_idx = []
    test_idx = []
    for k, v in class_by_class.items():
        tmp_train_idx = []
        new_v = [idx for idx in chunks(v, 5)]
        material_num_per_class = len(new_v)
        if material_num_per_class < 5:
            if log is None:
                print(k + ": " + "{:} and No Testing Data.".format(len(v)))
            else:
                assert isinstance(log, Logger)
                log.logger.info(k + ": " + "{:} and No Testing Data.".format(len(v)))
            tmp_train_idx = list_flatten(v)
            train_idx.extend(tmp_train_idx)
            continue

        # if k == "C1":
        #     if log is None:
        #         print(k + ": " + "{:} and No Testing Data.".format(len(v)))
        #     else:
        #         assert isinstance(log, Logger)
        #         log.logger.info(k + ": " + "{:} and No Testing Data.".format(len(v)))
        #     tmp_train_idx = list_flatten(v)
        #     train_idx.extend(tmp_train_idx)
        #     continue

        children_v = chunks_with_desired_nums(new_v, 5)

        for i in undesired_idx:
            tmp_train_idx.extend(children_v[i])
        tmp_train_idx = list_flatten(tmp_train_idx)
        tmp_test_idx = list_flatten(children_v[desired_idx[0]])

        train_idx.extend(tmp_train_idx)
        test_idx.extend(tmp_test_idx)
        if log is None:
            print(k + ": " + "{:} and {:} for Testing.".format(len(v), len(tmp_test_idx)))
        else:
            assert isinstance(log, Logger)
            log.logger.info(k + ": " + "{:} and {:} for Testing.".format(len(v), len(tmp_test_idx)))

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, test_dataset


def dataset_split(dataset, method=None, desired_idx=1, orignal_datasets=None, log=None):
    if method is not None:
        return _dataset_split_by_method(dataset, method, desired_idx, orignal_datasets, log)
    if isinstance(desired_idx, int):
        desired_idx = [desired_idx]

    undesired_idx = [_ for _ in range(5) if _ not in desired_idx]

    data_idx = [i for i in range(len(dataset))]
    data_ = np.array([idx for idx in chunks(data_idx, 5)])
    train_idx = data_[:, undesired_idx]
    test_idx = data_[:, desired_idx]
    train_idx, test_idx = train_idx.flatten(), test_idx.flatten()
    # pickle = isinstance(dataset, CBMDatasetPickle)
    train_dataset = Subset(dataset, train_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())

    return train_dataset, test_dataset


def to_pickle(dataset, pickle_file=None, desired_idx=None, corr_data=None):
    """

    :param dataset: torch.utils.data.Dataset
    :param pickle_file: str, file, or None
    :param desired_idx: if desired_idx is not None, only data in desired_idx will be dump in pickle_file.
    :param corr_data:
    :return:
    """
    if isinstance(pickle_file, str):
        pickle_file = open(pickle_file, 'wb')
    else:
        pass

    data = {}
    if desired_idx is None:
        for i in range(len(dataset)):
            print("Item {:} is processing!".format(i))
            if corr_data is not None:
                corr_data[i] = dataset.__getitem__(i)
            data[i] = dataset.__getitem__(i)
    else:
        assert isinstance(desired_idx, Iterable), \
            "desired_idx should be None or Iterable, but {:} is got".format(desired_idx)
        for i in desired_idx:
            print("Item {:} is processing!".format(i))
            if corr_data is not None:
                print(dataset.data_path[i])
                corr_data[i] = dataset.__getitem__(i)
            else:
                data[i] = dataset.__getitem__(i)
    try:
        if pickle_file is None:
            return data

        pickle.dump(data, pickle_file)
        pickle_file.close()

        return True

    except Exception:
        return False


def multiprocess_to_pickle(dataset, pickle_file=None, num_thread=5):
    idx = [i for i in range(len(dataset))]
    data_ = [Manager().dict() for _ in range(num_thread)]
    # for i, d in enumerate(chunks(idx, math.ceil(len(dataset)/num_thread))):
    #     print(i, d)
    pickle_process = [Process(target=to_pickle, args=(dataset, None, d, data_[i])) for i, d in enumerate(chunks(idx, math.ceil(len(dataset)/num_thread)))]

    for i in range(num_thread):
        print("Process {:} Start!".format(i))
        pickle_process[i].start()

    while(True):
        num = sum([len(n) for n in data_])
        if num >= len(dataset):
            break
    print("Total number of dataset {:}, and Load {:} from local.".format(len(dataset), num))
    data = {}
    for i in range(num_thread):
        print("Process {:} with nums of data {:}".format(i, len(data_[i])))
        data.update(data_[i])

    data['cfg'] = dataset.cfg
    data['cfg'].MODEL.CLASSES = dataset.classes

    if isinstance(pickle_file, str):
        pickle_file = open(pickle_file, 'wb')
    else:
        pass

    try:
        if pickle_file is None:
            return data

        data['cfg'].MODALITY.PICKLE_FILE = pickle_file

        pickle.dump(data, pickle_file)
        pickle_file.close()

        return True

    except Exception:
        return False


def compute_accuracy(predicts, targets):
    # _, predicted = torch.max(predicts.data, 1)
    # acc = (predicted == targets).sum().item()
    acc = (torch.argmax(predicts.data, 1) == targets).sum().item()
    acc_ratio = float(acc/len(targets))
    return acc, acc_ratio


def visualize_log(log_file, smooth_weight=None, lens=None, all_in_one=False, aux_loss=False):
    fig = plt.figure(2)

    if isinstance(log_file, str):
        fig_name = os.path.splitext(log_file)[0] + ".png"

        log_file = open(log_file, mode='r')
    else:
        fig_name = "visualize_log.png"

    train_record, test_record = read_log(log_file, aux_loss)
    if lens is None:
        lens = len(train_record)

    train_record, test_record = train_record[:lens, :], test_record[:lens, :]
    best_train_idx = np.argmax(train_record[:, -1])
    best_train = train_record[:, -1][best_train_idx]

    best_test_idx = np.argmax(test_record[:, 1])
    best_test = test_record[:, 1][best_test_idx]
    for i in range(len(train_record[:, -1])):
        if train_record[:, 2][i] > 0.95:
            print(i, train_record[:, -1][i])
            break
    for i in range(len(test_record[:, 1])):
        if test_record[:, 1][i] > 0.95:
            print(i, test_record[:, 1][i])
            break

    plt.subplot(311)
    plt.plot(train_record[:, 0], train_record[:, 1], label="loss")
    plt.legend()

    if aux_loss:
        plt.subplot(312)
        plt.plot(train_record[:, 0], train_record[:, 2], label="aux loss")
        plt.legend()

    plt.subplot(313)
    if isinstance(smooth_weight, float):
        smoothed_train_acc = smooth_log(train_record[:, -1], smooth_weight)
        smoothed_test_acc = smooth_log(test_record[:, 1], smooth_weight)
        plt.plot(train_record[:, 0], smoothed_train_acc, 'g-', label="smoothed train acc")
        plt.plot(test_record[:, 0], smoothed_test_acc, 'r-', label="smoothed test acc")

        best_smooth_test_idx = np.argmax(smoothed_test_acc)
        best_smooth_test = smoothed_test_acc[best_test_idx]
        print("Smoothed test acc: {:} {:}".format(best_smooth_test_idx, best_smooth_test))

    plt.plot(train_record[:, 0], train_record[:, -1], label="train acc", alpha=0.2)
    plt.plot(test_record[:, 0], test_record[:, 1], label="test acc", alpha=0.2)

    plt.plot(best_train_idx, best_train, "o", label="Best train")
    plt.plot(best_test_idx, best_test, "x", label="Best test")

    plt.legend()
    plt.savefig(fig_name)

    print('best train', best_train_idx, best_train)
    print('best test', best_test_idx, best_test)

    if all_in_one:
        pass
    else:
        plt.close()
    # plt.show()


def visualize_log_mtut(log_file, list_modality, lens=None, all_in_one=False, aux_loss=False):
    if isinstance(log_file, str):
        fig_name = os.path.splitext(log_file)[0] + ".png"

        log_file = open(log_file, mode='r')
    else:
        fig_name = "visualize_log.png"

    dict_train_record, dict_test_record = read_log_mtut(log_file, list_modality, aux_loss)

    for r in list_modality:
        if lens is None:
            lens = len(dict_train_record[r])
        dict_train_record[r], dict_test_record[r] = dict_train_record[r][:lens, :], dict_test_record[r][:lens, :]

    dict_best_train_idx = {}
    dict_best_train = {}
    dict_best_test_idx = {}
    dict_best_test = {}
    for r in list_modality:
        dict_best_train_idx[r] = np.argmax(dict_train_record[r][:, -1])
        dict_best_train[r] = dict_train_record[r][:, -1][dict_best_train_idx[r]]
        dict_best_test_idx[r] = np.argmax(dict_test_record[r][:, -1])
        dict_best_test[r] = dict_test_record[r][:, -1][dict_best_test_idx[r]]

    plt.subplot(411)
    for r in list_modality:
        label = r + "_total_loss"
        plt.plot(dict_train_record[r][:, 0], dict_train_record[r][:, 1], label=label)
    plt.legend()
    plt.subplot(412)

    for r in list_modality:
        label = r + "_ssa_loss"
        plt.plot(dict_train_record[r][:, 0], dict_train_record[r][:, 2], label=label)

    plt.legend()
    plt.subplot(413)
    if aux_loss:
        for r in list_modality:
            label = r + '_aux_loss'
            plt.plot(dict_train_record[r][:, 0], dict_train_record[r][:, 3], label=label)
    plt.legend()
    plt.subplot(414)
    for r in list_modality:
        label = r + "_train_acc"
        plt.plot(dict_train_record[r][:, 0], dict_train_record[r][:, -1], label=label, alpha=0.2)
        label = r + "_test_acc"
        plt.plot(dict_test_record[r][:, 0], dict_test_record[r][:, 1], label=label, alpha=0.2)
        label = r + "_best_train_acc"
        plt.plot(dict_best_train_idx[r], dict_best_train[r], "o", label=label)
        label = r + "_best_test_acc"
        plt.plot(dict_best_test_idx[r], dict_best_test[r], "x", label=label)

        print('best train ', r, dict_best_train_idx[r], dict_best_train[r])
        print('best test ', r, dict_best_test_idx[r], dict_best_test[r])

    plt.legend()
    plt.savefig(fig_name)

    if all_in_one:
        pass
    else:
        plt.close()


def read_log_mtut(f_log, list_modality, aux_loss=False):
    result = [line for line in f_log.readlines() if "EPOCH:" in line]
    dict_train_record = {}
    dict_test_record = {}
    for r in list_modality:
        dict_train_record[r] = []
        dict_test_record[r] = []

    for tmp in result:
        record = tmp.split(": ")[2:]
        one = []
        for r in record:
            if "}" in r:
                one.append(float(r.split("}")[0]))
            elif "," in r:
                one.append(float(r.split(", ")[0]))

        for i, r in enumerate(list_modality):
            if "TEST " in tmp:
                dict_test_record[r].append([one[0], one[i+1]])
            else:
                if aux_loss:
                    dict_train_record[r].append([one[0], one[i + 1], one[i + 3], one[i + 5], one[i + 7]])
                else:
                    dict_train_record[r].append([one[0], one[i+1], one[i+3], one[i+5]])

    for r in list_modality:
        dict_train_record[r] = np.array(dict_train_record[r])
        dict_test_record[r] = np.array(dict_test_record[r])

    return dict_train_record, dict_test_record


def read_log(f_log, aux_loss=False):
    result = [line for line in f_log.readlines() if "EPOCH:" in line]
    test_record = []
    train_record = []

    for tmp in result:
        record = tmp.split(": ")[2:]
        one = []
        for r in record:
            if ", " in r:
                one.append(float(r.split(", ")[0]))
            else:
                one.append(float(r))
        if "TEST " in tmp:
            test_record.append(one)
        else:
            train_record.append(one)

    test_record = np.array(test_record)
    train_record = np.array(train_record)

    return train_record, test_record


def smooth_log(log_record, weight=0.6):
    smoothed = []
    last = log_record[0]

    for point in log_record:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def split_pickle(pickle_data, m_dir=None, spectrogram=False, raw_data=False):
    modality_list = pickle_data['cfg'].MODALITY.REQUIRMENTS

    for i in range(len(modality_list)):
        print("Start split " + modality_list[i])
        m_dict = {}

        try:
            label_level = pickle_data['cfg'].MODALITY.LABEL_LEVEL
        except AttributeError:
            label_level = 'class'

        try:
            ts_noise_snr = pickle_data['cfg'].MODALITY.TS_NOISE_SNR
        except AttributeError:
            ts_noise_snr = -1

        if spectrogram:
            m_file = "CBMDataset_" + modality_list[i] + "_" + label_level + '_{:}'.format(ts_noise_snr) + "_spectrogram.data"
        else:
            m_file = "CBMDataset_" + modality_list[i] + "_" + label_level + '_{:}'.format(ts_noise_snr) + ".data"

        if raw_data:
            m_file = "CBMDataset_" + modality_list[i] + "_" + label_level + '_{:}'.format(ts_noise_snr) + "_raw.data"

        if m_dir is None:
            pass
        else:
            m_file = os.path.join(m_dir, m_file)
        f = open(m_file, 'wb')
        for k, v in pickle_data.items():
            if k == 'cfg':
                v.MODALITY.REQUIRMENTS = [modality_list[i]]
                v.MODALITY.PICKLE_FILE = m_file
                v.MODALITY.NUMS = 1
                m_dict[k] = v
            else:
                m_dict[k] = [v[i], v[-1]]
        pickle.dump(m_dict, f)
        f.close()


def concat_pickle(list_pickle_data, pickle_dir=None, spectrogram=False, log_mel=False, raw_data=False):
    result_pickle_data = list_pickle_data[0].copy()
    modality_list = []
    for i in range(len(list_pickle_data)):
        pickle_data = list_pickle_data[i]
        modality_list.extend(pickle_data['cfg'].MODALITY.REQUIRMENTS)
        if "Image" in pickle_data['cfg'].MODALITY.REQUIRMENTS:
            result_pickle_data['cfg'].MODALITY.IMAGE = pickle_data['cfg'].MODALITY.IMAGE

        if "sound" in pickle_data['cfg'].MODALITY.REQUIRMENTS:
            result_pickle_data['cfg'].MODALITY.SOUND = pickle_data['cfg'].MODALITY.SOUND

        if "accel" in pickle_data['cfg'].MODALITY.REQUIRMENTS:
            result_pickle_data['cfg'].MODALITY.ACCEL = pickle_data['cfg'].MODALITY.ACCEL

        if "Force" in pickle_data['cfg'].MODALITY.REQUIRMENTS:
            result_pickle_data['cfg'].MODALITY.FORCE = pickle_data['cfg'].MODALITY.FORCE

        if i == 0:
            continue
        else:
            for k, v in pickle_data.items():
                if k == 'cfg':
                    continue
                label = result_pickle_data[k][-1]
                tmp_list = result_pickle_data[k][:-1]
                tmp_list.extend(v[:-1])
                result_pickle_data[k] = tmp_list
                result_pickle_data[k].append(label)

    result_pickle_data['cfg'].MODALITY.NUMS = len(modality_list)
    result_pickle_data['cfg'].MODALITY.REQUIRMENTS = modality_list

    try:
        ts_noise_snr = result_pickle_data['cfg'].MODALITY.TS_NOISE_SNR
    except AttributeError:
        ts_noise_snr = -1

    try:
        label_level = result_pickle_data['cfg'].MODALITY.LABEL_LEVEL
    except AttributeError:
        label_level = 'class'

    if spectrogram:
        result_pickle_data['cfg'].MODALITY.SOUND.SPECTROGRAM = True
        result_pickle_data['cfg'].MODALITY.ACCEL.SPECTROGRAM = True
        result_pickle_data['cfg'].MODALITY.FORCE.SPECTROGRAM = True

        pickle_file = "CBMDataset_" + "_".join(modality_list) + "_" + label_level + '_{:}'.format(
            ts_noise_snr) + "_spectrogram.data"
    else:
        result_pickle_data['cfg'].MODALITY.SOUND.SPECTROGRAM = False
        result_pickle_data['cfg'].MODALITY.ACCEL.SPECTROGRAM = False
        result_pickle_data['cfg'].MODALITY.FORCE.SPECTROGRAM = False
        pickle_file = "CBMDataset_" + "_".join(modality_list) + "_" + label_level + '_{:}'.format(ts_noise_snr) + "_psd_svd.data"

    if raw_data:
        pickle_file = "CBMDataset_" + "_".join(modality_list) + "_" + label_level + '_{:}'.format(ts_noise_snr) + "_raw.data"
    if log_mel:
        pickle_file = "CBMDataset_" + "_".join(modality_list) + "_" + label_level + '_{:}'.format(ts_noise_snr) + "_logmel.data"
    if pickle_dir is None:
        pass
    else:
        pickle_file = os.path.join(pickle_dir, pickle_file)

    result_pickle_data['cfg'].MODALITY.PICKLE_FILE = pickle_file

    print(result_pickle_data['cfg'])

    f = open(pickle_file, 'wb')
    pickle.dump(result_pickle_data, f)
    f.close()


def DFT321(x, y, z):
    pass


def list_flatten(l):
    if isinstance(l[0], Iterable):
        result = []
        for n in l:
            result.extend(n)
        return list_flatten(result)
    else:
        return l


def log_filename(cfg, n=None, freeze=False):
    if cfg.FUSION_HEAD.METHOD == "Re-weighting Fusion":
        log_fn = os.path.join('log', 'UNTIL_CONVERGE_{:}'.format(cfg.MODEL.UNTIL_CONVERGE), "_".join(cfg.MODALITY.REQUIRMENTS), cfg.FUSION_HEAD.METHOD,
                              "CBMDataset_" + cfg.MODEL.RESNET_NAME
                              + '_' + cfg.MODEL.TS_NET + '_' + '_'.join(cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY)
                              + '_' + cfg.FUSION_HEAD.ACTIVATION + '_{:}'.format(cfg.FUSION_HEAD.FEATURE_DIMS)
                              + '_{:}'.format(cfg.MODEL.EXTRACTOR_FREEZE)
                              )
    elif cfg.FUSION_HEAD.METHOD == "MTUT":
        log_fn = os.path.join('log', 'UNTIL_CONVERGE_{:}'.format(cfg.MODEL.UNTIL_CONVERGE), "_".join(cfg.MODALITY.REQUIRMENTS), cfg.FUSION_HEAD.METHOD,
                              "CBMDataset_" + cfg.MODEL.RESNET_NAME
                              + '_' + cfg.MODEL.TS_NET + '_' + cfg.FUSION_HEAD.ACTIVATION
                              + '_{:}'.format(cfg.FUSION_HEAD.FEATURE_DIMS) + '_' + cfg.FUSION_HEAD.MTUT.AUXLOSS
                              + '_{:}'.format(cfg.MODEL.EXTRACTOR_FREEZE) + "_T{:}".format(cfg.FUSION_HEAD.MTUT.T))
    else:
        log_fn = os.path.join('log', 'Test', 'UNTIL_CONVERGE_{:}'.format(cfg.MODEL.UNTIL_CONVERGE), "_".join(cfg.MODALITY.REQUIRMENTS), cfg.FUSION_HEAD.METHOD,
                              "CBMDataset_" + cfg.MODEL.RESNET_NAME
                              + '_' + cfg.MODEL.TS_NET + '_' + cfg.FUSION_HEAD.ACTIVATION
                              + '_{:}'.format(cfg.FUSION_HEAD.FEATURE_DIMS) + '_{:}'.format(cfg.MODEL.EXTRACTOR_FREEZE))

    if cfg.MODEL.AUX_LOSS:
        log_fn += "_CenterLoss"

    if n is not None:
        log_fn += "_{}.log".format(n)
    else:
        log_fn += ".log"

    if freeze:
        log_fn = os.path.join("Freeze", log_fn)

    if not os.path.exists(os.path.split(log_fn)[0]):
        os.makedirs(os.path.split(log_fn)[0])

    return log_fn


def save_model(model, cfg, n=None, specific_modality=None, suffix=None, total_model=False):
    if specific_modality is None:
        specific_modality = cfg.MODALITY.REQUIRMENTS

    if cfg.FUSION_HEAD.METHOD == "Re-weighting Fusion":
        model_fn = os.path.join('model', 'UNTIL_CONVERGE_{:}'.format(cfg.MODEL.UNTIL_CONVERGE), '_'.join(cfg.MODALITY.REQUIRMENTS), cfg.FUSION_HEAD.METHOD,
                                "CBMDataset_" + "_".join(specific_modality) + '_' + cfg.MODEL.RESNET_NAME
                                + '_' + cfg.MODEL.TS_NET + '_' + '_'.join(cfg.FUSION_HEAD.REWEIGHTINGFUSION.ATTENTION_WAY)
                                + '_' + cfg.FUSION_HEAD.ACTIVATION + '_{:}'.format(cfg.FUSION_HEAD.FEATURE_DIMS)
                                )
    elif cfg.FUSION_HEAD.METHOD == "MTUT":
        model_fn = os.path.join('model', 'UNTIL_CONVERGE_{:}'.format(cfg.MODEL.UNTIL_CONVERGE), '_'.join(cfg.MODALITY.REQUIRMENTS), cfg.FUSION_HEAD.METHOD,
                                "CBMDataset_" + "_".join(specific_modality) + '_' + cfg.MODEL.RESNET_NAME
                                + '_' + cfg.MODEL.TS_NET + '_' + cfg.FUSION_HEAD.ACTIVATION
                                + '_{:}'.format(cfg.FUSION_HEAD.FEATURE_DIMS) + '_' + cfg.FUSION_HEAD.MTUT.AUXLOSS)
    else:
        model_fn = os.path.join('model', 'Compute', 'UNTIL_CONVERGE_{:}'.format(cfg.MODEL.UNTIL_CONVERGE), '_'.join(cfg.MODALITY.REQUIRMENTS), cfg.FUSION_HEAD.METHOD,
                                "CBMDataset_" + "_".join(specific_modality) + '_' + cfg.MODEL.RESNET_NAME
                                + '_' + cfg.MODEL.TS_NET + '_' + cfg.FUSION_HEAD.ACTIVATION
                                + '_{:}'.format(cfg.FUSION_HEAD.FEATURE_DIMS))

    if suffix is not None:
        model_fn = model_fn + "_" + suffix

    if n is not None:
        model_fn += "_{}.pkl".format(n)
    else:
        model_fn += ".pkl"

    if not os.path.exists(os.path.split(model_fn)[0]):
        os.makedirs(os.path.split(model_fn)[0])

    try:
        if total_model:
            torch.save(model, model_fn)
        else:
            torch.save(model.state_dict(), model_fn)
        return True
    except RuntimeError:
        return False


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def max_in_dict(dict_x):
    for key, value in dict_x.items():
        if (value == max(dict_x.values())):
            return key, value


def min_in_dict(dict_x):
    for key, value in dict_x.items():
        if (value == min(dict_x.values())):
            return key, value


def plot_confuse_matrix(confuse_matrix, classes, log_file_path):
    fig_name = os.path.splitext(log_file_path)[0] + "_confuse_matrix.png"

    plt.imshow(confuse_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confuse Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = confuse_matrix.max() / 2.
    for i, j in itertools.product(range(confuse_matrix.shape[0]), range(confuse_matrix.shape[1])):
        plt.text(j, i, format(confuse_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confuse_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(fig_name)
    plt.close()


def visualize_xela_sound(data, is_sound, save_dir, tacitle_type="normal"):
    if is_sound:
        _visualize_sound(data, save_dir)
    else:
        _visualize_xela(data, save_dir, tacitle_type)


def _visualize_sound(data, save_dir):
    # Settings
    clear_frames = False  # Should it clear the figure between each frame?
    fps = 15

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Sound', artist='WKOA')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(12.8, 7.2))

    fs = 44100
    s = 2048
    n_frames = len(data)//s

    with writer.saving(fig, save_dir, 100):
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        x, y = [], []
        # plt.title("sound")
        for i in range(n_frames):
            y += list(data[len(x): len(x)+s])
            x += [_/fs for _ in range(len(x), len(x) + s)]
            if clear_frames:
                fig.clear()

            ax, = plt.plot(x, y, 'b-', linestyle="solid", linewidth=1)
            writer.grab_frame()

    plt.close()

def _visualize_xela(data, save_dir, tactile_type):
    # Settings
    clear_frames = False  # Should it clear the figure between each frame?
    fps = 15

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tactile-' + tactile_type, artist='WKOA')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(12.8, 7.2))

    fs = 100
    s = 10
    n_frames = math.ceil(len(data) / s)

    with writer.saving(fig, save_dir, 100):

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        x, y = [], []
        for i in range(n_frames):

            y += list(data[len(x): len(x) + s])
            if len(x) + s > len(data):
                x += [_ / fs for _ in range(len(x), len(data))]
            else:
                x += [_ / fs for _ in range(len(x), len(x) + s)]

            if clear_frames:
                fig.clear()
            if tactile_type == "normal":
                ax, = plt.plot(x, y, '-', linestyle="solid", linewidth=1, color="brown")
            else:
                ax, = plt.plot(x, y, 'g-', linestyle="solid", linewidth=1)

            writer.grab_frame()
    plt.close()


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n, n])
    H = torch.eye(n) - h
    # H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components.float()