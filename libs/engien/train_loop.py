import time
import numpy as np

import torch

from libs.utils import data_list_to_dict, compute_accuracy, save_model, max_in_dict, min_in_dict
from ..modules.mtut_head import SSALoss, KLSSALoss, JSSSALoss, \
    AutoEncoderLoss, VAELoss, ACCSSALoss, ADSSALoss, BatchSSALoss, OUT_SSALoss, \
    AUX_SSALoss, AUX_OUT_SSALoss


class _BasicTrainer(object):
    def __init__(self, model, data_loader, loss_func, optimizer, cfg, ssa_loss=None, aux_loss=None, aux_optimizer=None):
        model.train()
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.ssa_loss = ssa_loss

        self.aux_loss = aux_loss
        self.aux_optimizer = aux_optimizer

        self.cuda = True if cfg.MODEL.DEVICE == "cuda" else False
        print("CUDA: ", self.cuda)
        if self.cuda:
            self.model = self.model.cuda()

        self.iter = 0
        self.cfg = cfg
        self.best_test_acc = 0

    def train(self):
        raise NotImplementedError

    def test(self, test_data_loader):
        raise NotImplementedError

    def run_step(self, data):
        raise NotImplementedError

    def save(self, n=None, specific_modality=None, suffix=None):
        save_model(self.model, self.cfg, n, specific_modality, suffix)

    def _detect_anomaly(self, losses):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!".format(
                    self.iter
                )
            )


class Trainer(object):
    def __init__(self, model, data_loader, loss_func, optimizer, cfg,
                 aux_loss=None, aux_optimizer=None):
        model.train()
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.aux_loss = aux_loss
        self.aux_optimizer = aux_optimizer

        self.cuda = True if cfg.MODEL.DEVICE == "cuda" else False
        print("CUDA: ", self.cuda)
        if self.cuda:
            self.model = self.model.cuda()
            if self.aux_loss is not None:
                self.aux_loss = self.aux_loss.cuda()

        self.iter = 0
        self.cfg = cfg
        self.best_test_acc = 0
        self.confuse_matrix = np.zeros((cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES))

    def train(self):
        total_losses, total_aux_losses = [], []
        total_acc, total_num = 0.0, 0.0
        for _, data in enumerate(self.data_loader):
            # with torch.autograd.detect_anomaly():
            #     losses, acc, nums = self.run_step(data)
            with torch.autograd.detect_anomaly():
                if self.aux_loss is not None:
                    losses, aux_loss, acc, nums = self.run_step(data)
                    total_aux_losses.append(aux_loss)
                else:
                    losses, acc, nums = self.run_step(data)
            # losses, acc, nums = self.run_step(data)
            total_acc += acc
            total_losses.append(losses)
            total_num += nums

        if self.aux_loss is not None:
            return np.mean(total_losses), np.mean(total_aux_losses), total_acc/total_num, total_num

        return np.mean(total_losses), total_acc/total_num, total_num

    def run_step(self, data):
        """
            Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        # data = next(self._data_loader_iter)
        label = data[1]
        data = data[0]
        x = data_list_to_dict(data, self.cfg)
        if self.cuda:
            label = label.cuda()
            for k, v in x.items():
                x[k] = v.cuda()
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        if self.cfg.MODEL.AUX_LOSS:
            out, feature, hid_feature = self.model(x, True)
            aux_losses = self.aux_loss(label, feature)[0]
        else:
            out = self.model(x)

        losses = self.loss_func(out, label)

        self._detect_anomaly(losses)

        if self.cfg.MODEL.AUX_LOSS:
            losses += self.cfg.MODEL.AUX_LOSS_GAMMA * aux_losses

        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.aux_loss is not None:
            self.aux_optimizer.zero_grad()

        self.optimizer.zero_grad()
        losses.backward()

        # cal training acc
        acc, acc_ratio = compute_accuracy(out, label)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
        if self.aux_loss is not None:
            self.aux_optimizer.step()
            print("Iter. {:}: Loss: {:}, Aux Loss: {:}, acc: {:}, nums: {:}".format(self.iter, float(losses.data),
                                                                                    self.cfg.MODEL.AUX_LOSS_GAMMA * float(aux_losses.data),
                                                                                    acc_ratio,
                                                                                    len(label)))

        else:
            print("Iter. {:}: Loss: {:}, acc: {:}, nums: {:}".format(self.iter, float(losses.data), acc_ratio,
                                                                     len(label)))

        self.iter += 1
        if self.aux_loss is not None:
            return float(losses.data), self.cfg.MODEL.AUX_LOSS_GAMMA * float(aux_losses.data), acc, len(label)

        return float(losses.data), acc, len(label)

    def _detect_anomaly(self, losses):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!".format(
                    self.iter
                )
            )

    def test(self, test_data_loader):
        training_mode = self.model.training
        self.model.eval()
        confuse_matrix = np.zeros((self.cfg.MODEL.NUM_CLASSES, self.cfg.MODEL.NUM_CLASSES))
        with torch.no_grad():
            total_acc, total = 0, 0
            for i, data in enumerate(test_data_loader):
                label = data[1]
                data = data[0]
                x = data_list_to_dict(data, self.cfg)

                if self.cuda:
                    label = label.cuda()
                    for k, v in x.items():
                        x[k] = v.cuda()
                out = self.model(x)
                acc, _ = compute_accuracy(out, label)
                preds = torch.argmax(out.data, 1)
                for t, p in zip(label.view(-1), preds.view(-1)):
                    confuse_matrix[t.cpu().long(), p.cpu().long()] += 1
                total_acc += acc
                total += len(label)

            # print("Total correct predicted data is {:} and {:}% in {:}".format(total_acc, float(total_acc/total), total))
        self.model.train(mode=training_mode)

        if total_acc > self.best_test_acc:
            self.best_test_acc = float(total_acc/total)
            self.confuse_matrix = confuse_matrix.astype('float')

        return float(total_acc/total), confuse_matrix


class TrainerMTUT(object):
    def __init__(self, model, data_loader, loss_func, optimizer, cfg, aux_loss=None, aux_optimizer=None):
        for r in cfg.MODALITY.REQUIRMENTS:
            model[r].train()
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.aux_loss = aux_loss
        self.aux_optimizer = aux_optimizer
        self.optimizer = optimizer

        if cfg.FUSION_HEAD.MTUT.AUXLOSS == "SSALoss":
            self.ssa_loss = SSALoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "JSDSSALoss":
            self.ssa_loss = JSSSALoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "KLSSALoss":
            self.ssa_loss = KLSSALoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "AELoss":
            self.ssa_loss = AutoEncoderLoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "VAELoss":
            self.ssa_loss = VAELoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "ACCSSALoss":
            self.ssa_loss = ACCSSALoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "ADSSALoss":
            self.ssa_loss = ADSSALoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "BatchSSALoss":
            self.ssa_loss = BatchSSALoss(cfg)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "AUX_SSALoss":
            self.ssa_loss = AUX_SSALoss(cfg.FUSION_HEAD.FEATURE_DIMS,
                                        cfg.MODALITY.REQUIRMENTS,
                                        cfg.FUSION_HEAD.MTUT.REG_METHOD,
                                        cfg.FUSION_HEAD.MTUT.BETA,
                                        cfg.FUSION_HEAD.MTUT.NORM,
                                        cfg.FUSION_HEAD.MTUT.THRESOLD)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "AUX_OUT_SSALoss":
            self.ssa_loss = AUX_OUT_SSALoss(cfg.MODALITY.REQUIRMENTS,
                                            cfg.FUSION_HEAD.MTUT.REG_METHOD,
                                            cfg.FUSION_HEAD.MTUT.BETA,
                                            cfg.FUSION_HEAD.MTUT.THRESOLD,
                                            cfg.FUSION_HEAD.MTUT.T)
        elif cfg.FUSION_HEAD.MTUT.AUXLOSS == "OUT_SSALoss":
            self.ssa_loss = OUT_SSALoss(cfg.MODALITY.REQUIRMENTS,
                                        cfg.FUSION_HEAD.MTUT.REG_METHOD,
                                        cfg.FUSION_HEAD.MTUT.LAMDA,
                                        cfg.FUSION_HEAD.MTUT.BETA,
                                        cfg.FUSION_HEAD.MTUT.THRESOLD,
                                        cfg.FUSION_HEAD.MTUT.T)
        self.cuda = True if cfg.MODEL.DEVICE == "cuda" else False
        print("CUDA: ", self.cuda)
        if self.cuda:
            for r in cfg.MODALITY.REQUIRMENTS:
                self.model[r] = self.model[r].cuda()
            self.ssa_loss = self.ssa_loss.cuda()
        self.iter = 0
        self.cfg = cfg

    def train(self):
        total_losses, ssa_losses, aux_losses, total_acc = {}, {}, {}, {}
        for r in self.cfg.MODALITY.REQUIRMENTS:
            total_losses[r] = []
            ssa_losses[r] = []
            aux_losses[r] = []
            total_acc[r] = 0

        total_num = 0.0
        for _, data in enumerate(self.data_loader):
            with torch.autograd.detect_anomaly():
                if self.aux_loss is not None:
                    losses, acc, nums, aux_loss = self.run_step(data)
                else:
                    losses, acc, nums = self.run_step(data)

            for r in self.cfg.MODALITY.REQUIRMENTS:
                total_losses[r].append(losses['total_loss'][r])
                ssa_losses[r].append(losses['ssaloss'][r])
                if self.aux_loss is not None:
                    aux_losses[r].append(aux_loss[r])
                total_acc[r] += acc[r]
            total_num += nums

        for r in self.cfg.MODALITY.REQUIRMENTS:
            total_losses[r] = np.mean(total_losses[r])
            ssa_losses[r] = np.mean(ssa_losses[r])
            aux_losses[r] = np.mean(aux_losses[r])
            total_acc[r] /= total_num
        if self.aux_loss is not None:
            return total_losses, ssa_losses, total_acc, total_num, aux_losses

        return total_losses, ssa_losses, total_acc, total_num

    def train_with_teacher(self, teacher_modality):
        total_losses, ssa_losses, aux_losses, total_acc = {}, {}, {}, {}
        for r in self.cfg.MODALITY.REQUIRMENTS:
            total_losses[r] = []
            ssa_losses[r] = []
            aux_losses[r] = []
            total_acc[r] = 0

        total_num = 0.0
        for _, data in enumerate(self.data_loader):
            with torch.autograd.detect_anomaly():
                if self.aux_loss is not None:
                    losses, acc, nums, aux_loss = self.run_step_teacher(data, teacher_modality)
                else:
                    losses, acc, nums = self.run_step_teacher(data, teacher_modality)

            for r in self.cfg.MODALITY.REQUIRMENTS:
                total_losses[r].append(losses['total_loss'][r])
                ssa_losses[r].append(losses['ssaloss'][r])
                if self.aux_loss is not None:
                    aux_losses[r].append(aux_loss[r])
                total_acc[r] += acc[r]
            total_num += nums

        for r in self.cfg.MODALITY.REQUIRMENTS:
            total_losses[r] = np.mean(total_losses[r])
            ssa_losses[r] = np.mean(ssa_losses[r])
            aux_losses[r] = np.mean(aux_losses[r])
            total_acc[r] /= total_num

        if self.aux_loss is not None:
            return total_losses, ssa_losses, total_acc, total_num, aux_losses

        return total_losses, ssa_losses, total_acc, total_num

    def run_step_teacher(self, data, teacher_m):
        for r in self.cfg.MODALITY.REQUIRMENTS:
            if r == teacher_m:
                pass
            else:
                assert self.model[r].training, "[SimpleTrainer] model was changed to eval mode!"

        label = data[1]
        data = data[0]
        x = data_list_to_dict(data, self.cfg)
        if self.cuda:
            label = label.cuda()
            for k, v in x.items():
                x[k] = v.cuda()

        """
            If you want to do something with the losses, you can wrap the model.
        """
        out, feature, hid_feature, dict_loss, aux_losses = {}, {}, {}, {}, {}
        dict_acc, dict_acc_ratio, dict_fake_ssa_loss = {}, {}, {}

        for r in self.cfg.MODALITY.REQUIRMENTS:
            out[r], feature[r], hid_feature[r] = self.model[r]({r: x[r]})
            dict_loss[r] = self.loss_func(out[r], label)

            self._detect_anomaly(dict_loss[r])

            aux_losses[r] = 0.0
            if self.aux_loss is not None:
                if self.cfg.FUSION_HEAD.MTUT.LAST_FEATURE:
                    aux_losses[r] = self.aux_loss[r](label, hid_feature[r])[0]
                else:
                    aux_losses[r] = self.aux_loss[r](label, feature[r])[0]
                dict_loss[r] += self.cfg.MODEL.AUX_LOSS_GAMMA * aux_losses[r]
            # cal training acc
            acc, acc_ratio = compute_accuracy(out[r], label)
            dict_acc[r] = acc
            dict_acc_ratio[r] = acc_ratio
            dict_fake_ssa_loss[r] = 0.0

        total_losses = self._compute_loss(dict_loss, aux_losses, feature, hid_feature, out, dict_acc)
        """
            If you need to accumulate gradients or something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
        """
        for r in self.cfg.MODALITY.REQUIRMENTS:
            self.optimizer[r].zero_grad()
            if self.aux_loss is not None:
                self.aux_optimizer[r].zero_grad()

        for i, r in enumerate(self.cfg.MODALITY.REQUIRMENTS):
            if i >= len(self.cfg.MODALITY.REQUIRMENTS) - 1:
                total_losses['total_loss'][r].backward()
            else:
                total_losses['total_loss'][r].backward(retain_graph=True)

        # losses.backward()
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        for r in self.cfg.MODALITY.REQUIRMENTS:
            if r == teacher_m:
                pass
            else:
                self.optimizer[r].step()
                if self.aux_loss is not None:
                    self.aux_optimizer[r].step()
            # self.optimizer[r].step()
            if self.aux_loss is not None:
                # self.aux_optimizer[r].step()
                print(
                    "Iter. {:}: Modality Name: {:}, Loss: {:}, Aux_loss: {:}, acc: {:}, nums: {:}".format(
                        self.iter, r, float(
                            total_losses['total_loss'][r].data), float(aux_losses[r].data), dict_acc_ratio[r],
                        len(label)))
            else:
                print(
                    "Iter. {:}: Modality Name: {:}, Loss: {:}, acc: {:}, nums: {:}".format(
                        self.iter, r, float(
                            total_losses['total_loss'][r].data), dict_acc_ratio[r], len(label)))

            total_losses['total_loss'][r] = float(total_losses['total_loss'][r].data)
            try:
                total_losses['ssaloss'][r] = self.cfg.FUSION_HEAD.MTUT.LAMDA * float(total_losses['ssaloss'][r].data)
            except AttributeError:
                total_losses['ssaloss'][r] = self.cfg.FUSION_HEAD.MTUT.LAMDA * float(total_losses['ssaloss'][r])
            try:
                aux_losses[r] = self.cfg.MODEL.AUX_LOSS_GAMMA * float(aux_losses[r].data)
            except AttributeError:
                aux_losses[r] = self.cfg.MODEL.AUX_LOSS_GAMMA * float(aux_losses[r])

        self.iter += 1
        if self.aux_loss is not None:
            return total_losses, dict_acc, len(label), aux_losses

        return total_losses, dict_acc, len(label)

    def run_step(self, data):
        """
            Implement the standard training logic described above.
        """
        for r in self.cfg.MODALITY.REQUIRMENTS:
            assert self.model[r].training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        # data = next(self._data_loader_iter)
        label = data[1]
        data = data[0]
        x = data_list_to_dict(data, self.cfg)
        if self.cuda:
            label = label.cuda()
            for k, v in x.items():
                x[k] = v.cuda()
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        out, feature, hid_feature, dict_loss, aux_losses = {}, {}, {}, {}, {}
        dict_acc, dict_acc_ratio,  dict_fake_ssa_loss = {}, {}, {}

        for r in self.cfg.MODALITY.REQUIRMENTS:
            out[r], feature[r], hid_feature[r] = self.model[r]({r: x[r]})
            dict_loss[r] = self.loss_func(out[r], label)

            self._detect_anomaly(dict_loss[r])

            aux_losses[r] = 0.0
            if self.aux_loss is not None:
                if self.cfg.FUSION_HEAD.MTUT.LAST_FEATURE:
                    aux_losses[r] = self.aux_loss[r](label, hid_feature[r])[0]
                else:
                    aux_losses[r] = self.aux_loss[r](label, feature[r])[0]
                dict_loss[r] += self.cfg.MODEL.AUX_LOSS_GAMMA * aux_losses[r]
            # cal training acc
            acc, acc_ratio = compute_accuracy(out[r], label)
            dict_acc[r] = acc
            dict_acc_ratio[r] = acc_ratio
            dict_fake_ssa_loss[r] = 0.0

        converge_flag = None
        if self.cfg.MODEL.UNTIL_CONVERGE:
            converge_flag = 1
            for r in self.cfg.MODALITY.REQUIRMENTS:
                if dict_acc_ratio[r] >= self.cfg.MODEL.UNTIL_CONVERGE_THRESHOLD:
                    pass
                else:
                    converge_flag = 0
                    break

            if converge_flag == 1:
                total_losses = self._compute_loss(dict_loss, aux_losses, feature, hid_feature, out, dict_acc)
            else:
                total_losses = {'ssaloss': dict_fake_ssa_loss, 'total_loss': dict_loss}
        else:
            total_losses = self._compute_loss(dict_loss, aux_losses, feature, hid_feature, out, dict_acc)
                # total_losses = self.ssa_loss(dict_loss, feature)

        # total_losses = dict_loss

        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        for r in self.cfg.MODALITY.REQUIRMENTS:
            self.optimizer[r].zero_grad()
            if self.aux_loss is not None:
                self.aux_optimizer[r].zero_grad()

        for i, r in enumerate(self.cfg.MODALITY.REQUIRMENTS):
            if i >= len(self.cfg.MODALITY.REQUIRMENTS)-1:
                total_losses['total_loss'][r].backward()
            else:
                total_losses['total_loss'][r].backward(retain_graph=True)

        # losses.backward()
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        for r in self.cfg.MODALITY.REQUIRMENTS:
            self.optimizer[r].step()
            if self.aux_loss is not None:
                self.aux_optimizer[r].step()
                print(
                    "Iter. {:}: Modality Name: {:}, Loss: {:}, Aux_loss: {:}, acc: {:}, nums: {:}, convergence: {:}".format(
                        self.iter, r, float(
                            total_losses['total_loss'][r].data), float(aux_losses[r].data), dict_acc_ratio[r], len(label),
                            converge_flag))
            else:
                print(
                    "Iter. {:}: Modality Name: {:}, Loss: {:}, acc: {:}, nums: {:}, convergence: {:}".format(
                        self.iter, r, float(
                            total_losses['total_loss'][r].data), dict_acc_ratio[r], len(label),
                        converge_flag))

            total_losses['total_loss'][r] = float(total_losses['total_loss'][r].data)
            try:
                total_losses['ssaloss'][r] = self.cfg.FUSION_HEAD.MTUT.LAMDA * float(total_losses['ssaloss'][r].data)
            except AttributeError:
                total_losses['ssaloss'][r] = self.cfg.FUSION_HEAD.MTUT.LAMDA * float(total_losses['ssaloss'][r])
            try:
                aux_losses[r] = self.cfg.MODEL.AUX_LOSS_GAMMA * float(aux_losses[r].data)
            except AttributeError:
                aux_losses[r] = self.cfg.MODEL.AUX_LOSS_GAMMA * float(aux_losses[r])

        self.iter += 1
        if self.aux_loss is not None:
            return total_losses, dict_acc, len(label), aux_losses

        return total_losses, dict_acc, len(label)

    def _detect_anomaly(self, losses):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!".format(
                    self.iter
                )
            )

    def _compute_loss(self, dict_loss, dict_aux_loss, dict_feature, dict_hid_feature, dict_out, dict_acc):
        if self.cfg.FUSION_HEAD.MTUT.AUXLOSS == "ACCSSALoss":
            total_losses = self.ssa_loss(dict_loss, dict_feature, dict_acc)
        elif "AUX_" in self.cfg.FUSION_HEAD.MTUT.AUXLOSS:
            if "OUT_" in self.cfg.FUSION_HEAD.MTUT.AUXLOSS:
                dict_ssa_loss = self.ssa_loss(dict_aux_loss, dict_out)
            else:
                if self.cfg.FUSION_HEAD.MTUT.LAST_FEATURE:
                    dict_ssa_loss = self.ssa_loss(dict_aux_loss, dict_hid_feature)
                else:
                    dict_ssa_loss = self.ssa_loss(dict_aux_loss, dict_feature)
            for r in self.cfg.MODALITY.REQUIRMENTS:
                dict_loss[r] += self.cfg.FUSION_HEAD.MTUT.LAMDA * dict_ssa_loss[r]
            total_losses = {'ssaloss': dict_ssa_loss, 'total_loss': dict_loss}
        else:
            if "OUT_" in self.cfg.FUSION_HEAD.MTUT.AUXLOSS:
                # TODO: Need check
                total_losses = self.ssa_loss(dict_loss, dict_out)
            else:
                if self.cfg.FUSION_HEAD.MTUT.LAST_FEATURE:
                    total_losses = self.ssa_loss(dict_loss, dict_hid_feature)
                else:
                    total_losses = self.ssa_loss(dict_loss, dict_feature)
            # total_losses = self.ssa_loss(dict_loss, feature)

        return total_losses

    def test(self, test_data_loader):
        for r in self.cfg.MODALITY.REQUIRMENTS:
            training_mode = self.model[r].training
            self.model[r].eval()
        with torch.no_grad():
            total = 0
            dict_acc = {}
            for r in self.cfg.MODALITY.REQUIRMENTS:
                dict_acc[r] = 0

            for i, data in enumerate(test_data_loader):
                label = data[1]
                data = data[0]
                x = data_list_to_dict(data, self.cfg)

                if self.cuda:
                    label = label.cuda()
                    for k, v in x.items():
                        x[k] = v.cuda()

                out, feature = {}, {}
                for r in self.cfg.MODALITY.REQUIRMENTS:
                    out[r], _, _ = self.model[r]({r: x[r]})

                # cal training acc
                for r in self.cfg.MODALITY.REQUIRMENTS:
                    acc, acc_ratio = compute_accuracy(out[r], label)
                    dict_acc[r] += acc
                total += len(label)

            # print("Total correct predicted data is {:} and {:}% in {:}".format(total_acc, float(total_acc/total), total))
        for r in self.cfg.MODALITY.REQUIRMENTS:
            dict_acc[r] = float(dict_acc[r]/total)
            self.model[r].train(mode=training_mode)

        return dict_acc


class TrainerAUXLoss(object):
    def __init__(self):
        pass

    def train(self):
        pass