import math
import torch
import numpy as np
from torch import nn
from .early_fusion import EarlyFusion
from torch.nn import functional as F
from .autoencoder import NaiveAutoEncoder, NaiveVAE
from .center_loss import CenterLoss


class MTUTHead(nn.Module):
    def __init__(self, cfg):
        super(MTUTHead, self).__init__()
        self.cfg = cfg
        cfg.MODEL.FEATURE_FUSION = "add"
        self.classifies = nn.ModuleDict()
        for r in cfg.MODALITY.REQUIRMENTS:
            self.classifies[r] = EarlyFusion(cfg)

    def forward(self, x):
        out = {}
        for k, v in x.items():
            out[k] = self.classifies[k](v)

        return out


class AUX_SSALoss(nn.Module):
    def __init__(self, feature_dim, requirments, reg_method='log', beta=0.5, norm=None, threshold=1e-12):
        super(AUX_SSALoss, self).__init__()
        self.beta = beta
        self.reg_method = reg_method
        self.requirments = requirments
        self.threshold = threshold
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, dict_loss, dict_x):
        dict_corr = {}
        for r in self.requirments:
            dict_x[r] = self.norm(dict_x[r])
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.bmm(dict_x[r].unsqueeze(2), dict_x[r].unsqueeze(1))

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m + "_" + n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = torch.sqrt(1e-12 + torch.sum(torch.sub(dict_corr[m], dict_corr[n]) ** 2))
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]

        return dict_ssa_loss


class AUX_OUT_SSALoss(nn.Module):
    def __init__(self, requirments, reg_method='log', beta=0.5, threshold=1e-12, T=5):
        super(AUX_OUT_SSALoss, self).__init__()
        self.beta = beta
        self.reg_method = reg_method
        self.requirments = requirments
        self.threshold = threshold
        self.T = T

    def forward(self, dict_loss, dict_out):
        dict_soft_softmax = {}
        for r in self.requirments:
            dict_soft_softmax[r] = F.softmax(dict_out[r]/self.T, dim=1)

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m + "_" + n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = F.kl_div(F.log_softmax(dict_out[m] / self.T, dim=1),
                                             F.softmax(dict_out[n] / self.T, dim=1)) * (self.T * self.T)
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]
        return dict_ssa_loss


class OUT_SSALoss(nn.Module):
    def __init__(self, requirments, reg_method='log', lamda=0.5, beta=2, threshold=1e-12, T=5):
        super(OUT_SSALoss, self).__init__()
        self.lamda = lamda
        self.beta = beta
        self.reg_method = reg_method
        self.requirments = requirments
        self.threshold = threshold
        self.T = T

    def forward(self, dict_loss, dict_out):
        dict_soft_softmax = {}
        for r in self.requirments:
            dict_soft_softmax[r] = F.softmax(dict_out[r]/self.T, dim=1)

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m + "_" + n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = F.kl_div(F.log_softmax(dict_out[m] / self.T, dim=1),
                                             F.softmax(dict_out[n] / self.T, dim=1)) * (self.T * self.T)
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda * dict_ssa_loss[r]
        # print(total_loss)
        return total_loss


class SSALoss(nn.Module):
    def __init__(self, cfg):
        super(SSALoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD
        self.norm = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.norm = self.norm.cuda()

    def forward(self, dict_loss, dict_x):
        dict_corr = {}
        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_x[r] = self.norm(dict_x[r])
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.bmm(dict_x[r].unsqueeze(2), dict_x[r].unsqueeze(1))
        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m+"_"+n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = torch.sqrt(1e-9 + torch.sum(torch.sub(dict_corr[m], dict_corr[n])**2))
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda*dict_ssa_loss[r]
        # print(total_loss)
        return total_loss


class BatchSSALoss(nn.Module):
    def __init__(self, cfg):
        super(BatchSSALoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD
        self.norm = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.norm = self.norm.cuda()

        self.cfg = cfg

    def forward(self, dict_loss, dict_x):
        dict_corr = {}
        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_x[r] = self.norm(dict_x[r])
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.matmul(dict_x[r] / (self.cfg.FUSION_HEAD.FEATURE_DIMS ** 0.5), dict_x[r].transpose(0, 1))
        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m+"_"+n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = torch.sqrt(1e-9 + torch.sum(torch.sub(dict_corr[m], dict_corr[n])**2))
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda*dict_ssa_loss[r]
        # print(total_loss)
        return total_loss


class ADSSALoss(nn.Module):
    def __init__(self, cfg):
        super(ADSSALoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD
        self.norm = nn.BatchNorm1d(cfg.FUSION_HEAD.FEATURE_DIMS)

        if cfg.MODEL.DEVICE == "cuda":
            self.norm = self.norm.cuda()

    def forward(self, dict_loss, dict_x):
        dict_corr = {}
        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_x[r] = self.norm(dict_x[r])
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.bmm(dict_x[r].unsqueeze(2), dict_x[r].unsqueeze(1))

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m+"_"+n
                dict_focal[i] = _regularizer(dict_loss[n], dict_loss[m], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = torch.sqrt(1e-9 + torch.sum(torch.sub(dict_corr[m], dict_corr[n])**2))
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda*dict_ssa_loss[r]
        # print(total_loss)
        return total_loss


class ACCSSALoss(nn.Module):
    def __init__(self, cfg):
        super(ACCSSALoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD

    def forward(self, dict_loss, dict_x, dict_acc):
        dict_corr = {}
        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.bmm(dict_x[r].unsqueeze(2), dict_x[r].unsqueeze(1))

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            for n in self.requirments:
                i = m+"_"+n
                dict_focal[i] = _regularizer(dict_acc[m], dict_acc[n], self.beta, self.threshold, self.reg_method)
                dict_corr_diff[i] = torch.sqrt(1e-12 + torch.sum(torch.sub(dict_corr[m], dict_corr[n])**2))
                dict_ssa_loss[m] += dict_focal[i] * dict_corr_diff[i]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda*dict_ssa_loss[r]
        # print(total_loss)
        return total_loss


class KLSSALoss(nn.Module):
    def __init__(self, cfg):
        super(KLSSALoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD
        self.klloss = nn.KLDivLoss()

    def forward(self, dict_loss, dict_x):
        dict_log_probs_x = {}
        dict_probs_x = {}

        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_probs_x[r] = F.softmax(dict_x[r], dim=1)
            dict_log_probs_x[r] = F.log_softmax(dict_x[r], dim=1)

        dict_focal = {}
        dict_kldiv = {}
        for m in self.requirments:
            dict_kldiv[m] = 0.0
            for n in self.requirments:
                i = m + "_" + n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_kldiv[m] += dict_focal[i] * self.klloss(dict_log_probs_x[m], dict_probs_x[n])
        total_loss = {'ssaloss': dict_kldiv, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda * dict_kldiv[r]
        # print(total_loss)
        return total_loss

    def regularizer(self, loss1, loss2, beta=2.0):
        if loss1 - loss2 > 0:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0


class JSSSALoss(nn.Module):
    def __init__(self, cfg):
        super(JSSSALoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD

    def forward(self, dict_loss, dict_x):
        dict_probs_x = {}
        for r in self.requirments:
            dict_probs_x[r] = F.softmax(dict_x[r], dim=1)

        dict_focal = {}
        dict_jsdiv = {}
        for m in self.requirments:
            dict_jsdiv[m] = 0.0
            for n in self.requirments:
                i = m + "_" + n
                dict_focal[i] = _regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method)
                dict_jsdiv[m] += dict_focal[i] * self.js_div(dict_probs_x[m], dict_probs_x[n])
        total_loss = {'ssaloss': dict_jsdiv, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda * dict_jsdiv[r]
        # print(total_loss)
        return total_loss

    def js_div(self, p, q, reduction="mean"):
        """
        To compute JS div.

        :param p:
        :param q:
        :param reduction:
        :return:
        """
        p_q_mean = ((p + q)/2).log()
        return (F.kl_div(p_q_mean, p, reduction=reduction) + F.kl_div(p_q_mean, q, reduction=reduction))/2

    def regularizer(self, loss1, loss2, beta=2.0):
        if loss1 - loss2 > 0:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0


class AutoEncoderLoss(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderLoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.lamda_ae = cfg.FUSION_HEAD.MTUT.LAMDA_AE
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD

        self.autoencoder = NaiveAutoEncoder(cfg.MODALITY.NUMS*cfg.FUSION_HEAD.FEATURE_DIMS,
                                            cfg.FUSION_HEAD.FEATURE_DIMS)

        self.autoencoderloss = nn.MSELoss()
        if cfg.MODEL.DEVICE == "cuda":
            self.autoencoder = self.autoencoder.cuda()
        self.ssa_loss = KLSSALoss(cfg)

    def forward(self, dict_loss, dict_x):
        concat_x = None
        for i, r in enumerate(self.requirments):
            if i == 0:
                concat_x = self.extractor[r](dict_x[r])
            else:
                tmpout = self.extractor[r](dict_x[r])
                concat_x = torch.cat([concat_x, tmpout], dim=1)

        decoder_out, emb = self.autoencoder(concat_x)
        autoencoder_loss = self.autoencoderloss(decoder_out, concat_x)

        emb = F.normalize(emb, p=2, dim=1, eps=1e-12)
        corr_emb = torch.bmm(emb.unsqueeze(2), emb.unsqueeze(1))

        dict_corr = {}
        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.bmm(dict_x[r].unsqueeze(2), dict_x[r].unsqueeze(1))

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            dict_corr_diff[m] = torch.sqrt(torch.sum(torch.sub(dict_corr[m], corr_emb) ** 2))
            dict_focal[m] = []
            for n in self.requirments:
                dict_focal[m].append(_regularizer(dict_loss[m], dict_loss[n], self.beta, self.threshold, self.reg_method))

            max_focal = np.max(dict_focal[m])
            dict_ssa_loss[m] = max_focal * dict_corr_diff[m]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda * dict_ssa_loss[r] + self.lamda_ae * autoencoder_loss

        return total_loss


class VAELoss(nn.Module):
    def __init__(self, cfg):
        super(VAELoss, self).__init__()
        self.requirments = cfg.MODALITY.REQUIRMENTS
        self.beta = cfg.FUSION_HEAD.MTUT.BETA
        self.lamda = cfg.FUSION_HEAD.MTUT.LAMDA
        self.threshold = cfg.FUSION_HEAD.MTUT.THRESOLD
        self.reg_method = cfg.FUSION_HEAD.MTUT.REG_METHOD
        self.autoencoder = NaiveVAE(cfg.MODALITY.NUMS * cfg.FUSION_HEAD.FEATURE_DIMS,
                                    cfg.FUSION_HEAD.FEATURE_DIMS)
        self.lamda_ae = cfg.FUSION_HEAD.MTUT.LAMDA_AE

        self.ssa_loss = KLSSALoss(cfg)

        if cfg.MODEL.DEVICE == "cuda":
            self.autoencoder = self.autoencoder.cuda()

    def forward(self, dict_loss, dict_x):
        concat_x = None
        for i, r in enumerate(self.requirments):
            if i == 0:
                concat_x = dict_x[r]
            else:
                tmpout = dict_x[r]
                concat_x = torch.cat([concat_x, tmpout], dim=1)

        decoder_out, mu, log_var, emb = self.autoencoder(concat_x)
        autoencoder_loss = self.loss_function(decoder_out, concat_x, mu, log_var)

        emb = F.normalize(emb, p=2, dim=1, eps=1e-12)
        corr_emb = torch.bmm(emb.unsqueeze(2), emb.unsqueeze(1))

        dict_corr = {}
        for r in self.requirments:
            # torch.mean(dict_x[r], dim=0)
            dict_x[r] = F.normalize(dict_x[r], p=2, dim=1, eps=1e-12)
            dict_corr[r] = torch.bmm(dict_x[r].unsqueeze(2), dict_x[r].unsqueeze(1))

        dict_focal = {}
        dict_corr_diff = {}
        dict_ssa_loss = {}
        for m in self.requirments:
            dict_ssa_loss[m] = 0.0
            dict_corr_diff[m] = torch.sqrt(torch.sum(torch.sub(dict_corr[m], corr_emb) ** 2))
            dict_focal[m] = (self._regularizer(dict_loss[m], self.beta, self.threshold, self.reg_method))

            # max_focal = np.max(dict_focal[m])
            dict_ssa_loss[m] = dict_focal[m] * dict_corr_diff[m]
        total_loss = {'ssaloss': dict_ssa_loss, 'total_loss': {}}
        for r in self.requirments:
            total_loss['total_loss'][r] = dict_loss[r] + self.lamda * dict_ssa_loss[r] + self.lamda_ae * autoencoder_loss

        # print(total_loss)
        return total_loss

    def _regularizer(self, loss_m, beta, threshold=0, method="exp"):
        if method == "exp":
            if loss_m > threshold:
                return (beta * math.exp(loss_m)) - 1
            else:
                return 0

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="mean")
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD


class CenterSSALoss(nn.Module):
    def __init__(self, cfg):
        super(CenterSSALoss, self).__init__()

    def forward(self, dict_loss, dict_x):
        pass


def _regularizer(loss1, loss2, beta=2.0, thresold=1e-5, method='exp'):
    if method == 'exp':
        if loss1 - loss2 > 0+thresold:
            return (beta * math.exp(loss1 - loss2)) - 1
        else:
            return 0.0
    elif method == "switch":
        if loss1 - loss2 > 0+thresold:
            return (loss1 - loss2) * F.sigmoid(beta * (loss1 - loss2))
        else:
            return 0.0
    elif method == "identity":
        if loss1 - loss2 > 0+thresold:
            return beta * (loss1 - loss2)
        else:
            return 0.0
    elif method == "multiply":
        if loss1 - loss2 > 0 + thresold:
            return (beta * math.exp(loss1 - loss2)) - 1
        else:
            return 0.0
    elif method == 'exp_in':
        if loss1 - loss2 > 0+thresold:
            return math.exp(beta*(loss1 - loss2)) - 1
        else:
            return 0.0
    elif method == "bi_exp":
        return (beta * math.exp(loss1 - loss2)) - 1
    elif method == "bi_switch":
        return (loss1 - loss2) * F.sigmoid(beta * (loss1 - loss2))
    elif method == "bi_exp_in":
        if loss1 - loss2 > 0+thresold:
            return (2 * math.exp(beta * (loss1 - loss2))) - 1
        else:
            return 0.0
    elif method == "bibi_exp_in":
        return (2 * math.exp(beta * (loss1 - loss2))) - 1

    elif method == "log":
        if loss1 - loss2 > 0 + thresold:
            return math.log(beta * (loss1 - loss2) + 1)
        return 0.0
