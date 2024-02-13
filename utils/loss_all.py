#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2023/3/28 20:08
# @Author: hanluyt
import torch
import torch.nn as nn
from typing import List, Optional, Union
import torch.nn.functional as F

class SIMSE(nn.Module):
    # scale–invariant mean squared error
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return mse + simse

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class Similarity_MMD(nn.Module):
    def __init__(self):
        super(Similarity_MMD, self).__init__()
    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        delta = input1_l2 - input2_l2  #[batch, 256]
        delta_mat = torch.mm(delta, delta.t())  # [batch, batch]
        diag = torch.diag(delta_mat)
        mmd_loss = torch.mean(diag)
        return mmd_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, sample_weight=None):
        """Compute loss for model
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)  # (bs, 256, 1)

        batch_size = features.shape[0]  # bs

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1] # 256
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # 256
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # self
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if sample_weight is None:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) * sample_weight / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: Union[List[float], float],
                 gamma: Optional[int] = 2,
                 with_logits: Optional[bool] = True):
        """
        :param alpha: weight of each class
        :param gamma: modulate the importance of easy samples
        :param with_logits: Whether the logits is followed by sigmoid or softmax function
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.FloatTensor([alpha]) if isinstance(alpha, float) else torch.FloatTensor(alpha)
        self.smooth = 1e-8
        self.with_logits = with_logits

    def _binary_class(self, input, target):
        prob = torch.sigmoid(input) if self.with_logits else input
        prob += self.smooth
        alpha = self.alpha.to(target.device)
        loss = -alpha * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob)
        return loss

    def _multiple_class(self, input, target):
        prob = F.softmax(input, dim=1) if self.with_logits else input

        alpha = self.alpha.to(target.device)
        alpha = alpha.gather(0, target)

        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)

        loss = -alpha * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input: [bs, num_classes]
        :param target: [bs]
        :return:
        """
        if len(input.shape) > 1 and input.shape[-1] != 1:
            loss = self._multiple_class(input, target)
        else:
            loss = self._binary_class(input, target)

        return loss.mean()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


