#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2024/2/13 16:46
# @Author: hanluyt

import torch
# from torch._six import inf
import os
from typing import Iterable, Optional
from timm.utils import ModelEma
import math
import sys
import numpy as np
from .dist_train import get_rank
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import math
import pandas as pd
import random
from PIL import Image
from .loss_all import DiffLoss, SupConLoss, FocalLoss, mmd_rbf

def getStats(trainset):
    """Compute mean and variance for facial expression training data [RAF-DB, AffectNet, FERPlus]
    """
    print("Compute mean and variance for training data")
    print(len(trainset))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    device = torch.device('cuda')
    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)

    for X, _ in train_loader:
        X = X.to(device)
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(trainset))
    std.div_(len(trainset))
    return list(mean.cpu().numpy()), list(std.cpu().numpy())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()  # np.array

def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model

def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_model(args, model_without_ddp, tag):
    checkpoint_path = os.path.join(args.output_dir, tag)
    to_save = {
        'model': model_without_ddp.state_dict()
    }

    save_on_master(to_save, checkpoint_path)
    print("Saved model checkpoint to [DIR: %s]" % (args.output_dir))

def valid(args, model, test_loader, step_epoch, global_epoch):
    # validation
    eval_losses = AverageMeter()
    total_im, total_correct = 0, 0
    model.eval()

    loss_fct = nn.CrossEntropyLoss()
    for step, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        total_im += imgs.size(0)
        with torch.no_grad():
            return_all = model(imgs)

            if isinstance(return_all, tuple):
                logits = return_all[-1]
            else:
                logits = return_all

            eval_loss = loss_fct(logits, labels)

            eval_losses.update(eval_loss.item())
            correct_num = (logits.max(-1)[-1] == labels).sum()
            total_correct += correct_num

    val_acc = (total_correct / total_im).item()
    val_loss = eval_losses.avg

    print('Validation: {%d / %d epoch} ----- val acc: %2.5f, val_loss: %2.5f' % (step_epoch, global_epoch, val_acc, val_loss))
    return val_acc, val_loss

def cfc_learning(emotion_feature, targets):
    cc = targets.detach().cpu().numpy()
    stat, class_weight = {}, {}
    for key in cc:
        stat[key] = stat.get(key, 0) + 1

    for k, v in stat.items():
        class_weight[k] = 1 - v / targets.size(0)

    sample_weight = [class_weight[key] for key in cc]
    sample_weight = torch.tensor(sample_weight, device="cuda").repeat(emotion_feature.size(1))

    emotion_feature_norm = F.normalize(emotion_feature, dim=1)
    loss_contrastive = SupConLoss()(emotion_feature_norm.view(emotion_feature_norm.size(0),
                                                         emotion_feature_norm.size(1), 1), targets, sample_weight)
    return loss_contrastive


def rr_loss(args, attention_weights):
    tops = int(args.batch_size * 0.7)
    # Rank Regularization
    _, top_idx = torch.topk(attention_weights.squeeze(), tops)
    _, down_idx = torch.topk(attention_weights.squeeze(), args.batch_size - tops, largest=False)

    high_group = attention_weights[top_idx]
    low_group = attention_weights[down_idx]
    high_mean = torch.mean(high_group)
    low_mean = torch.mean(low_group)

    diff = low_mean - high_mean + 0.15

    if diff > 0:
        RR_loss = diff
    else:
        RR_loss = 0
    return RR_loss


def train_one_epoch_concept(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, step_epoch: int, global_epoch: int,
                    lr_scheduler=None, cfc=False, focal=False):
    model.train()
    train_losses = AverageMeter()
    train_acces = AverageMeter()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        return_all = model(samples)
        assert len(return_all) >=2 and len(return_all) <=4

        if len(return_all) == 2:
            logits = return_all[-1]
            train_loss = criterion(logits, targets)

        elif len(return_all) == 3:
            emotion_feature, identity_feature, logits = return_all
            loss_task = criterion(logits, targets)
            loss_disen = DiffLoss()(emotion_feature, identity_feature)
            if cfc:
                loss_cfc = cfc_learning(emotion_feature, targets)
                train_loss = args.alpha * loss_disen + args.beta * loss_task + args.gamma * loss_cfc
            else:
                train_loss = args.alpha * loss_disen + args.beta * loss_task
        else:
            emotion_feature, identity_feature, attention_weights, logits = model(samples)
            if focal:
                focal_loss = FocalLoss(alpha=[0.7, 0.8, 0.9, 0.2, 0.4, 0.4, 0.6], gamma=2) # RAF-DB
                loss_task = focal_loss(logits, targets)
            else:
                loss_task = criterion(logits, targets)

            loss_RR = rr_loss(args, attention_weights)
            loss_disen = DiffLoss()(emotion_feature, identity_feature)
            loss_cfc = cfc_learning(emotion_feature, targets)

            train_loss = args.alpha * loss_disen + args.beta * (loss_task + loss_RR) + args.gamma * loss_cfc


        train_losses.update(train_loss.item())


        if not math.isfinite(train_loss.item()):
            print("Loss is {}, stopping training".format(train_loss.item()))
            sys.exit(1)

        # print(f'{data_iter_step}/{len(data_loader)}', train_loss)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        torch.cuda.synchronize()

        class_acc = (logits.max(-1)[-1] == targets).float().mean()
        train_acces.update(class_acc.item())

    if args.distributed:
        t_loss = torch.tensor([train_losses.count, train_losses.sum], dtype=torch.float64, device="cuda")
        t_acc = torch.tensor([train_acces.count, train_acces.sum], dtype=torch.float64, device="cuda")
        dist.barrier() # synchronizes all processes
        dist.all_reduce(t_loss, op=torch.distributed.ReduceOp.SUM, )
        dist.all_reduce(t_acc, op=torch.distributed.ReduceOp.SUM, )
        t_loss = t_loss.tolist()
        t_acc = t_acc.tolist()
        t_loss_count, t_loss_sum = int(t_loss[0]), t_loss[1]
        t_acc_count, t_acc_sum = int(t_acc[0]), t_acc[1]
        loss_avg = t_loss_sum / t_loss_count
        acc_avg = t_acc_sum / t_acc_count
    else:
        loss_avg = train_losses.avg
        acc_avg = train_acces.avg

    print('Train: {%d / %d epochs} ----- train acc: %2.5f, train loss: %2.5f' % (step_epoch, global_epoch, acc_avg, loss_avg))
    return acc_avg, loss_avg


def train_one_epoch_resnet18(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, step_epoch: int, global_epoch: int,
                    model_ema: Optional[ModelEma] = None, lr_scheduler=None):
    model.train()
    train_losses = AverageMeter()
    train_acces = AverageMeter()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        _, logits = model(samples)

        train_loss = criterion(logits, targets)
        train_losses.update(train_loss.item())

        if not math.isfinite(train_loss.item()):
            print("Loss is {}, stopping training".format(train_loss.item()))
            sys.exit(1)

        # print(f'{data_iter_step}/{len(data_loader)}', train_loss)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        class_acc = (logits.max(-1)[-1] == targets).float().mean()
        train_acces.update(class_acc.item())

    if args.distributed:
        t_loss = torch.tensor([train_losses.count, train_losses.sum], dtype=torch.float64, device="cuda")
        t_acc = torch.tensor([train_acces.count, train_acces.sum], dtype=torch.float64, device="cuda")
        dist.barrier() # synchronizes all processes
        dist.all_reduce(t_loss, op=torch.distributed.ReduceOp.SUM, )
        dist.all_reduce(t_acc, op=torch.distributed.ReduceOp.SUM, )
        t_loss = t_loss.tolist()
        t_acc = t_acc.tolist()
        t_loss_count, t_loss_sum = int(t_loss[0]), t_loss[1]
        t_acc_count, t_acc_sum = int(t_acc[0]), t_acc[1]
        loss_avg = t_loss_sum / t_loss_count
        acc_avg = t_acc_sum / t_acc_count
    else:
        loss_avg = train_losses.avg
        acc_avg = train_acces.avg

    print('Train: {%d / %d epochs} ----- train acc: %2.5f, train loss: %2.5f' % (step_epoch, global_epoch, acc_avg, loss_avg))
    return acc_avg, loss_avg


def concept_k(args, model, root_dir, transform, img, mask, label, high_all, K=8, combine=False):
    """
    mask: compare with threshold (>: 1, <: 0)
    return: emotion representation corresponding to perceptual process"""
    # model.eval()
    concept_img, mask_allocate, concept_feature = [], [], []
    for i, ele in enumerate(mask):
        if ele == 1:
            concept_img.append(img[i].unsqueeze(dim=0))
            mask_allocate.append(1)
        else:
            high_group = high_all[high_all['label'] == label[i].detach().cpu().numpy()]
            concept_dir = high_group['img'].astype(str).tolist()
            assert K <= len(concept_dir)
            subset = random.sample(concept_dir, K)
            if combine:
                path_concept = subset
            else:
                path_concept = [os.path.join(root_dir, ele) for ele in subset]
            for img_path in path_concept:
                img_c = pil_loader(img_path)
                img_c = transform(img_c)
                img_c = img_c.to(args.device)
                concept_img.append(img_c.unsqueeze(dim=0))
                mask_allocate.append(0)
    concept_img = torch.cat(concept_img)
    with torch.no_grad():
        emotion_feature, _, _, logits_c = model(concept_img.detach())
    i = 0
    while i < len(mask_allocate):
        if mask_allocate[i] == 1:
            concept_feature.append(emotion_feature[i].unsqueeze(dim=0))
            i += 1
        else:
            concept_feature.append(torch.mean(emotion_feature[i:i+K], dim=0).unsqueeze(dim=0))
            i += K

    concept_feature = torch.cat(concept_feature)
    mask_allocate = torch.tensor(mask_allocate, device="cuda")
    return concept_feature, logits_c, mask_allocate

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def train_one_epoch_emotion_guidance(args, transform, model_p: torch.nn.Module, model_c: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, step_epoch: int, global_epoch: int, high_all,
                    lr_scheduler=None, if_percept=False, combine=False):
    model_p.train()
    model_c.eval()
    train_losses = AverageMeter()
    train_acces = AverageMeter()


    for data_iter_step, (samples, mask, targets) in enumerate(data_loader):
        samples = samples.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        percept_feature, logits_p = model_p(samples)

        if if_percept:
            concept_feature, logits_c, mask_allocate = concept_k(args=args, model=model_c,
                                                  root_dir=args.data_path,
                                                  transform=transform, img=samples, mask=mask,
                                                  label=targets, high_all=high_all, K=8, combine=combine)
            loss_sim = mmd_rbf(concept_feature, percept_feature)

            mask_nonzero = torch.nonzero(mask, as_tuple=False).squeeze(dim=1)
            logits_p_high = logits_p[mask_nonzero]

            mask_allocate_nonzero = torch.nonzero(mask_allocate, as_tuple=False).squeeze(dim=1)

            logits_c_high = logits_c[mask_allocate_nonzero]

            loss_kl = F.kl_div(logits_p_high.softmax(dim=-1).log(), logits_c_high.softmax(dim=-1),
                               reduction='batchmean')

        else:
            with torch.no_grad():
                emotion_feature, _, _, logits_c = model_c(samples.detach())
            loss_sim = mmd_rbf(emotion_feature, percept_feature)
            loss_kl = F.kl_div(logits_p.softmax(dim=-1).log(), logits_c.softmax(dim=-1), reduction='batchmean')

        loss_task = criterion(logits_p, targets)
        # focal_loss = FocalLoss(alpha=[0.6, 0.8, 0.8, 0.2, 0.4, 0.6, 0.7], gamma=2)
        # loss_task = focal_loss(logits_p, targets)

        train_loss = args.alpha * loss_task + args.beta * loss_sim + args.gamma * loss_kl


        train_losses.update(train_loss.item())

        if not math.isfinite(train_loss.item()):
            print("Loss is {}, stopping training".format(train_loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        torch.cuda.synchronize()

        class_acc = (logits_p.max(-1)[-1] == targets).float().mean()
        train_acces.update(class_acc.item())

    if args.distributed:
        t_loss = torch.tensor([train_losses.count, train_losses.sum], dtype=torch.float64, device="cuda")
        t_acc = torch.tensor([train_acces.count, train_acces.sum], dtype=torch.float64, device="cuda")
        dist.barrier() # synchronizes all processes
        dist.all_reduce(t_loss, op=torch.distributed.ReduceOp.SUM, )
        dist.all_reduce(t_acc, op=torch.distributed.ReduceOp.SUM, )
        t_loss = t_loss.tolist()
        t_acc = t_acc.tolist()
        t_loss_count, t_loss_sum = int(t_loss[0]), t_loss[1]
        t_acc_count, t_acc_sum = int(t_acc[0]), t_acc[1]
        loss_avg = t_loss_sum / t_loss_count
        acc_avg = t_acc_sum / t_acc_count
    else:
        loss_avg = train_losses.avg
        acc_avg = train_acces.avg

    print('Train: {%d / %d epochs} ----- train acc: %2.5f, train loss: %2.5f' % (step_epoch, global_epoch, acc_avg, loss_avg))

    return acc_avg, loss_avg
