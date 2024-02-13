#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2022/9/17 4:01 下午
# @Author: hanluyt
# Email: hlu20@fudan.edu.cn

from __future__ import print_function
import argparse
import torch
from torchvision import transforms
import os
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from dataset_loader import build_dataset, FERImageFolder,  Percept_weight, train_transform, valid_transform
from models.model_emotion import ConCept, PerCept
from timm.utils import ModelEma
import pandas as pd
import time
import datetime
from utils import dist_train, data_util


def get_args():
    parser = argparse.ArgumentParser('Facial Expression Recognition', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--alpha', default=2, type=float, help="hyperparameter for CE")
    parser.add_argument('--beta', default=1, type=float, help="hyperparameter for similarity")
    parser.add_argument('--gamma', default=1, type=float, help="hyperparameter for distill")

    # model parameters
    parser.add_argument('--model_name', default='percept_result.pth')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay (default: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')

    # Dataset parameters
    parser.add_argument('--data_path', default='../data/RAF-DB_lab/train', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='../data/RAF-DB_lab/test', type=str,
                        help='dataset path for evaluation')

    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='../result',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser.parse_args()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4"
    args = get_args()
    dist_train.init_distributed_mode(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    # Compute the mean and variance for the dataset
    dataset = FERImageFolder(args.data_path, transform=transforms.ToTensor())
    mean, std = data_util.getStats(dataset)

    transform_train = train_transform(mean=mean, std=std)
    transform_val = valid_transform(mean=mean, std=std)


    dataset_train = Percept_weight(img_dir=args.data_path,
                                   label_file='../data/raf_confidence_train.txt', is_train=True,
                                   transform=transform_train, combine=False)
    dataset_val = Percept_weight(img_dir=args.eval_data_path,
                                   label_file='../data/raf_confidence_test.txt', is_train=False,
                                   transform=transform_val)
    print("TrainSet:", len(dataset_train))
    print("TestSet:", len(dataset_val))

    if args.distributed:
        num_tasks = dist_train.get_world_size()  # 4
        global_rank = dist_train.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks,
                                                            rank=global_rank, shuffle=True)
        print("Length of Sampler_train = %s" % len(sampler_train))

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        print("Length of Sampler_train = %s" % len(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    sampler=sampler_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem,
                                                    drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  sampler=sampler_val,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.num_workers,
                                                  pin_memory=args.pin_mem,
                                                  drop_last=False)

    # Model
    print('-----------------------')
    print(f"==> Creating conceptual model, Train on {args.model_name[:-4]}")
    model_percept = PerCept(n_class=7, baseline=False)
    model_concept = ConCept(n_class=7, atten=True)

    model_percept.to(args.device)
    model_concept.to(args.device)

    checkpoint = torch.load('../result/concept_result.pth')
    model_concept.load_state_dict(checkpoint['model'])


    model_percept_without_ddp = model_percept
    model_concept_without_ddp = model_concept
    n_parameters = sum(p.numel() for p in model_percept.parameters() if p.requires_grad)
    print('Number of params in Perceptual Model:', n_parameters / 1e6, 'M')

    n_parameters = sum(p.numel() for p in model_concept.parameters() if p.requires_grad)
    print('Number of params in Conceptual Model:', n_parameters / 1e6, 'M')

    for k, v in model_concept.named_parameters():
        v.requires_grad = False

    total_batch_size = args.batch_size * dist_train.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / args.batch_size
    # args.lr = 0.00002
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)
    print(f"CE: {args.alpha}, sim:{args.beta}, kl:{args.gamma}")

    if args.distributed:
        model_percept = torch.nn.parallel.DistributedDataParallel(model_percept, device_ids=[args.gpu], find_unused_parameters=True)
        model_percept_without_ddp = model_percept.module


    optimizer = torch.optim.Adam(model_percept_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.500, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

    t_total = args.epochs

    criterion = nn.CrossEntropyLoss()  # recognition task
    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    total_start_time = time.time()
    model_percept.zero_grad()

    best_acc = 0
    total_train_acc, total_train_loss, total_test_acc, total_test_loss = [], [], [], []

    model_concept.eval()

    high_all = pd.read_csv('../data/raf_high.csv')

    for epoch in range(1, t_total + 1):

        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_acc, train_loss = data_util.train_one_epoch_emotion_guidance(args, transform_train, model_percept, model_concept,
                                                                           criterion, data_loader_train, optimizer,
                                                                           args.device, epoch, t_total, high_all,
                                                                           lr_scheduler=lr_scheduler, if_percept=False, combine=False)


        total_train_acc.append(train_acc), total_train_loss.append(train_loss)

        test_acc, test_loss = data_util.valid(args, model_percept, data_loader_val, epoch, t_total)
        total_test_acc.append(test_acc), total_test_loss.append(test_loss)

        if best_acc < test_acc:
            data_util.save_model(args, model_percept_without_ddp, tag=args.model_name)
            best_acc = test_acc

        model_percept.train()

        total_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Best Accuracy: %f, time for this epoch: %s" % (best_acc, total_time_str))


    print("###################################")
    print("Best Accuracy: \t%f" % best_acc)
    print("End Training!")

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

