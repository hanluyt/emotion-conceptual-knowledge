#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2024/2/13 16:00
# @Author: hanluyt

from __future__ import print_function
import argparse
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from dataset_loader import FERImageFolder, build_dataset
from models.model_emotion import ConCept
import time
import datetime
from utils import dist_train, data_util
import random
import os

def get_args():
    parser = argparse.ArgumentParser('Facial Expression Recognition', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--alpha', default=0.1, type=float, help="hyperparameter for orthogonal loss L_orth")
    parser.add_argument('--beta', default=1, type=float, help="hyperparameter for L_WCE + L_RR")

    parser.add_argument('--gamma', default=0.01, type=float, help="hyperparameter for contrastive loss L_CC")

    # model parameters
    parser.add_argument('--model_name', default='concept_result.pth')
    parser.add_argument('--seed', default=42, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='learning rate (default: 0.0001)')

    # Dataset parameters
    parser.add_argument('--data_path', default='../data/RAF-DB_lab/train', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='../data/RAF-DB_lab/test', type=str,
                        help='dataset path for evaluation')

    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='../result',
                        help='path where to save, empty for no saving')


    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser.parse_args()

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = get_args()
    dist_train.init_distributed_mode(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    # Compute the mean and variance for the dataset
    dataset = FERImageFolder(args.data_path, transform=transforms.ToTensor())
    mean, std = data_util.getStats(dataset)

    # Dataset
    dataset_train, _ = build_dataset(is_train=True, mean=mean, std=std, args=args)
    dataset_val, _ = build_dataset(is_train=False, mean=mean, std=std, args=args)
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
    model = ConCept(n_class=7, atten=True)  # return 4 params
    model.to(args.device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params in Conceptual Model:', n_parameters / 1e6, 'M')

    total_batch_size = args.batch_size * dist_train.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / args.batch_size  # 256
    # args.lr = 0.00001
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)
    print(f"Orthogonal: {args.alpha}, L_WCE+LRR:{args.beta}, CC:{args.gamma}")

    if args.distributed:

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Configuration for training
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_without_ddp.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay, betas=(0.500, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    criterion = nn.CrossEntropyLoss()  # recognition task
    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    total_start_time = time.time()
    model.zero_grad()

    t_total = args.epochs
    best_acc = 0.85
    total_train_acc, total_train_loss, total_test_acc, total_test_loss = [], [], [], []

    # Start training and save model
    for epoch in range(1, t_total + 1):
        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_acc, train_loss = data_util.train_one_epoch_concept(args, model, criterion, data_loader_train, optimizer,
                                                          args.device, epoch, t_total,
                                                          lr_scheduler=lr_scheduler, cfc=True, focal=True)

        total_train_acc.append(train_acc), total_train_loss.append(train_loss)

        test_acc, test_loss = data_util.valid(args, model, data_loader_val, epoch, t_total)
        total_test_acc.append(test_acc), total_test_loss.append(test_loss)

        if test_acc > best_acc:
            data_util.save_model(args, model_without_ddp, tag=args.model_name)
            best_acc = test_acc

        model.train()

        total_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        if best_acc > 0.85:
            print("Best Accuracy: %f, time for this epoch: %s" % (best_acc, total_time_str))
        else:
            print("Time for this epoch: %s" % total_time_str)

    print("###################################")
    if best_acc > 0.85:
        print("Best Accuracy: \t%f" % best_acc)
    print("End Training!")

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))




