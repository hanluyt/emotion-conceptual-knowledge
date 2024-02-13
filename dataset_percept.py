#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2024/2/13 21:29
# @Author: hanluyt

import torch
import os
from main_concept import get_args
from dataset_loader import FERImageFolder, build_dataset
from models.model_emotion import ConCept
import time
import datetime
from utils import data_util

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
args = get_args()
args.device = torch.device('cuda')
args.data_path = '../data/RAF-DB_lab/train'

# RAF-DB
# Compute the mean and variance for the dataset

mean = [0.4012176, 0.4495195, 0.57520276]
std = [0.18263723, 0.19109425, 0.20839755]

dataset_train, _ = build_dataset(is_train=True, mean=mean, std=std, args=args, return_path=True)
sampler_train = torch.utils.data.RandomSampler(dataset_train)
print("TrainSet:", len(dataset_train))
# dataset_val, _ = build_dataset(is_train=False, mean=mean, std=std, args=args, return_path=True)
# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
# print("TestSet:", len(dataset_val))

data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    sampler=sampler_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem,
                                                    drop_last=False)

# data_loader_val = torch.utils.data.DataLoader(dataset_val,
#                                                   sampler=sampler_val,
#                                                   batch_size=args.batch_size,
#                                                   num_workers=args.num_workers,
#                                                   pin_memory=args.pin_mem,
#                                                   drop_last=False)


model = ConCept(n_class=7, atten=True)
model.to(args.device)
checkpoint = torch.load('../result/concept_result.pth')

model.load_state_dict(checkpoint['model'])
model.eval()

# test_acc, test_loss = data_util.valid(args, model, data_loader_val, 1, 1)
# print(test_acc)

datasize = len(dataset_train)
# datasize = len(dataset_val)
attention_all = torch.zeros(datasize, device="cuda")
label_all = torch.zeros(datasize, device="cuda")
predict_all = torch.zeros(datasize, device="cuda")
path_all = []

for i, (path, img, label) in enumerate(data_loader_train):
    img, label = img.to(args.device), label.to(args.device)
    with torch.no_grad():
        _, _, attention, logits = model(img)

        preds = torch.argmax(logits, dim=-1)

        attention = attention.squeeze()
        path_all.append(path)
        inte = len(dataset_train) // args.batch_size
        # print(i)
        if i == inte:
            attention_all[i * args.batch_size:] = attention
            label_all[i * args.batch_size:] = label
            predict_all[i*args.batch_size:] = preds
            break
        attention_all[i * args.batch_size:(i + 1) * args.batch_size] = attention
        label_all[i * args.batch_size:(i + 1) * args.batch_size] = label
        predict_all[i * args.batch_size:(i + 1) * args.batch_size] = preds

path_all = sum(path_all, [])
# path_all_new = path_all
path_all_new = []
for ele in path_all:
    cls, imgname = ele.split('/')[-2:][0], ele.split('/')[-2:][1]
    newname = os.path.join(cls, imgname)
    path_all_new.append(newname)

print(attention_all.size())
print(label_all)
print(path_all_new[0])

attention_all = attention_all.detach().cpu().numpy()
label_all = label_all.detach().cpu().numpy()

predict_all = predict_all.detach().cpu().numpy()
save_file = '../data/raf_confidence_train.txt'

with open(save_file, 'w') as f:
    for path, attention, label in zip(path_all_new, attention_all, label_all):
        f.write(path)
        f.write(' ')
        f.write(str(attention))
        f.write(' ')
        f.write(str(label))
        # f.write(' ')
        # f.write(str(predict))
        f.write('\n')