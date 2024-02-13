#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2024/2/13 22:11
# @Author: hanluyt

from dataset_loader import get_cls_threshold
import pandas as pd

label_file = '../data/raf_confidence_train.txt'
data_path = '../data/RAF-DB_lab/train'
cls_threshold = get_cls_threshold(label_file=label_file, data_path=data_path)
print(cls_threshold)


high = []
with open(label_file) as f:
    img_label_list = f.read().splitlines()

for info in img_label_list:
    img_name, attention, label = info.split(' ')
    if float(attention) >= cls_threshold[int(float(label))]:
        high.append([img_name, float(attention), int(float(label))])

df_high = pd.DataFrame(high, columns=['img', 'attention', 'label'])
df_high.to_csv('../data/raf_high.csv', index=False)
print(df_high)

