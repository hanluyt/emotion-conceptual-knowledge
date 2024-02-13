#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2022/8/25 3:54 下午
# @Author: hanluyt

import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
from PIL import Image
import numpy as np
import cv2

def train_transform(mean, std):
    t = []
    t.append(transforms.Resize(256))
    t.append(transforms.RandomCrop(224))
    t.append(transforms.RandomHorizontalFlip())
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def valid_transform(mean, std):
    t = []
    t.append(transforms.Resize(256))
    t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int]
) -> List[Tuple[str, int]]:
    """ return [(path_target1, label1), (..., ...)]"""
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)

        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances

def make_dataset_combine(
        directory: str,
        class_to_idx: Dict[str, int]
) -> List[Tuple[str, int]]:
    """ return [(path_target1, label1), (..., ...)]"""
    instances = []
    dir1, dir2 = directory.split()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir1 = os.path.join(dir1, target_class)
        target_dir2 = os.path.join(dir2, target_class)

        for root, _, fnames in sorted(os.walk(target_dir1, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)

        for root, _, fnames in sorted(os.walk(target_dir2, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances


class FERDataSet(Dataset):
    def __init__(self,
                 dir: str,
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None,
                 return_path = False,
                 combine = False,
                 ) -> None:
        self.root = dir

        if combine: # Training for RAF-DB + AffectNet
            classes, class_to_idx = self._find_classes_combine(self.root)
            samples = make_dataset_combine(self.root, class_to_idx)
        else:
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx)

        self.loader = loader
        self.transform = transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.return_path = return_path

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _find_classes_combine(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        dir1, dir2 = dir.split()
        classes = [d.name for d in os.scandir(dir1) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            try:
                path, label = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_path:
            return path, sample, label
        else:
            return sample, label

    def __len__(self) -> int:
        return len(self.samples)


# def pil_loader(path: str) -> Image.Image:
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = cv2.imread(path)
        return Image.fromarray(img)


class FERImageFolder(FERDataSet):
    def __init__(
            self,
            dir: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            return_path=False,
            combine = False,
    ):
        super(FERImageFolder, self).__init__(dir, loader, transform=transform,
                                             return_path=return_path, combine=combine)
        self.imgs = self.samples


def build_dataset(is_train, mean, std, args, return_path=False, combine=False):
    if is_train:
        transform = train_transform(mean, std)
        root = args.data_path
        dataset = FERImageFolder(root, transform=transform, return_path=return_path, combine=combine)

        # print("Transform (train) = ")
        # print(transform)
        # print("---------------------------")

    else:
        transform = valid_transform(mean, std)
        # print("Transform (val) = ")
        # print(transform)
        # print("---------------------------")
        root = args.eval_data_path
        dataset = FERImageFolder(root, transform=transform, return_path=return_path, combine=combine)

    nb_classes = args.nb_classes
    assert len(dataset.class_to_idx) == nb_classes
    # print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes

"""K=0.2 """
def get_cls_threshold(label_file, data_path, k=0.2):
    cls_list = os.listdir(data_path)
    cls_threshold = {}

    with open(label_file) as f:
        img_label_list = f.read().splitlines()

    for i, cls in enumerate(cls_list):
        attention_list = []
        for info in img_label_list:
            img_name, attention, label = info.split(' ')
            img_label = img_name.split('/')[0]

            if img_label == cls:
                attention_list.append(float(attention))

        attention_list = np.array(attention_list)
        low_high_index = np.argsort(attention_list)
        # ambiguity
        low_datasize = int(len(attention_list) * k)
        thre = attention_list[low_high_index[low_datasize]]
        cls_threshold[i] = thre
    return cls_threshold


def get_cls_threshold_combine(label_file, k=0.2):

    cls_threshold = {}

    with open(label_file) as f:
        img_label_list = f.read().splitlines()

    path_0 = img_label_list[0].split()[0]
    path_list = path_0.split('/')[:-2]
    path_list = [ele + '/' for ele in path_list]
    data_path = ''.join(path_list)
    cls_list = os.listdir(data_path)

    for i, cls in enumerate(cls_list):
        attention_list = []
        for info in img_label_list:
            img_name, attention, label = info.split(' ')
            img_label = img_name.split('/')[-2]

            if img_label == cls:
                attention_list.append(float(attention))

        # print(cls, len(attention_list))
        attention_list = np.array(attention_list)
        low_high_index = np.argsort(attention_list)
        # ambiguity
        low_datasize = int(len(attention_list) * k)
        thre = attention_list[low_high_index[low_datasize]]
        cls_threshold[i] = thre
    return cls_threshold


class Percept_weight(Dataset):
    def __init__(self, img_dir, label_file, transform=None, is_train=True, loader=pil_loader, combine=False):
        self.root = img_dir
        self.loader = loader
        self.transform = transform
        self.is_train = is_train
        img_list = []
        label_list = []
        attention_mask_list = []

        if combine:
            cls_threshold = get_cls_threshold_combine(label_file=label_file)
        else:
            cls_threshold = get_cls_threshold(label_file=label_file, data_path=img_dir)

        with open(label_file) as f:
            img_label_list = f.read().splitlines()

        for info in img_label_list:
            img_name, attention, label = info.split(' ')

            if combine:
                img_list.append(img_name)
            else:
                img_path = os.path.join(self.root, img_name)
                img_list.append(img_path)
            label_list.append(int(float(label)))
            if float(attention) < cls_threshold[int(float(label))]:
                mask = 0
            else:
                mask = 1
            attention_mask_list.append(mask)

        self.image_list = img_list
        self.label_list = label_list
        self.attention_mask_list = attention_mask_list

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            try:
                img_path = self.image_list[index]
                attention_mask = self.attention_mask_list[index]
                label_img = self.label_list[index]
                img = self.loader(img_path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.image_list) - 1)

        if self.transform is not None:
            img = self.transform(img)

        if self.is_train:
            return img, attention_mask, label_img
        else:
            return img, label_img

    def __len__(self) -> int:
        return len(self.image_list)


################## Analysis 1: emotion concept #######################
def split_combine_dataset(directory: str, random_state=0):
    """ return two split-half txt file"""
    dir1, dir2 = directory.split()
    classes = [d.name for d in os.scandir(dir1) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    path_all_human, label_all_human = [], []
    path_all_cartoon, label_all_cartoon = [], []
    random.seed(random_state)

    for target_class in sorted(class_to_idx.keys()):
        instances1 = []
        instances2 = []
        class_index = class_to_idx[target_class]
        target_dir1 = os.path.join(dir1, target_class)
        target_dir2 = os.path.join(dir2, target_class)

        for root, _, fnames in sorted(os.walk(target_dir1, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                instances1.append(path)

        half_length1 = len(instances1) // 2
        random_half_instances1 = random.sample(instances1, half_length1)
        remain_instances1 = list(set(instances1) - set(random_half_instances1))

        for root, _, fnames in sorted(os.walk(target_dir2, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                instances2.append(path)

        half_length2 = len(instances2) // 2
        random_half_instances2 = random.sample(instances2, half_length2)
        remain_instances2 = list(set(instances2) - set(random_half_instances2))

        path_all_cartoon.append(random_half_instances1 + random_half_instances2)
        label_all_cartoon.append([class_index] * (half_length1 + half_length2))

        path_all_human.append(remain_instances1 + remain_instances2)
        label_all_human.append([class_index] * (len(remain_instances1) + len(remain_instances2)))


    return path_all_human, label_all_human, path_all_cartoon, label_all_cartoon


# save_file = 'human_all.txt'
# with open(save_file, 'w') as f:
#     for path, label in zip(path_all_human, label_all_human):
#         f.write(path)
#         f.write(' ')
#         f.write(str(label))
#         f.write('\n')

"""Split-half human facial expression dataset: train"""
class Half_FER(Dataset):
    def __init__(self, img_file, transform=None, loader=pil_loader):
        self.loader = loader
        self.transform = transform

        img_list = []
        label_list = []

        with open(img_file) as f:
            img_label_list = f.read().splitlines()

        for info in img_label_list:
            img_name, label = info.split(' ')
            img_path = img_name
            img_list.append(img_path)
            label_list.append(int(float(label)))

        self.image_list = img_list
        self.label_list = label_list
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            try:
                img_path = self.image_list[index]
                label_img = self.label_list[index]
                img = self.loader(img_path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.image_list) - 1)

        if self.transform is not None:
            img = self.transform(img)


        return img, label_img

    def __len__(self) -> int:
        return len(self.image_list)
