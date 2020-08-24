# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import cv2
import PIL
from PIL import Image

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import Sampler

# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

from RandAugment import RandAugment
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import SubsetRandomSampler

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

def get_dataset(dataset, batch_size, split=1.0, cutout_length=0, N=3, M=5, RandA=False):
    """
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]

    normalize = [
        transforms.Resize(224), # 32
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf + normalize + cutout)
    # if RandA:
        # Add RandAugment with N, M(hyperparameter)
        # train_transform.transforms.insert(0, RandAugment(N, M))

    valid_transform = transforms.Compose(normalize)
    """

    train_transform = transforms.Compose([
        # transforms.Resize((224, 224)), # 224 Spop
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if RandA:
        # Add RandAugment with N, M(hyperparameter)
        train_transform.transforms.insert(0, RandAugment(N, M))

    valid_transform = transforms.Compose([
        # transforms.Resize((224, 224)), # 224 Spos
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar100':
        dataset_train = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR100(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError

    split_idx = 0
    train_sampler = None
    if split < 1.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=1-split, random_state=0)
        sss = sss.split(list(range(len(dataset_train))), dataset_train.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=32,
        pin_memory=True, sampler=train_sampler, drop_last=True)

    # valid_loader = torch.utils.data.DataLoader(
    #     dataset_valid, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=16,
    #     pin_memory=True, sampler=valid_sampler, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    print(
          'train/valid:{}/{},\t '
          'batchsize:{} * train-step/valid-step:{}/{} -> train/valid:{}/{},\t'
          'Split train/valid:{}/{} '.format(
        len(dataset_train),
        len(dataset_valid),
        batch_size,
        int(len(train_loader)),
        int(len(valid_loader)),
        int(len(train_loader)) * batch_size,
        int(len(valid_loader)) * batch_size,
        len(train_loader)/(len(dataset_train)//batch_size),
        len(valid_loader)/(len(dataset_valid)//batch_size),
        )
    )

    train_dataprovider = DataIterator(train_loader)
    val_dataprovider = DataIterator(valid_loader)

    return train_dataprovider, val_dataprovider, int(len(train_loader)), int(len(valid_loader))
