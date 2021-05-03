# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .transforms import RandomErasing
from torchvision import transforms

def get_train_transforms(cfg):
    train_transforms = transforms.Compose([
        RandomErasing(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    return train_transforms

def get_valid_transforms(cfg):
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    return valid_transforms

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = get_train_transforms(cfg)
    else:
        transform = get_valid_transforms(cfg)

    return transform
