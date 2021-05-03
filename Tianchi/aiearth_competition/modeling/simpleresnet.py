# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        if cfg.MODEL.BACKBONE.PRETRAIN is not '':
           resnet.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_PATH))
        resnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, x):
        return self.model(x)


