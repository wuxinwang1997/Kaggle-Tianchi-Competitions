# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
import torchvision.models as models
from .resnet import build_resnet_backbone
class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cnn = build_resnet_backbone(cfg)
        #if cfg.MODEL.BACKBONE.PRETRAIN:
        #   self.model.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_PATH))
        self.cnn.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.cnn.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 24)
    def forward(self, x):
        x = self.cnn(x)
        x = torch.squeeze(self.avgpool(x))
        x = self.fc(x)
        return x


