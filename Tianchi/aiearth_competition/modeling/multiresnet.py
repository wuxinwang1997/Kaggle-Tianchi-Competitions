# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
from .simpleresnet import SimpleCNN


class MultiResnet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cnn = nn.ModuleList([SimpleCNN(cfg) for i in range(4)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 128))
        self.lstm = nn.LSTM(input_size=3 * 4, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(512, affine=False)
        self.linear = nn.Linear(128, 24)

    def forward(self, x):
        sst, t300, ua, va = x

        sst = self.cnn[0](sst)
        t300 = self.cnn[1](t300)
        ua = self.cnn[2](ua)
        va = self.cnn[3](va)

        sst = torch.flatten(sst, start_dim=2)
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)

        output = torch.cat([sst, t300, ua, va], dim=-1)
        output = self.batch_norm(output)
        output, _ = self.lstm(output)
        output = self.avgpool(output).squeeze(dim=-2)
        output = self.linear(output)

        return output
