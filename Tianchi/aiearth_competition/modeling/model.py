# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
from .resnet import build_resnet_backbone
from .threedcnn import generate_model

class AIEarthModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cnn = generate_model(cfg.MODEL.BACKBONE.DEPTH, with_se=cfg.MODEL.BACKBONE.WITH_SE)

    def forward(self, x):
        sst = x
        sst = torch.unsqueeze(sst, dim=1)
        return self.cnn(sst)
    #     self.backbones = nn.ModuleList([build_resnet_backbone(cfg) for i in range(2)])
    #     self.avgpool = nn.AdaptiveAvgPool2d((1,96))
    #     self.lstm = nn.LSTM(input_size=3 * 2 ,hidden_size=32,num_layers=3,batch_first=True,bidirectional=True)
    #     self.batch_norm = nn.BatchNorm1d(512, affine=False)
    #     self.linear = nn.Linear(96, 24)
    #
    # def forward(self, x):
    #     sst, t300 = x#, ua, va = x
    #
    #     sst = self.backbones[0](sst)
    #     t300 = self.backbones[1](t300)
    #     #ua = self.backbones[2](ua)
    #     #va = self.backbones[3](va)
    #
    #     sst = torch.flatten(sst, start_dim=2)
    #     t300 = torch.flatten(t300, start_dim=2)
    #     #ua = torch.flatten(ua, start_dim=2)
    #     #va = torch.flatten(va, start_dim=2)
    #
    #     output = torch.cat([sst, t300], dim=-1)#, ua, va], dim=-1)
    #     output = self.batch_norm(output)
    #     output, _ = self.lstm(output)
    #     output = self.avgpool(output).squeeze(dim=-2)
    #     output = self.linear(output)
    #
    #     return output
