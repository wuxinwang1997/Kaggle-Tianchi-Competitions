# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .simplecnn import SimpleCNN
from .simplepcb import PCB
from .model import AIEarthModel
from .threedcnn import generate_model
from .multiresnet import MultiResnet
def build_model(cfg):
    model = MultiResnet(cfg)
    return model
