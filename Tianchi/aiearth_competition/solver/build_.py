# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
import torch.nn as nn

def make_optimizer(cfg, model):
    params = []
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        #if "bias" in key:
        #    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        #    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        #if "bn" in key:
        #    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BN
        if cfg.SOLVER.TRAIN_SODA:
            if "fc" in key:
                lr = cfg.SOLVER.BASE_LR * 0.5
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(pg0, lr=lr, momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(pg0, betas=(cfg.SOLVER.MOMENTUM, 0.999))
    optimizer.add_param_group({'params': pg1, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    return optimizer
