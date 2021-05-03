# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import argparse
import os
import sys
sys.path.append('.')
from os import mkdir
from config import cfg
from data import make_test_data_loader
from engine.predicter import Predicter
from modeling import build_model
import random
import torch
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def predict(cfg):
    seed_everything(cfg.SEED)
    models = []
    for i in range(5):
        models.append(build_model(cfg))
    if torch.cuda.is_available():
        device = 'cuda'
        models[0].load_state_dict(torch.load(cfg.TEST.WEIGHT_00)['model_state_dict'])
        models[1].load_state_dict(torch.load(cfg.TEST.WEIGHT_01)['model_state_dict'])
        models[2].load_state_dict(torch.load(cfg.TEST.WEIGHT_02)['model_state_dict'])
        models[3].load_state_dict(torch.load(cfg.TEST.WEIGHT_03)['model_state_dict'])
        models[4].load_state_dict(torch.load(cfg.TEST.WEIGHT_04)['model_state_dict'])
        # models[5].load_state_dict(torch.load(cfg.TEST.WEIGHT_10)['model_state_dict'])
        # models[6].load_state_dict(torch.load(cfg.TEST.WEIGHT_11)['model_state_dict'])
        # models[7].load_state_dict(torch.load(cfg.TEST.WEIGHT_12)['model_state_dict'])
        # models[8].load_state_dict(torch.load(cfg.TEST.WEIGHT_13)['model_state_dict'])
        # models[9].load_state_dict(torch.load(cfg.TEST.WEIGHT_14)['model_state_dict'])
        # models[10].load_state_dict(torch.load(cfg.TEST.WEIGHT_20)['model_state_dict'])
        # models[11].load_state_dict(torch.load(cfg.TEST.WEIGHT_21)['model_state_dict'])
        # models[12].load_state_dict(torch.load(cfg.TEST.WEIGHT_22)['model_state_dict'])
        # models[13].load_state_dict(torch.load(cfg.TEST.WEIGHT_23)['model_state_dict'])
        # models[14].load_state_dict(torch.load(cfg.TEST.WEIGHT_24)['model_state_dict'])
        # models[15].load_state_dict(torch.load(cfg.TEST.WEIGHT_30)['model_state_dict'])
        # models[16].load_state_dict(torch.load(cfg.TEST.WEIGHT_31)['model_state_dict'])
        # models[17].load_state_dict(torch.load(cfg.TEST.WEIGHT_32)['model_state_dict'])
        # models[18].load_state_dict(torch.load(cfg.TEST.WEIGHT_33)['model_state_dict'])
        # models[19].load_state_dict(torch.load(cfg.TEST.WEIGHT_34)['model_state_dict'])
        # models[20].load_state_dict(torch.load(cfg.TEST.WEIGHT_40)['model_state_dict'])
        # models[21].load_state_dict(torch.load(cfg.TEST.WEIGHT_41)['model_state_dict'])
        # models[22].load_state_dict(torch.load(cfg.TEST.WEIGHT_42)['model_state_dict'])
        # models[23].load_state_dict(torch.load(cfg.TEST.WEIGHT_43)['model_state_dict'])
        # models[24].load_state_dict(torch.load(cfg.TEST.WEIGHT_44)['model_state_dict'])
    else:
        device = 'cpu'
        models[0].load_state_dict(torch.load(cfg.TEST.WEIGHT_00, map_location=device)['model_state_dict'])
        models[1].load_state_dict(torch.load(cfg.TEST.WEIGHT_01, map_location=device)['model_state_dict'])
        models[2].load_state_dict(torch.load(cfg.TEST.WEIGHT_02, map_location=device)['model_state_dict'])
        models[3].load_state_dict(torch.load(cfg.TEST.WEIGHT_03, map_location=device)['model_state_dict'])
        models[4].load_state_dict(torch.load(cfg.TEST.WEIGHT_04, map_location=device)['model_state_dict'])
        # models[5].load_state_dict(torch.load(cfg.TEST.WEIGHT_10, map_location=device)['model_state_dict'])
        # models[6].load_state_dict(torch.load(cfg.TEST.WEIGHT_11, map_location=device)['model_state_dict'])
        # models[7].load_state_dict(torch.load(cfg.TEST.WEIGHT_12, map_location=device)['model_state_dict'])
        # models[8].load_state_dict(torch.load(cfg.TEST.WEIGHT_13, map_location=device)['model_state_dict'])
        # models[9].load_state_dict(torch.load(cfg.TEST.WEIGHT_14, map_location=device)['model_state_dict'])
        # models[10].load_state_dict(torch.load(cfg.TEST.WEIGHT_20, map_location=device)['model_state_dict'])
        # models[11].load_state_dict(torch.load(cfg.TEST.WEIGHT_21, map_location=device)['model_state_dict'])
        # models[12].load_state_dict(torch.load(cfg.TEST.WEIGHT_22, map_location=device)['model_state_dict'])
        # models[13].load_state_dict(torch.load(cfg.TEST.WEIGHT_23, map_location=device)['model_state_dict'])
        # models[14].load_state_dict(torch.load(cfg.TEST.WEIGHT_24, map_location=device)['model_state_dict'])
        # models[15].load_state_dict(torch.load(cfg.TEST.WEIGHT_30, map_location=device)['model_state_dict'])
        # models[16].load_state_dict(torch.load(cfg.TEST.WEIGHT_31, map_location=device)['model_state_dict'])
        # models[17].load_state_dict(torch.load(cfg.TEST.WEIGHT_32, map_location=device)['model_state_dict'])
        # models[18].load_state_dict(torch.load(cfg.TEST.WEIGHT_33, map_location=device)['model_state_dict'])
        # models[19].load_state_dict(torch.load(cfg.TEST.WEIGHT_34, map_location=device)['model_state_dict'])
        # models[20].load_state_dict(torch.load(cfg.TEST.WEIGHT_40, map_location=device)['model_state_dict'])
        # models[21].load_state_dict(torch.load(cfg.TEST.WEIGHT_41, map_location=device)['model_state_dict'])
        # models[22].load_state_dict(torch.load(cfg.TEST.WEIGHT_42, map_location=device)['model_state_dict'])
        # models[23].load_state_dict(torch.load(cfg.TEST.WEIGHT_43, map_location=device)['model_state_dict'])
        # models[24].load_state_dict(torch.load(cfg.TEST.WEIGHT_44, map_location=device)['model_state_dict'])

    test_loader = make_test_data_loader(cfg)

    predicter = Predicter(models=models, device=device, cfg=cfg, test_loader=test_loader)
    predicter.predict()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    result_dir = cfg.RESULT_DIR
    if result_dir and not os.path.exists(result_dir):
        mkdir(result_dir)
    predict(cfg)


if __name__ == '__main__':
    main()
