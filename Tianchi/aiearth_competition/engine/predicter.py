# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import time
import numpy as np
import warnings
from datetime import datetime
import torch
from .average import AverageMeter
from evaluate.evaluate import evaluate
from tqdm import tqdm
import pandas as pd
import zipfile
from solver.build import make_optimizer
from solver.lr_scheduler import make_scheduler
warnings.filterwarnings("ignore")

class Predicter:
    def __init__(self, models, device, cfg, test_loader):
        self.config = cfg
        self.test_loader = test_loader

        self.base_dir = f'{self.config.RESULT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.result_path = f'{self.config.RESULT_PATH}'

        self.models = models
        self.device = device
        for model in self.models:
            model.to(self.device)

    def predict(self):
        for model in self.models:
            model.eval()
        t = time.time()
        test_predicts_dict = {}
        test_loader = tqdm(self.test_loader, total=len(self.test_loader), desc="Validating")
        with torch.no_grad():
            for step, ((sst, t300, ua, va), name) in enumerate(test_loader):
                sst = sst.to(self.device).float()
                t300 = t300.to(self.device).float()
                ua = ua.to(self.device).float()
                va = va.to(self.device).float()
                outputs = []
                for i in range(5):
                    outputs.append(torch.unsqueeze(self.models[i]((sst, t300, ua, va)), dim=2))
                outputs = torch.cat(outputs, dim=2)
                outputs = torch.mean(outputs, dim=2)
                for i in range(len(name)):
                    test_predicts_dict[name[i]] = outputs[i].reshape(-1, )
                test_loader.set_description(f'Test Step {step}/{len(self.test_loader)}, ' + \
                                             f'time: {(time.time() - t):.5f}')
        for file_name, val in test_predicts_dict.items():
            np.save(self.base_dir + file_name, val.cpu().detach().numpy())
            val = np.load(self.base_dir+file_name)
            print("----type----")
            print(type(val))
            print("----shape----")
            print(val.shape)
            print("----data----")
            print(val)
        self.make_zip(self.base_dir, self.result_path)

    # 打包目录为zip文件（未压缩）
    def make_zip(self, source_dir, output_filename):
        zipf = zipfile.ZipFile(output_filename, 'w')
        pre_len = len(os.path.dirname(source_dir))
        source_dirs = os.walk(source_dir)
        print(source_dirs)
        for parent, dirnames, filenames in source_dirs:
            print(parent, dirnames)
            for filename in filenames:
                if '.npy' not in filename:
                    continue
                pathfile = os.path.join(parent, filename)
                arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
                zipf.write(pathfile, arcname)
        zipf.close()
