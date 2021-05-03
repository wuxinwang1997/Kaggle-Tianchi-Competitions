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
from solver.build import make_optimizer
from solver.lr_scheduler import make_scheduler
warnings.filterwarnings("ignore")

class Fitter:
    def __init__(self, model, device, cfg, train_loader, val_loader, logger):
        self.config = cfg
        self.epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.base_dir = f'{self.config.OUTPUT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.logger = logger
        self.best_final_loss = 9999.0

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.loss = torch.nn.MSELoss(reduce=True, size_average=True)
        self.optimizer = make_optimizer(cfg, model)

        self.scheduler = make_scheduler(cfg, self.optimizer, train_loader)

        self.logger.info(f'Fitter prepared. Device is {self.device}')
        self.early_stop_epochs = 0
        self.early_stop_patience = self.config.SOLVER.EARLY_STOP_PATIENCE
        self.do_scheduler = True
        self.logger.info("Start training")

    def fit(self):
        for epoch in range(self.epoch, self.config.SOLVER.MAX_EPOCHS ):
            if epoch < self.config.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(1., float(epoch + 1) / float(self.config.SOLVER.WARMUP_EPOCHS))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr_scale * self.config.SOLVER.BASE_LR
                self.do_scheduler = False
            else:
                self.do_scheduler = True
            if self.config.VERBOSE:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.logger.info(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch()

            self.logger.info(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            score, val_loss = self.validation()

            if val_loss.avg < self.best_final_loss:
                self.best_final_loss = val_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint.bin')
                self.save_model(f'{self.base_dir}/best-model.bin')
            self.logger.info( f'[RESULT]: Val. Epoch: {self.epoch}, Val loss: {val_loss.avg:.5f}, Best Val loss: {self.best_final_loss:.5f}, Score: {score:.5f}, time: {(time.time() - t):.5f}')
            self.early_stop(val_loss.avg)
            if self.early_stop_epochs > self.config.SOLVER.EARLY_STOP_PATIENCE:
                self.logger.info('Early Stopping!')
                break

            self.epoch += 1

    def validation(self):
        self.model.eval()
        t = time.time()
        y_true = []
        y_pred = []
        summary_loss = AverageMeter()
        valid_loader = tqdm(self.val_loader, total=len(self.val_loader), desc="Validating")
        with torch.no_grad():
            for step, ((sst, t300, ua, va), labels) in enumerate(valid_loader):
                sst = sst.to(self.device).float()
                t300 = t300.to(self.device).float()
                ua = ua.to(self.device).float()
                va = va.to(self.device).float()
                labels = labels.to(self.device).float()
                outputs = self.model((sst, t300, ua, va))
                loss = self.loss(outputs, labels)
                summary_loss.update(loss.item(), sst.shape[0])
                y_pred.append(outputs)
                y_true.append(labels)
                valid_loader.set_description(f'Validate Step {step}/{len(self.val_loader)}, ' + \
                                             f'summary_loss: {summary_loss.avg:.5f}, ' + \
                                             f'time: {(time.time() - t):.5f}')
        y_true = torch.cat(y_true, axis=0)
        y_pred = torch.cat(y_pred, axis=0)
        score = evaluate(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        return score, summary_loss

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc="Training")
        for step, ((sst, t300, ua, va), labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            sst = sst.to(self.device).float()
            t300 = t300.to(self.device).float()
            ua = ua.to(self.device).float()
            va = va.to(self.device).float()
            labels = labels.to(self.device).float()
            outputs = self.model((sst, t300, ua, va))
            loss = self.loss(outputs, labels)
            loss.backward()

            summary_loss.update(loss.item(), sst.shape[0])
            self.optimizer.step()

            if self.do_scheduler:
                self.scheduler.step()
            train_loader.set_description(f'Train Step {step}/{len(self.train_loader)}, ' + \
                                         f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                                         f'summary_loss: {summary_loss.avg:.5f}, ' + \
                                         f'time: {(time.time() - t):.5f}')

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_final_loss': self.best_final_loss,
            'epoch': self.epoch,
        }, path)

    def save_model(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_final_loss': self.best_final_loss,
        }, path)

    def save_predictions(self, path):
        df = pd.DataFrame(self.all_predictions)
        df.to_csv(path, index=False)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_final_loss = checkpoint['best_final_loss']
        self.epoch = checkpoint['epoch'] + 1

    def early_stop(self, loss):
        if loss > self.best_final_loss:
            self.early_stop_epochs += 1
        else:
            self.early_stop_epochs = 0
