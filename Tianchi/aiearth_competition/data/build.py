# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import socket

import netCDF4 as nc4
import numpy as np
import torch
import torch.utils.data as data
from .datasets.dataset import EarthDataset, TestDataset
from .transforms.build import build_transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def prepare_cmip_data(cfg, cmip_data, cmip_label):
    cmip = dict()
    for name in ['sst', 't300', 'ua', 'va']:
        tmp = np.array(cmip_data[name])
        tmp = np.nan_to_num(tmp)
        tmp = torch.tensor(tmp)
        tmp = torch.flatten(tmp, start_dim=0, end_dim=1)
        cmip[name] = tmp.numpy()
    tmp = np.array(cmip_label[:, :])
    last_year_nino = np.array(cmip_label[-1, -12:].reshape((1, 12)))
    tmp = np.concatenate((tmp, last_year_nino), axis=0)
    cmip['label'] = tmp.flatten()

    dict_cmip = dict()
    for var in ['sst', 't300', 'ua', 'va', 'label']:
        dict_cmip[var] = cmip[var]
    return dict_cmip

def prepare_cmips_data(cfg):
    root_dir = cfg.DATASETS.ROOT_DIR
    cmip_data = nc4.Dataset(root_dir + 'CMIP_train.nc').variables
    cmip_label = nc4.Dataset(root_dir + 'CMIP_label.nc').variables

    cmip6_year = [151 for i in range(15)]
    cmip5_year = [140 for i in range(17)]
    cmip_year = cmip6_year + cmip5_year
    start = 0
    end = cmip_year[0]
    dict_cmips = []
    for i in range(32):
        dict_cmip = prepare_cmip_data(cfg, {'sst': cmip_data['sst'][start:end, 0:12, :, :],
                                            't300': cmip_data['t300'][start:end, 0:12, :, :],
                                            'ua': cmip_data['ua'][start:end, 0:12, :, :],
                                            'va': cmip_data['va'][start:end, 0:12, :, :]},
                                      cmip_label['nino'][start:end, 12:24])
        if i < 31:
            start += cmip_year[i]
            end += cmip_year[i+1]
        dict_cmips.append(dict_cmip)
    return dict_cmips

def prepare_soda_data(cfg):
    root_dir = cfg.DATASETS.ROOT_DIR
    soda_data = nc4.Dataset(root_dir + 'SODA_train.nc').variables
    soda_label = nc4.Dataset(root_dir + 'SODA_label.nc').variables

    soda = dict()
    for var in ['sst', 't300', 'ua', 'va']:
        tmp = np.array(soda_data[var][:, 0:12, :, :])
        tmp = np.nan_to_num(tmp)
        tmp = torch.tensor(tmp)
        tmp = torch.flatten(tmp, start_dim=0, end_dim=1)
        soda[var] = tmp.numpy()

    tmp = np.array(soda_label['nino'][:, 12:24])
    last_year_nino = np.array(soda_label['nino'][-1, -12:].reshape((1, 12)))
    tmp = np.concatenate((tmp, last_year_nino), axis=0)
    soda['label'] = tmp.flatten()

    dict_soda = dict()
    for var in ['sst', 't300', 'ua', 'va', 'label']:
        dict_soda[var] = soda[var]
    return dict_soda

def prepare_test_data(cfg):
    test_path = cfg.DATASETS.TEST_DIR
    files = os.listdir(test_path)
    test_sst = np.zeros((len(files), 12, 24, 72))
    test_t300 = np.zeros((len(files), 12, 24, 72))
    test_ua = np.zeros((len(files), 12, 24, 72))
    test_va = np.zeros((len(files), 12, 24, 72))
    for i in range(len(files)):
        file = np.load(test_path + files[i])
        sst, t300, ua, va = np.split(file, 4, axis=3)
        test_sst[i, :, :, :] = sst.transpose(3, 0, 1, 2)
        test_t300[i, :, :, :] = t300.transpose(3, 0, 1, 2)
        test_ua[i, :, :, :] = ua.transpose(3, 0, 1, 2)
        test_va[i, :, :, :] = va.transpose(3, 0, 1, 2)

    dict_test = {
        'sst': test_sst,
        't300': test_t300,
        'ua': test_ua,
        'va': test_va,
        'name': np.array(files)
    }
    return dict_test


def build_cmip_dataset(cfg, dict_cmip):
    dataset_cmip = EarthDataset(
        data_dict=dict_cmip,
        transforms=build_transforms(cfg, is_train=True),
    )
    if cfg.DEBUG:
        fold_num = 2
        dataset0, dataset1, dataset2, dataset3, dataset4, dataset_other = data.random_split(dataset_cmip,
                                                                                                lengths=[fold_num,
                                                                                                         fold_num,
                                                                                                         fold_num,
                                                                                                         fold_num,
                                                                                                         fold_num,
                                                                                                         len(dataset_cmip) - 5 * fold_num],
                                                                                                generator=torch.Generator().manual_seed(
                                                                                                    cfg.SEED))
        dataset_cmips = [dataset0,dataset1,dataset2,dataset3,dataset4]
    else:
        fold_num = int(len(dataset_cmip) * 0.2)
        dataset0, dataset1, dataset2, dataset3, dataset4 = data.random_split(dataset_cmip,
                                                                lengths=[fold_num, fold_num, fold_num, fold_num, len(dataset_cmip)-4*fold_num],
                                                                generator=torch.Generator().manual_seed(cfg.SEED))
        dataset_cmips = [dataset0,dataset1,dataset2,dataset3,dataset4]
    return dataset_cmips

def build_soda_dataset(cfg):
    dict_soda = prepare_soda_data(cfg)
    dataset_soda = EarthDataset(
        data_dict=dict_soda,
        transforms=build_transforms(cfg, is_train=False),
    )

    if cfg.DEBUG:
        fold_num = 2
        dataset0, dataset1, dataset2, dataset3, dataset4, dataset_other = data.random_split(dataset_soda,
                                                                                 lengths=[fold_num, fold_num, fold_num,
                                                                                          fold_num,
                                                                                          fold_num, dataset_soda.len-5*fold_num],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     cfg.SEED))
    else:
        fold_num = int(dataset_soda.len*0.2)
        dataset0, dataset1, dataset2, dataset3, dataset4 = data.random_split(dataset_soda,
                                                                lengths=[fold_num, fold_num, fold_num, fold_num, dataset_soda.len-4*fold_num],
                                                                generator=torch.Generator().manual_seed(cfg.SEED))
    dataset_soda = [dataset0,dataset1,dataset2,dataset3,dataset4]
    return dataset_soda


def build_test_dataset(cfg):
    dict_test = prepare_test_data(cfg)
    test_dataset = TestDataset(
        data_dict=dict_test,
        transforms=build_transforms(cfg, is_train=False),
    )

    return test_dataset


def make_cmip_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
    val_datasets = []
    train_datasets = []
    dict_cmips = prepare_cmips_data(cfg)
    for i in range(32):
        dataset_cmip = build_cmip_dataset(cfg, dict_cmips[i])
        data_list = [0, 1, 2, 3, 4]
        val_datasets.append(dataset_cmip[cfg.DATASETS.VAL_FOLD])
        data_list.remove(cfg.DATASETS.VAL_FOLD)
        train_datasets.append(data.ConcatDataset([dataset_cmip[i] for i in data_list]))
    val_dataset = data.ConcatDataset(val_datasets)
    train_dataset = data.ConcatDataset(train_datasets)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True)

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, val_loader

def make_soda_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH

    dataset_soda = build_soda_dataset(cfg)
    data_list = [0, 1, 2, 3, 4]
    val_dataset = dataset_soda[cfg.DATASETS.VAL_FOLD]
    data_list.remove(cfg.DATASETS.VAL_FOLD)
    train_dataset = data.ConcatDataset([dataset_soda[i] for i in data_list])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True)

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, val_loader

def make_data_loader(cfg, is_train=True):
    if cfg.DATASETS.SODA:
        return make_soda_loader(cfg, is_train)
    else:
        return make_cmip_loader(cfg, is_train)

def make_test_data_loader(cfg):
    batch_size = cfg.TEST.IMS_PER_BATCH

    test_dataset = build_test_dataset(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False)

    return test_loader
