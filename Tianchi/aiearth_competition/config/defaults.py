# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import sys
import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEBUG = False
_C.SEED = 66
_C.VERBOSE = True

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.PRETRAINED_CMIP = "../usr_data/model_data/resnet-lr1e4-sst-epoch30-cmip/best-model.bin"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.DEPTH = 18
_C.MODEL.BACKBONE.LAST_STRIDE = 2
_C.MODEL.BACKBONE.NORM = "BN"
# Mini-batch split of Ghost BN
_C.MODEL.BACKBONE.NORM_SPLIT = 2
_C.MODEL.BACKBONE.RATIO = 16
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = True
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = True
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_CB = False
_C.MODEL.BACKBONE.WITH_NL = False
_C.MODEL.BACKBONE.PRETRAIN = False
_C.MODEL.BACKBONE.PRETRAIN_PATH = '../external_data/resnet18.pth'
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root dir of dataset
_C.DATASETS.SODA = False
_C.DATASETS.ROOT_DIR = "../tcdata/enso_round1_train_20210201/"
_C.DATASETS.TEST_DIR = "../tcdata/enso_final_test_data_B/"
# Fold to validate
_C.DATASETS.VAL_FOLD = 0
_C.DATASETS.X_DIM = 72
_C.DATASETS.Y_DIM = 24
_C.DATASETS.Z_DIM = 48

# Upscale ratio
_C.DATASETS.UP_RATIO = 1

# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# # List of the dataset names for testing, as present in paths_catalog.py
# _C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.SCHEDULER_NAME = "CosineAnnealingWarmRestarts"
_C.SOLVER.COS_EPOCH = 35
_C.SOLVER.T_MUL = 1

_C.SOLVER.MAX_EPOCHS = 40

_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY_BN = 0

_C.SOLVER.WARMUP_EPOCHS = 5

_C.SOLVER.EARLY_STOP_PATIENCE = 5

_C.SOLVER.TRAIN_SODA = False
_C.SOLVER.TRAIN_CHECKPOINT = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.WEIGHT_00 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold00/best-model.bin"
_C.TEST.WEIGHT_01 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold01/best-model.bin"
_C.TEST.WEIGHT_02 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold02/best-model.bin"
_C.TEST.WEIGHT_03 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold03/best-model.bin"
_C.TEST.WEIGHT_04 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold04/best-model.bin"
_C.TEST.WEIGHT_10 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold10/best-model.bin"
_C.TEST.WEIGHT_11 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold11/best-model.bin"
_C.TEST.WEIGHT_12 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold12/best-model.bin"
_C.TEST.WEIGHT_13 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold13/best-model.bin"
_C.TEST.WEIGHT_14 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold14/best-model.bin"
_C.TEST.WEIGHT_20 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold20/best-model.bin"
_C.TEST.WEIGHT_21 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold21/best-model.bin"
_C.TEST.WEIGHT_22 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold22/best-model.bin"
_C.TEST.WEIGHT_23 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold23/best-model.bin"
_C.TEST.WEIGHT_24 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold24/best-model.bin"
_C.TEST.WEIGHT_30 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold30/best-model.bin"
_C.TEST.WEIGHT_31 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold31/best-model.bin"
_C.TEST.WEIGHT_32 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold32/best-model.bin"
_C.TEST.WEIGHT_33 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold33/best-model.bin"
_C.TEST.WEIGHT_34 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold34/best-model.bin"
_C.TEST.WEIGHT_40 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold40/best-model.bin"
_C.TEST.WEIGHT_41 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold41/best-model.bin"
_C.TEST.WEIGHT_42 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold42/best-model.bin"
_C.TEST.WEIGHT_43 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold43/best-model.bin"
_C.TEST.WEIGHT_44 = "../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold44/best-model.bin"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "../usr_data/model_data/resnet18_lstm-lr1e4-epoch30-soda/"
_C.RESULT_DIR = "../result/"
_C.RESULT_PATH = "../result.zip"
