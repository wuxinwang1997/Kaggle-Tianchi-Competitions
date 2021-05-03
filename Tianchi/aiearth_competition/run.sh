#!/bin/sh
CURDIR="`dirname $0`" #获取此脚本所在目录
echo $CURDIR
cd $CURDIR #切换到该脚本所在目录
python ./tools/train_net.py DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip-fold0/")' DATASETS.SODA '(False)' MODEL.BACKBONE.PRETRAIN '(True)' SOLVER.TRAIN_SODA '(False)'
python ./tools/train_net.py DATASETS.VAL_FOLD '(0)' OUTPUT_DIR '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold00/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python ./tools/train_net.py DATASETS.VAL_FOLD '(1)' OUTPUT_DIR '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold01/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python ./tools/train_net.py DATASETS.VAL_FOLD '(2)' OUTPUT_DIR '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold02/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python ./tools/train_net.py DATASETS.VAL_FOLD '(3)' OUTPUT_DIR '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold03/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'
python ./tools/train_net.py DATASETS.VAL_FOLD '(4)' OUTPUT_DIR '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-soda-fold04/")' DATASETS.SODA '(True)' MODEL.PRETRAINED_CMIP '("../usr_data/model_data/resnet18-lr1e4-sst-epoch30-cmip-fold0/best-model.bin")' MODEL.BACKBONE.PRETRAIN '(False)' SOLVER.TRAIN_SODA '(True)' SOLVER.BASE_LR '(1e-4)'

python ./tools/predict.py
