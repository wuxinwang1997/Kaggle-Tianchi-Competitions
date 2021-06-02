import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import model_selection

from configs.effnet_fold3 import CFG
from src.utils import init_logger, get_device, set_seed
from src.models import PANNsEfnetAtt
from src.dataset import PANNsMultiLabelDataset, TARGET_COLUMNS
from src.criterion import ImprovedPANNsLoss
from src.apis import train_one_epoch, eval_one_epoch, calc_metrics

"""
used by last year winner solution
self.model_config =  {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        }
"""


# Custom optimizer
__OPTIMIZERS__ = {}


def get_optimizer(model: nn.Module):
    optimizer_name = CFG.optimizer_name
    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **CFG.optimizer_params)
    else:
        return torch.optim.__getattribute__(optimizer_name)(model.parameters(),
                                                            **CFG.optimizer_params)


def get_scheduler(optimizer):
    scheduler_name = CFG.scheduler_name

    if scheduler_name is None:
        return
    else:
        return torch.optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **CFG.scheduler_params)


if __name__ == '__main__':

    logdir = Path("effnet_fold3")
    logdir.mkdir(exist_ok=True, parents=True)
    if (logdir / "train.log").exists():
        os.remove(logdir / "train.log")
    logger = init_logger(log_file=logdir / "train.log")

    # environment
    set_seed(CFG.seed)
    device = get_device()

    # validation
    splitter = getattr(model_selection, CFG.split)(**CFG.split_params)

    # data
    train = pd.read_csv(CFG.train_csv)

    # main loop
    for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
        if i not in CFG.folds:
            continue
        logger.info("=" * 120)
        logger.info(f"Fold {i} Training")
        logger.info("=" * 120)

        best_score = 0
        total_early_stop = 8
        early_stop = 0
        train_metrics = {}
        eval_metrics = {}


        trn_df = train.loc[trn_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: torch.utils.data.DataLoader(
                PANNsMultiLabelDataset(
                    df_,
                    CFG.train_datadir,
                    transforms=CFG.transforms if phase == "train" else None,
                    period=CFG.period,
                ),
                **CFG.loader_params[phase])
            for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
        }

        model = PANNsEfnetAtt(**CFG.model_config).to(device)
        #model = torch.nn.DataParallel(model).to(device)
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)
        criterion = ImprovedPANNsLoss(
            output_key='clipwise_output', weights=[1.0, 0.5]).to(device)

        for epoch in range(CFG.epochs):
            logger.info(f"############# Epoch{epoch+1} ##############")
            # logger.info("#"*20, epoch + 1, "Epoch", "#"*20)
            avg_train_loss, train_y_pred, train_y_true = train_one_epoch(
                model=model,
                dataloader=loaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                input_key="waveform",
                input_target_key="targets"
            )

            # step the scheduler
            scheduler.step()

            #train_y_true = (train_y_true > 0.5)*1.0
            train_mAP, train_classwise_f1, train_sample_f1 = calc_metrics(
                train_y_true, train_y_pred)
            torch.cuda.empty_cache()
            train_metrics["loss"] = avg_train_loss
            train_metrics["mAP"] = train_mAP
            train_metrics["classwise_f1"] = train_classwise_f1
            train_metrics["sample_f1"] = train_sample_f1

            logger.info("#" * 20)
            logger.info("Train metrics")
            for key, value in train_metrics.items():
                logger.info(f"{key}: {value:.5f}")
            del train_y_pred, train_y_true

            with torch.no_grad():
                avg_loss, y_pred, y_true = eval_one_epoch(
                    model=model,
                    dataloader=loaders["valid"],
                    criterion=criterion,
                    device=device,
                    input_key="waveform",
                    input_target_key="targets"
                )
                #y_true = (y_true > 0.5) * 1.0
                mAP, classwise_f1, sample_f1 = calc_metrics(y_true, y_pred)
                torch.cuda.empty_cache()
                eval_metrics["loss"] = avg_loss
                eval_metrics["mAP"] = mAP
                eval_metrics["classwise_f1"] = classwise_f1
                eval_metrics["sample_f1"] = sample_f1
                del y_true, y_pred

                logger.info("Saving model...")
                torch.save(model.state_dict(), f"last_epoch_model_{i}.pth")

                logger.info("Valid metrics")
                for key, value in eval_metrics.items():
                    logger.info(f"{key}: {value:.5f}")

                if sample_f1 > best_score:
                    best_score = sample_f1
                    early_stop = 0
                    logger.info("Saving the best model...")
                    torch.save(model.state_dict(), f"best_model_{i}.pth")

                if sample_f1 < best_score:
                    early_stop += 1
                if total_early_stop <= early_stop:
                    break
