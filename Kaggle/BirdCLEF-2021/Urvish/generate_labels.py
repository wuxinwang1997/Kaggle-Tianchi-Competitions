import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from pathlib import Path
import albumentations
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import model_selection
from fastprogress import progress_bar

from configs.generate_labels_config import CFG
from src.utils import init_logger, get_device, set_seed
from src.models import PANNsDense121Att
from src.dataset import PANNsSedDataset, TARGET_COLUMNS, INV_BIRD_CODE, BIRD_CODE

if __name__ == '__main__':

    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logdir = Path("out")
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
        logger.info(f"Fold {i} Predicting")
        logger.info("=" * 120)

        val_df = train.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: torch.utils.data.DataLoader(
                PANNsSedDataset(
                    df_,
                    CFG.train_datadir,
                    transforms=CFG.transforms,
                    period=CFG.period,
                ),
                **CFG.loader_params[phase])
            for phase, df_ in zip(["valid"], [val_df])
        }

        model = PANNsDense121Att(**CFG.model_config)
        model = torch.nn.DataParallel(model).to(device)
        trained_model = torch.load(f"{CFG.model_dir}/best_model_{i}.pth")
        model.load_state_dict(trained_model)

        model.eval()

        estimated_event_list = []
        for batch in progress_bar(loaders["valid"]):
            waveform = batch["waveform"]
            ebird_code = batch["ebird_code"][0]
            wav_name = batch["wav_name"][0]
            target = batch["targets"].detach().cpu().numpy()[0]
            global_time = 0.0
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)

            batch_size = 32
            whole_size = waveform.size(0)
            if whole_size % batch_size == 0:
                n_iter = whole_size // batch_size
            else:
                n_iter = whole_size // batch_size + 1

            for index in range(n_iter):
                iter_batch = waveform[index * batch_size:(index + 1) * batch_size]
                if iter_batch.ndim == 1:
                    iter_batch = iter_batch.unsqueeze(0)
                iter_batch = iter_batch.to(device)
                with torch.no_grad():
                    prediction = model(iter_batch)
                    framewise_output = prediction["framewise_output"].detach(
                    ).cpu().numpy()

                thresholded = framewise_output >= CFG.threshold
                target_indices = np.argwhere(target).reshape(-1)
                for short_clip in thresholded:
                    for target_idx in target_indices:
                        if short_clip[:, target_idx].mean() == 0:
                            pass
                        else:
                            detected = np.argwhere(
                                short_clip[:, target_idx]).reshape(-1)
                            head_idx = 0
                            tail_idx = 0
                            while True:
                                if (tail_idx + 1 == len(detected)) or (
                                        detected[tail_idx + 1] -
                                        detected[tail_idx] != 1):
                                    onset = 0.01 * detected[head_idx] + global_time
                                    offset = 0.01 * detected[tail_idx] + global_time
                                    estimated_event = {
                                        "filename": wav_name,
                                        "ebird_code": INV_BIRD_CODE[target_idx],
                                        "onset": onset,
                                        "offset": offset
                                    }
                                    estimated_event_list.append(estimated_event)
                                    head_idx = tail_idx + 1
                                    tail_idx = tail_idx + 1
                                    if head_idx > len(detected):
                                        break
                                else:
                                    tail_idx = tail_idx + 1
                    global_time += 5.0

        estimated_event_df = pd.DataFrame(estimated_event_list)
        save_filename = CFG.save_path.replace(".csv", "")
        save_filename += f"_th{CFG.threshold}" + ".csv"
        save_path = output_dir / save_filename
        if save_path.exists():
            event_level_labels = pd.read_csv(save_path)
            estimated_event_df = pd.concat(
                [event_level_labels, estimated_event_df], axis=0,
                sort=False).reset_index(drop=True)
            estimated_event_df.to_csv(save_path, index=False)
        else:
            estimated_event_df.to_csv(save_path, index=False)