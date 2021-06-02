from sklearn.metrics import f1_score, average_precision_score, label_ranking_average_precision_score
from  sklearn.model_selection  import StratifiedKFold
import numpy as np
from fastprogress import progress_bar
from ignite.utils import convert_tensor

def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    criterion,
                    device,
                    input_key="image",
                    input_target_key="targets"):
    avg_loss = 0.0
    model.train()
    preds = []
    targs = []
    for step, batch in enumerate(progress_bar(dataloader)):

        x = batch[input_key].to(device)
        y = batch[input_target_key].to(device).float()

        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(dataloader)

        clipwise_output = outputs["clipwise_output"].detach().cpu().numpy()
        target = y.detach().cpu().numpy()

        preds.append(clipwise_output)
        targs.append(target)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)
    return avg_loss, y_pred, y_true


def eval_one_epoch(model,
                   dataloader,
                   criterion,
                   device,
                   input_key="image",
                   input_target_key="targets"):
    avg_loss = 0.0
    model.eval()
    preds = []
    targs = []
    for step, batch in enumerate(progress_bar(dataloader)):
        x = batch[input_key].to(device)
        y = batch[input_target_key].to(device).float()

        outputs = model(x)
        #logits = outputs["clipwise_output"]
        loss = criterion(outputs, y).detach()

        avg_loss += loss.item() / len(dataloader)

        clipwise_output = outputs["clipwise_output"].detach().cpu().numpy()
        target = y.detach().cpu().numpy()

        preds.append(clipwise_output)
        targs.append(target)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)
    return avg_loss, y_pred, y_true


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold=0.5):
    mAP = average_precision_score(y_true, y_pred, average=None)
    mAP = np.nan_to_num(mAP).mean()

    classwise_f1s = []
    for i in range(len(y_true[0])):
        class_i_pred = y_pred[:, i] > threshold
        class_i_targ = y_true[:, i]
        if class_i_targ.sum() == 0 and class_i_pred.sum() == 0:
            classwise_f1s.append(1.0)
        else:
            classwise_f1s.append(
                f1_score(y_true=class_i_targ, y_pred=class_i_pred))

    classwise_f1 = np.mean(classwise_f1s)

    y_pred_thresholded = (y_pred > threshold).astype(int)
    sample_f1 = f1_score(
        y_true=y_true, y_pred=y_pred_thresholded, average="samples")
    return mAP, classwise_f1, sample_f1
