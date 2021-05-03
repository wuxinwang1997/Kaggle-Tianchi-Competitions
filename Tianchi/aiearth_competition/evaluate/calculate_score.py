# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
from sklearn.metrics import mean_squared_error
import numpy as np

def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean)**2) * sum((y - y_mean)**2)
    return c1/np.sqrt(c2)

def rmse(preds, y):
    return np.sqrt(sum((preds - y)**2)/preds.shape[0])

def score(preds, label):
    # preds = preds.cpu().detach().numpy().squeeze()
    # label = label.cpu().detach().numpy().squeeze()
    acskill = 0
    RMSE = 0
    a = 0
    a = [1.5]*4 + [2]*7 + [3]*7 + [4]*6
    for i in range(24):
        RMSE += rmse(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])
                     
        acskill += a[i] * np.log(i+1) * cor
    return 2/3 * acskill - RMSE

# def rmse(y_true, y_preds):
#     return np.sqrt(mean_squared_error(y_pred=y_preds, y_true=y_true))


# def score(y_true, y_preds):
#     accskill_score = 0
#     rmse_scores = 0
#     a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
#     y_true_mean = np.mean(y_true, axis=0)
#     y_pred_mean = np.mean(y_preds, axis=0)
    #     print(y_true_mean.shape, y_pred_mean.shape)

#     for i in range(24):
#         fenzi = np.sum((y_true[:, i] - y_true_mean[i]) * (y_preds[:, i] - y_pred_mean[i]))
#         fenmu = np.sqrt(np.sum((y_true[:, i] - y_true_mean[i]) ** 2) * np.sum((y_preds[:, i] - y_pred_mean[i]) ** 2))
#         cor_i = fenzi / fenmu

#         accskill_score += a[i] * np.log(i + 1) * cor_i
#         rmse_score = rmse(y_true[:, i], y_preds[:, i])
        #         print(cor_i,  2 / 3.0 * a[i] * np.log(i+1) * cor_i - rmse_score)
#         rmse_scores += rmse_score

#     return 2 / 3.0 * accskill_score - rmse_scores
