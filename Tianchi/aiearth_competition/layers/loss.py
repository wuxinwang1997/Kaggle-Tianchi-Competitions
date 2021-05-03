import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error
import numpy as np

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def coreff(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        c1 = torch.sum((x - x_mean) * (y - y_mean))
        c2 = torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2)
        return c1 / torch.sqrt(c2)
    
    def rmse(self, label, preds):
        return torch.sqrt(torch.sum((label - preds) ** 2) / preds.shape[0])

    def score(self, preds, label):
        # preds = preds.cpu().detach().numpy().squeeze()
        # label = label.cpu().detach().numpy().squeeze()
        wmse = 0
        a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
        logs = np.log(np.arange(1, 25))
        for i in range(24):
            wmse += a[i] * logs[i] * self.rmse(label[:,i], preds[:,i]) / np.sum(a * logs)

        return wmse

    def forward(self, x, y):
        return self.score(x, y)
