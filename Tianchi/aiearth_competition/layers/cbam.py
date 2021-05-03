import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // reduction, in_planes, 1, bias=False)

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_channel = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_channel = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        channel = avg_channel + max_channel
        channel_att = self.sigmoid(channel)
        x = x * channel_att.expand_as(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        spatial_att = self.sigmoid(out)
        x = x * spatial_att.expand_as(x)
        return x