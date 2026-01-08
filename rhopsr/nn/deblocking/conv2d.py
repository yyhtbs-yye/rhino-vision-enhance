import torch
import torch.nn as nn

class BasicDeblocking(nn.Module):
    def __init__(self, dim, mode='1conv'):
        super().__init__()
        if mode == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif mode == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), 
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x):
        return self.conv(x)