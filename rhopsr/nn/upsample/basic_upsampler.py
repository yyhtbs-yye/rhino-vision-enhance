import math

import torch
import torch.nn as nn

class BasicUpsampler(nn.Module):
    def __init__(self, scale, dim, out_channels=3):
        super().__init__()

        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(dim, 4 * dim, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(dim, 9 * dim, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        
        self.upsample = nn.Sequential(*m)

        self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_after_body(x) + x
        
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x
    