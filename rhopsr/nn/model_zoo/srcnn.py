
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels):
        super().__init__()

        # Valid convolutions (no padding) as in the original.
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(base_channels, base_channels // 2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(base_channels // 2, out_channels, kernel_size=5, padding=2)

        # Original uses ReLU between layers
        self.acti = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        # Optional: weight init like many repros (paper used random Gaussian; this is a common sensible default)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv2.bias)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = self.acti(self.conv1(x))
        x = self.acti(self.conv2(x))
        x = self.conv3(x)
        return x

