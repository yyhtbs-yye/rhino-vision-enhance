import math
import torch
import torch.nn as nn

ACTI = nn.ReLU

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = ACTI(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class DownsampleBlockNoBN(nn.Module):
    """Simple strided-convolution downsample block with ReLU, no BN."""

    def __init__(self, in_channels, out_channels=None, stride=2):
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        out_channels = in_channels if out_channels is None else out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=True)
        self.act = ACTI(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

class UpsampleBlockNoBN(nn.Module):
    """PixelShuffle based upsample block with ReLU, no BN."""

    def __init__(self, in_channels, out_channels=None, scale_factor=2):
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        out_channels = in_channels if out_channels is None else out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2),
                              kernel_size=3, stride=1, padding=1, bias=True)
        self.pix = nn.PixelShuffle(scale_factor)
        self.act = ACTI(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pix(x)
        return self.act(x)

class Encoder(nn.Module):
    """Convolutional encoder used to patchify spatial features."""

    def __init__(self, in_channels, num_levels=3, num_res_blocks=2):
        super().__init__()

        layers = []
        curr_channels = in_channels

        for _ in range(num_levels):
            for _ in range(num_res_blocks):
                layers.append(ResidualBlockNoBN(curr_channels))
            layers.append(DownsampleBlockNoBN(curr_channels, curr_channels*2, stride=2))
            curr_channels = curr_channels * 2

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """Convolutional decoder used to unpatchify latent features."""

    def __init__(self, in_channels, num_levels=3, num_res_blocks=2):
        super().__init__()

        layers = []
        curr_channels = in_channels

        for _ in range(num_levels):
            layers.append(UpsampleBlockNoBN(curr_channels, curr_channels//2, scale_factor=2))
            for _ in range(num_res_blocks):
                layers.append(ResidualBlockNoBN(curr_channels//2))
            curr_channels = curr_channels // 2

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Quick sanity check to visualize channel changes through encoder/decoder.
    b, c, h, w = 1, 96, 64, 64
    patch_size = 4
    latent_dim = 192

    encoder = Encoder(in_channels=c, num_levels=3, num_res_blocks=1)
    decoder = Decoder(in_channels=c*8, num_levels=3, num_res_blocks=1)

    x = torch.randn(b, c, h, w)
    z = encoder(x)
    y = decoder(z)

    print(f"Input: {tuple(x.shape)}")
    print(f"Encoded: {tuple(z.shape)} (channels: {z.size(1)})")
    print(f"Decoded: {tuple(y.shape)} (channels: {y.size(1)})")
