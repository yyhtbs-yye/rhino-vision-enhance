import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic building blocks
# -----------------------------
def default_init_conv(conv: nn.Conv2d):
    # Kaiming normal keeps variance stable for deep nets; bias = 0
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)

class ResidualBlockNoBN(nn.Module):
    """EDSR residual block: Conv-ReLU-Conv, no BN, with residual scaling."""
    def __init__(self, n_feats: int, res_scale: float = 0.1, act=nn.ReLU(True)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            act,
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )
        self.res_scale = res_scale

        # init
        for m in self.body:
            if isinstance(m, nn.Conv2d):
                default_init_conv(m)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        return x + res

class Upsampler(nn.Module):
    """PixelShuffle upsampler used in EDSR. If scale==1 it's identity."""
    def __init__(self, scale: int, n_feats: int):
        super().__init__()
        layers = []
        if scale == 1:
            # identity
            pass
        elif scale in (2, 3):
            layers += [
                nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, 1, 1),
                nn.PixelShuffle(scale),
            ]
        elif scale in (4, 8):  # stack x2
            n = int(torch.log2(torch.tensor(scale)))
            for _ in range(n):
                layers += [
                    nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                ]
        else:
            raise ValueError(f"Unsupported scale: {scale}")

        self.body = nn.Sequential(*layers)
        for m in self.body:
            if isinstance(m, nn.Conv2d):
                default_init_conv(m)

    def forward(self, x):
        if len(self.body) == 0:
            return x
        return self.body(x)

# -----------------------------
# EDSR (Enhanced Deep SR)
# -----------------------------
class EDSR(nn.Module):
    """
    EDSR backbone.
    - For deblurring/same-resolution tasks, set scale=1.
    - For super-resolution, set scale=2/3/4 and provide labels of corresponding size.

    Args:
        in_channels: Input channels (RGB=3, grayscale=1)
        out_channels: Output channels (usually same as input)
        base_channels: Backbone channels (paper uses 256, 64/128 is common for efficiency)
        num_blocks: Number of residual blocks (paper uses 32, 16/32 is common)
        res_scale: Residual scaling for training stability (0.1 is classic config)
        scale: Upscaling factor (1 means no upscaling)
        act: Activation function (EDSR uses ReLU)
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 num_blocks: int = 16,
                 res_scale: float = 0.1,
                 scale: int = 1,
                 act=nn.ReLU(True)):
        super().__init__()

        # head
        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        default_init_conv(self.head)

        # body: N residual blocks + a conv
        blocks = [ResidualBlockNoBN(base_channels, res_scale=res_scale, act=act)
                  for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks, nn.Conv2d(base_channels, base_channels, 3, 1, 1))
        default_init_conv(self.body[-1])

        # upsampler (identity if scale=1)
        self.upsampler = Upsampler(scale=scale, n_feats=base_channels)

        # tail
        self.tail = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        default_init_conv(self.tail)

    def forward(self, x):
        # long skip connection over the residual trunk
        feat = self.head(x)
        res = self.body(feat)
        feat = feat + res  # long skip

        feat = self.upsampler(feat)  # no-op if scale=1
        out = self.tail(feat)
        return out
