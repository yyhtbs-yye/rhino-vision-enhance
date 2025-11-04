import torch
import torch.nn as nn
from rhcore.nn.ops.channel_layer_norm import ChannelLayerNorm2d

# ----------------------------
# Minimal norm/activation makers
# ----------------------------
def _make_norm(norm, num_channels):
    if norm is None or norm == "none":
        return nn.Identity()
    if isinstance(norm, nn.Module):
        return norm
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm == "ln":
        return ChannelLayerNorm2d(num_channels)
    if norm == "gn":
        groups = 32 if num_channels % 32 == 0 else 1
        return nn.GroupNorm(groups, num_channels)
    return nn.Identity()

def _make_act(act):
    if act is None or act == "none":
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "mish":
        return nn.Mish()
    if act == "prelu":
        return nn.PReLU()
    return nn.Identity()

# ----------------------------
# Core blocks
# ----------------------------
class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, bias=True, norm="ln", act="relu"):
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=pad, dilation=d, bias=bias)
        self.norm = _make_norm(norm, out_ch)
        self.act  = _make_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, k=3, d=1, norm="ln", act="relu", res_scale=1.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(ch, ch, k=k, d=d, norm=norm, act=act),
            ConvNormAct(ch, ch, k=k, d=d, norm=norm, act="none"),
        )
        self.act = _make_act(act)
        self.res_scale = res_scale

    def forward(self, x):
        return self.act(x + self.block(x) * self.res_scale)

# ----------------------------
# UNet-style (no resizing)
# ----------------------------
class UNetStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, n_convs=1, k=3, norm="ln", act="relu"):
        super().__init__()
        layers = []
        ch_in = in_ch
        for i in range(n_convs):
            layers.append(ConvNormAct(ch_in, out_ch, k=k, d=1, norm=norm, act=act))
            ch_in = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.net(x)

class UNetBodyNoResize(nn.Module):
    def __init__(self, channels=64, depth=3, num_blocks_per_level=2, dilations=None, norm="ln", act="relu"):
        super().__init__()
        self.channels = channels
        self.depth = max(2, depth)
        if not isinstance(dilations, list) or len(dilations) != self.depth:
            dilations = [1] * self.depth

        self.encoder = nn.ModuleList()
        for lvl in range(self.depth):
            blocks = [ResBlock(channels, d=dilations[lvl], norm=norm, act=act) for _ in range(num_blocks_per_level)]
            self.encoder.append(nn.Sequential(*blocks))

        self.bottleneck = nn.Sequential(
            ResBlock(channels, d=max(dilations), norm=norm, act=act),
            ResBlock(channels, d=max(dilations), norm=norm, act=act),
        )

        self.fusions = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(self.depth - 1):
            self.fusions.append(ConvNormAct(channels * 2, channels, k=1, d=1, norm=norm, act=act))
            blocks = [ResBlock(channels, d=1, norm=norm, act=act) for _ in range(num_blocks_per_level)]
            self.decoder.append(nn.Sequential(*blocks))

    def forward(self, x, **kwargs):
        skips = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)

        h = self.bottleneck(h)

        for i in range(self.depth - 2, -1, -1):
            h = torch.cat([h, skips[i]], dim=1)
            idx = self.depth - 2 - i
            h = self.fusions[idx](h)
            h = self.decoder[idx](h)

        return h

class UNetHead(nn.Module):
    def __init__(self, in_ch=64, out_ch=3, k=3, norm="none", act="none", residual=True):
        super().__init__()
        self.proj = ConvNormAct(in_ch, out_ch, k=k, d=1, norm=norm, act=act)
        self.residual = residual

    def forward(self, x, **kwargs):
        y = self.proj(x)
        if self.residual:
            src = kwargs.get("inp", kwargs.get("residual_input", None))
            if src is not None:
                y = y + src
        return y
