import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def init_patchgan_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and module.affine:
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def _build_norm(num_channels: int, norm: str) -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "instance":
        return nn.InstanceNorm2d(num_channels, affine=False, track_running_stats=False)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm '{norm}'. Expected one of: batch, instance, none.")


def _conv2d(*args, use_spectral_norm: bool = False, **kwargs) -> nn.Module:
    conv = nn.Conv2d(*args, **kwargs)
    if use_spectral_norm:
        conv = spectral_norm(conv)
    return conv


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
        norm: str = "none",
        negative_slope: float = 0.2,
        max_channel_multiplier: int = 8,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        use_bias = norm.lower() in {"instance", "none"}

        layers = [
            _conv2d(
                in_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=use_spectral_norm,
            ),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]

        prev_multiplier = 1
        curr_multiplier = 1

        for layer_idx in range(1, num_layers):
            prev_multiplier = curr_multiplier
            curr_multiplier = min(2**layer_idx, max_channel_multiplier)
            out_channels = base_channels * curr_multiplier

            layers.extend(
                [
                    _conv2d(
                        base_channels * prev_multiplier,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                        use_spectral_norm=use_spectral_norm,
                    ),
                    _build_norm(out_channels, norm),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                ]
            )

        prev_multiplier = curr_multiplier
        curr_multiplier = min(2**num_layers, max_channel_multiplier)
        head_channels = base_channels * curr_multiplier

        layers.extend(
            [
                _conv2d(
                    base_channels * prev_multiplier,
                    head_channels,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    use_spectral_norm=use_spectral_norm,
                ),
                _build_norm(head_channels, norm),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                _conv2d(
                    head_channels,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    use_spectral_norm=use_spectral_norm,
                ),
            ]
        )

        self.model = nn.Sequential(*layers)
        self.apply(init_patchgan_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)