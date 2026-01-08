from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class _HaarKernels:
    ll: torch.Tensor  # (2, 2)
    lh: torch.Tensor  # (2, 2)
    hl: torch.Tensor  # (2, 2)
    hh: torch.Tensor  # (2, 2)


def _haar_kernels(device: torch.device, dtype: torch.dtype) -> _HaarKernels:
    # Orthonormal Haar (separable) filters:
    # low = [1, 1] / sqrt(2), high = [1, -1] / sqrt(2)
    # 2D outer products => scale factor (1/sqrt(2))^2 = 1/2.
    scale = 0.5
    ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device, dtype=dtype) * scale
    lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]], device=device, dtype=dtype) * scale
    hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=device, dtype=dtype) * scale
    hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=device, dtype=dtype) * scale
    return _HaarKernels(ll=ll, lh=lh, hl=hl, hh=hh)


def _maybe_flatten_video(x: torch.Tensor) -> tuple[torch.Tensor, int | None]:
    if x.dim() != 5:
        return x, None
    b, t, c, h, w = x.shape
    return x.reshape(b * t, c, h, w), t


def _maybe_expand_weights(weights: torch.Tensor | None, b: int, t: int | None) -> torch.Tensor | None:
    if weights is None or t is None:
        return weights
    if weights.dim() != 1:
        raise ValueError(f"`weights` must be 1D, got {tuple(weights.shape)}")
    if weights.numel() == b * t:   # already per-frame
        return weights
    if weights.numel() == b:       # per-video, expand to per-frame
        return weights.repeat_interleave(t, dim=0)
    raise ValueError(f"`weights` must have length B ({b}) or B*T ({b*t}), got {weights.numel()}")


def _normalize_level_weights(
    level_weights: Sequence[float] | torch.Tensor | None,
) -> list[float] | None:
    if level_weights is None:
        return None
    if isinstance(level_weights, torch.Tensor):
        if level_weights.dim() != 1:
            raise ValueError(
                f"`level_weights` must be 1D, got shape={tuple(level_weights.shape)}"
            )
        weights_list = level_weights.detach().cpu().tolist()
    else:
        if isinstance(level_weights, (int, float)):
            raise ValueError("`level_weights` must be a sequence, not a scalar.")
        weights_list = list(level_weights)
    if not weights_list:
        raise ValueError("`level_weights` must contain at least one value.")
    return [float(w) for w in weights_list]


def _wavelet_decompose_once(
    x: torch.Tensor, *, pad_mode: str = "reflect"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got shape={tuple(x.shape)}")
    if x.size(-2) < 2 or x.size(-1) < 2:
        raise ValueError(
            f"Input too small for Haar decomposition, got HxW={x.size(-2)}x{x.size(-1)}"
        )

    pad_right = int(x.size(-1) % 2)
    pad_bottom = int(x.size(-2) % 2)
    if pad_right or pad_bottom:
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)

    b, c, _, _ = x.shape
    kernels = _haar_kernels(device=x.device, dtype=x.dtype)
    weight = torch.stack([kernels.ll, kernels.lh, kernels.hl, kernels.hh], dim=0).view(4, 1, 2, 2)
    weight = weight.repeat(c, 1, 1, 1)  # (4*C, 1, 2, 2)

    y = F.conv2d(x, weight, stride=2, padding=0, groups=c)  # (B, 4*C, H/2, W/2)
    y = y.view(b, c, 4, y.size(-2), y.size(-1))
    ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
    return ll, lh, hl, hh


def _loss_map(
    preds: torch.Tensor, targets: torch.Tensor, *, base_loss: str, eps: float
) -> torch.Tensor:
    if base_loss == "l1":
        return (preds - targets).abs()
    if base_loss == "mse":
        diff = preds - targets
        return diff * diff
    if base_loss == "charbonnier":
        diff = preds - targets
        return torch.sqrt(diff * diff + eps * eps)
    raise ValueError(f"Unsupported base_loss: {base_loss}")


def _reduce_per_sample(
    loss_map: torch.Tensor, *, reduction: str, weights: torch.Tensor | None
) -> torch.Tensor:
    if loss_map.numel() == 0:
        return loss_map.sum()

    if loss_map.dim() == 0:
        loss_per_sample = loss_map.view(1)
    else:
        loss_per_sample = loss_map.view(loss_map.size(0), -1).mean(dim=1)

    if weights is not None:
        if weights.shape != loss_per_sample.shape:
            raise ValueError(
                f"`weights` shape must match per-sample loss shape {tuple(loss_per_sample.shape)}, "
                f"got {tuple(weights.shape)}"
            )
        loss_per_sample = loss_per_sample * weights

    if reduction == "none":
        return loss_per_sample
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "mean":
        return loss_per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def haar_wavelet_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    levels: int | None = None,
    level_weights: Sequence[float] | torch.Tensor | None = None,
    base_loss: str = "l1",
    eps: float = 1e-3,
    low_freq_levels: str = "final",
    pad_mode: str = "reflect",
    reduction: str = "mean",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multi-level Haar wavelet loss between `preds` and `targets`.

    Decomposes `levels` times on the LL (low-frequency) subband. At each level it
    penalizes high-frequency subbands (LH, HL, HH). Optionally penalizes low-frequency
    (LL) either at all levels or only the final LL.

    Shapes:
      - (B, C, H, W) or (B, T, C, H, W)
      - `weights` (optional): (B,) for images, or (B,) / (B*T,) for videos

    `low_freq_levels`:
      - "none": no LL loss
      - "final": LL loss only on the final LL (default)
      - "all": LL loss at every decomposition level

    If `level_weights` is provided, its length defines the number of levels and
    each entry scales both high/low-frequency terms for that level.
    """
    normalized_level_weights = _normalize_level_weights(level_weights)
    if normalized_level_weights is not None:
        levels = len(normalized_level_weights)
    if levels is None:
        levels = 1
    if levels < 1:
        raise ValueError(f"`levels` must be >= 1, got {levels}")
    if low_freq_levels not in {"none", "final", "all"}:
        raise ValueError(f"Unsupported low_freq_levels: {low_freq_levels}")

    b = preds.shape[0]
    preds_4d, t = _maybe_flatten_video(preds)
    targets_4d, t2 = _maybe_flatten_video(targets)

    if t != t2:
        raise ValueError(f"preds/targets video mismatch: {tuple(preds.shape)} vs {tuple(targets.shape)}")
    weights = _maybe_expand_weights(weights, b=b, t=t)

    if preds_4d.shape != targets_4d.shape:
        raise ValueError(f"preds/targets shape mismatch: {tuple(preds_4d.shape)} vs {tuple(targets_4d.shape)}")
    if preds_4d.dim() != 4:
        raise ValueError(f"Expected 4D or 5D tensors, got preds shape={tuple(preds.shape)}")

    ll_p, ll_t = preds_4d, targets_4d
    total_loss_map = ll_p.new_zeros((ll_p.size(0),), dtype=ll_p.dtype)

    for level in range(levels):
        level_scale = (
            normalized_level_weights[level] if normalized_level_weights is not None else 1.0
        )
        ll_p, lh_p, hl_p, hh_p = _wavelet_decompose_once(ll_p, pad_mode=pad_mode)
        ll_t, lh_t, hl_t, hh_t = _wavelet_decompose_once(ll_t, pad_mode=pad_mode)

        hf_map = (
            _loss_map(lh_p, lh_t, base_loss=base_loss, eps=eps)
            + _loss_map(hl_p, hl_t, base_loss=base_loss, eps=eps)
            + _loss_map(hh_p, hh_t, base_loss=base_loss, eps=eps)
        )
        total_loss_map = total_loss_map + level_scale * hf_map.view(hf_map.size(0), -1).mean(dim=1)

        if low_freq_levels == "all":
            lf_map = _loss_map(ll_p, ll_t, base_loss=base_loss, eps=eps)
            total_loss_map = total_loss_map + level_scale * lf_map.view(lf_map.size(0), -1).mean(dim=1)

        if level == levels - 1 and low_freq_levels == "final":
            lf_map = _loss_map(ll_p, ll_t, base_loss=base_loss, eps=eps)
            total_loss_map = total_loss_map + level_scale * lf_map.view(lf_map.size(0), -1).mean(dim=1)

    # Apply weights/reduction on the already per-sample aggregated loss.
    if weights is not None:
        if weights.shape != total_loss_map.shape:
            raise ValueError(
                f"`weights` shape must match per-sample loss shape {tuple(total_loss_map.shape)}, "
                f"got {tuple(weights.shape)}"
            )
        total_loss_map = total_loss_map * weights

    if reduction == "none":
        return total_loss_map
    if reduction == "sum":
        return total_loss_map.sum()
    if reduction == "mean":
        return total_loss_map.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


class HaarWaveletLoss(nn.Module):
    def __init__(
        self,
        levels: int | None = None,
        level_weights: Sequence[float] | torch.Tensor | None = None,
        base_loss: str = "l1",
        eps: float = 1e-3,
        low_freq_levels: str = "final",
        pad_mode: str = "reflect",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.levels = levels
        self.level_weights = level_weights
        self.base_loss = base_loss
        self.eps = eps
        self.low_freq_levels = low_freq_levels
        self.pad_mode = pad_mode
        self.reduction = reduction

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return haar_wavelet_loss(
            preds=preds,
            targets=targets,
            levels=self.levels,
            level_weights=self.level_weights,
            base_loss=self.base_loss,
            eps=self.eps,
            low_freq_levels=self.low_freq_levels,
            pad_mode=self.pad_mode,
            reduction=self.reduction,
            weights=weights,
        )
