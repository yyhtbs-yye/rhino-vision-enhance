from __future__ import annotations

from typing import Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _loss_map(preds: torch.Tensor, targets: torch.Tensor, base_loss: str, eps: float) -> torch.Tensor:
    if base_loss == "l1":
        return (preds - targets).abs()
    if base_loss == "mse":
        diff = preds - targets
        return diff * diff
    if base_loss == "charbonnier":
        diff = preds - targets
        return torch.sqrt(diff * diff + eps * eps)
    raise ValueError(f"Unsupported base_loss: {base_loss}")


def _pad_to_even_hw(x: torch.Tensor, pad_mode: str) -> torch.Tensor:
    # Pad right/bottom if needed so H,W are even (required for stride-2 Haar).
    h, w = x.shape[-2], x.shape[-1]
    if h < 2 or w < 2:
        raise ValueError(f"Input too small for Haar decomposition: HxW={h}x{w}")
    pad_right = int(w % 2)
    pad_bottom = int(h % 2)
    if pad_right or pad_bottom:
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    return x


def _haar_base_filters(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Orthonormal separable Haar:
    # low=[1,1]/sqrt(2), high=[1,-1]/sqrt(2) -> 2D outer => scale 1/2
    scale = 0.5
    ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device, dtype=dtype) * scale
    lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]], device=device, dtype=dtype) * scale
    hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=device, dtype=dtype) * scale
    hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=device, dtype=dtype) * scale
    # (4,1,2,2)
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)


def _wavelet_decompose_once(x: torch.Tensor, pad_mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: (N, C, H, W)
    returns: ll, lh, hl, hh each (N, C, H/2, W/2) after padding to even H/W
    """
    if x.dim() != 4:
        raise ValueError(f"Expected (N,C,H,W), got shape={tuple(x.shape)}")

    x = _pad_to_even_hw(x, pad_mode=pad_mode)
    n, c, _, _ = x.shape

    base = _haar_base_filters(device=x.device, dtype=x.dtype)  # (4,1,2,2)
    weight = base.repeat(c, 1, 1, 1)  # (4*C,1,2,2)

    y = F.conv2d(x, weight, stride=2, padding=0, groups=c)  # (N, 4*C, H/2, W/2)
    y = y.view(n, c, 4, y.size(-2), y.size(-1))
    ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
    return ll, lh, hl, hh


def haar_wavelet_loss_2d(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    level_weights: torch.Tensor,
    base_loss: str = "l1",
    eps: float = 1e-3,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Pure 2D Haar wavelet loss.

    Inputs:
      preds, targets: (N, C, H, W)
      level_weights: (L,) direct multipliers on each level's contribution

    Definition used:
      total[n] = sum_{level=0..L-1} w[level] * HF_loss_level[n]
                 + w[L-1] * LL_loss_final[n]

    where HF_loss_level is the mean of (LH, HL, HH) loss maps at that level.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"preds/targets mismatch: {tuple(preds.shape)} vs {tuple(targets.shape)}")
    if preds.dim() != 4:
        raise ValueError(f"Expected preds/targets 4D (N,C,H,W), got {tuple(preds.shape)}")

    if level_weights.dim() != 1 or level_weights.numel() < 1:
        raise ValueError(f"level_weights must be 1D with >=1 element, got {tuple(level_weights.shape)}")

    ll_p, ll_t = preds, targets
    per_sample = preds.new_zeros((preds.size(0),), dtype=preds.dtype)

    L = int(level_weights.numel())
    for level in range(L):
        w = level_weights[level]

        ll_p, lh_p, hl_p, hh_p = _wavelet_decompose_once(ll_p, pad_mode=pad_mode)
        ll_t, lh_t, hl_t, hh_t = _wavelet_decompose_once(ll_t, pad_mode=pad_mode)

        hf = (
            _loss_map(lh_p, lh_t, base_loss=base_loss, eps=eps)
            + _loss_map(hl_p, hl_t, base_loss=base_loss, eps=eps)
            + _loss_map(hh_p, hh_t, base_loss=base_loss, eps=eps)
        )
        per_sample = per_sample + w * hf.view(hf.size(0), -1).mean(dim=1)

    # Final low-frequency anchor, weighted by last level weight.
    lf = _loss_map(ll_p, ll_t, base_loss=base_loss, eps=eps)
    per_sample = per_sample + level_weights[-1] * lf.view(lf.size(0), -1).mean(dim=1)

    return per_sample


def _reduce(per_sample: torch.Tensor, reduction: str, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        if weights.shape != per_sample.shape:
            raise ValueError(f"weights must match shape {tuple(per_sample.shape)}, got {tuple(weights.shape)}")
        if reduction == "none":
            return per_sample * weights
        if reduction == "sum":
            return (per_sample * weights).sum()
        if reduction == "mean":
            denom = weights.sum().clamp_min(1e-12)
            return (per_sample * weights).sum() / denom
        raise ValueError(f"Unsupported reduction: {reduction}")

    # Unweighted reduction
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


class HaarWaveletLoss(nn.Module):
    """
    Wrapper that accepts:
      - images: (B,C,H,W) with weights (B,)
      - videos: (B,T,C,H,W) with weights (B,) or (B*T,)

    Video flattening + weights expansion happen here (not in the functional loss).
    """
    def __init__(
        self,
        *,
        levels: Optional[int] = None,
        level_weights: Optional[Sequence[float] | torch.Tensor] = None,
        base_loss: str = "l1",
        eps: float = 1e-3,
        pad_mode: str = "reflect",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.eps = eps
        self.pad_mode = pad_mode
        self.reduction = reduction

        if level_weights is None:
            L = 1 if levels is None else int(levels)
            if L < 1:
                raise ValueError(f"levels must be >= 1, got {L}")
            lw = torch.ones(L, dtype=torch.float32)
        else:
            if isinstance(level_weights, torch.Tensor):
                lw = level_weights.detach().to(dtype=torch.float32).flatten()
            else:
                lw = torch.tensor(list(level_weights), dtype=torch.float32)
            if lw.numel() < 1:
                raise ValueError("level_weights must contain at least one value.")
            if levels is not None and int(levels) != int(lw.numel()):
                raise ValueError(f"levels ({levels}) must match len(level_weights) ({lw.numel()}).")

        self.register_buffer("level_weights", lw, persistent=False)

    @staticmethod
    def _flatten_video(x: torch.Tensor) -> Tuple[torch.Tensor, int, Optional[int]]:
        if x.dim() == 4:
            b = x.size(0)
            return x, b, None
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b * t, c, h, w), b, t
        raise ValueError(f"Expected 4D or 5D input, got shape={tuple(x.shape)}")

    @staticmethod
    def _expand_weights(weights: Optional[torch.Tensor], b: int, t: Optional[int]) -> Optional[torch.Tensor]:
        if weights is None or t is None:
            return weights
        if weights.dim() != 1:
            raise ValueError(f"weights must be 1D, got {tuple(weights.shape)}")
        if weights.numel() == b * t:
            return weights
        if weights.numel() == b:
            return weights.repeat_interleave(t, dim=0)
        raise ValueError(f"weights must have length B ({b}) or B*T ({b*t}), got {weights.numel()}")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        preds_2d, b, t = self._flatten_video(preds)
        targets_2d, b2, t2 = self._flatten_video(targets)
        if (b != b2) or (t != t2):
            raise ValueError(f"preds/targets mismatch: {tuple(preds.shape)} vs {tuple(targets.shape)}")
        
        weights = self._expand_weights(weights, b=b, t=t)

        per_sample = haar_wavelet_loss_2d(
            preds_2d,
            targets_2d,
            level_weights=self.level_weights.to(device=preds_2d.device, dtype=preds_2d.dtype),
            base_loss=self.base_loss,
            eps=self.eps,
            pad_mode=self.pad_mode,
        )
        return _reduce(per_sample, reduction=self.reduction, weights=weights)
