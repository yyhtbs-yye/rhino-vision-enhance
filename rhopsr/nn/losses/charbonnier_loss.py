from __future__ import annotations

import torch
import torch.nn as nn

def charbonnier_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-3,
    alpha: float = 0.5,
    reduction: str = "mean",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Charbonnier loss: ( (preds-targets)^2 + eps^2 )^alpha.

    - `reduction="none"` returns the elementwise loss map.
    - If `weights` is provided, it is applied per-sample after spatial/channel reduction.
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    diff = preds - targets
    loss_map = torch.pow(diff * diff + eps * eps, alpha)

    if reduction == "none":
        return loss_map

    if loss_map.numel() == 0:
        return loss_map.sum()

    if loss_map.dim() == 0:
        loss_per_sample = loss_map.view(1)
    else:
        loss_per_sample = loss_map.view(loss_map.size(0), -1).mean(dim=1)

    if weights is not None:
        loss_per_sample = loss_per_sample * weights

    if reduction == "sum":
        return loss_per_sample.sum()
    return loss_per_sample.mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, alpha: float = 0.5, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.eps = eps
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return charbonnier_loss(
            preds=preds,
            targets=targets,
            eps=self.eps,
            alpha=self.alpha,
            reduction=self.reduction,
            weights=weights,
        )
