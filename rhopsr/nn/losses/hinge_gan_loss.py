import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# -----------------------------
# Base interface + reduction
# -----------------------------

class _ReduceMixin:
    def __init__(self, reduction="mean"):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def _reduce(self, x):
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        return x

class GANLossBase(nn.Module, ABC, _ReduceMixin):
    """
    Interface:
      D step: forward(d_real, d_fake)
      G step: forward(d_fake, None)
    """
    def __init__(self, reduction="mean"):
        nn.Module.__init__(self)
        _ReduceMixin.__init__(self, reduction=reduction)

    @abstractmethod
    def forward(self, real_imgs, fake_imgs):
        ...


# -----------------------------
# Hinge loss (spectral-norm D common)
# D: E[relu(1 - D(real))] + E[relu(1 + D(fake))]
# G: -E[D(fake)]
# -----------------------------

class HingeGANLoss(GANLossBase):
    def forward(self, real_imgs, fake_imgs):
        if fake_imgs is None:  # generator
            return self._reduce(-real_imgs)
        # discriminator
        d_real, d_fake = real_imgs, fake_imgs
        loss_real = F.relu(1.0 - d_real)
        loss_fake = F.relu(1.0 + d_fake)
        return self._reduce(loss_real + loss_fake)
