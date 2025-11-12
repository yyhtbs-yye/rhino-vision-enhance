# losses_deblur.py
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Charbonnier / L1
# -------------------------
def charbonnier_loss(preds: torch.Tensor,
                     targets: torch.Tensor,
                     eps: float = 1e-3,
                     reduction: str = "mean") -> torch.Tensor:
    """
    Smooth L1 (Charbonnier) loss: sqrt((x - y)^2 + eps^2)
    """
    diff = preds - targets
    loss = torch.sqrt(diff * diff + eps * eps)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# -------------------------
# 2) Laplacian High-Frequency Loss
# -------------------------
def _laplacian_kernel(device, dtype):
    k = torch.tensor([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], device=device, dtype=dtype)
    return k


def laplacian_map(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Laplacian filter on each channel to get high-frequency map
    """
    b, c, h, w = x.shape
    k = _laplacian_kernel(x.device, x.dtype).view(1, 1, 3, 3)
    k = k.repeat(c, 1, 1, 1)  # (C,1,3,3)
    return F.conv2d(x, k, padding=1, groups=c)


def laplacian_l1_loss(preds: torch.Tensor,
                      targets: torch.Tensor,
                      reduction: str = "mean") -> torch.Tensor:
    lp = laplacian_map(preds)
    lt = laplacian_map(targets)
    loss = torch.abs(lp - lt)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# -------------------------
# 3) SSIM / MS-SSIM
# -------------------------
def _gaussian_window(window_size: int, sigma: float, device, dtype):
    # 1D
    gauss = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    gauss = torch.exp(-(gauss ** 2) / (2 * sigma * sigma))
    gauss /= gauss.sum()
    # 2D
    window_2d = gauss[:, None] @ gauss[None, :]
    window_2d = window_2d / window_2d.sum()
    return window_2d


def _ssim_map(x, y, data_range=1.0, window_size=11, sigma=1.5):
    """
    Return SSIM map between x and y (no average).
    """
    c = x.size(1)
    window = _gaussian_window(window_size, sigma, x.device, x.dtype)
    window = window.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(x * x, window, padding=window_size // 2, groups=c) - mu_x2
    sigma_y = F.conv2d(y * y, window, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=c) - mu_xy

    # SSIM 常数
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-12)
    return ssim_map


def ssim_loss(preds: torch.Tensor,
              targets: torch.Tensor,
              data_range: float = 1.0,
              window_size: int = 11,
              sigma: float = 1.5,
              reduction: str = "mean") -> torch.Tensor:
    """
    L_ssim = 1 - SSIM
    Input is assumed to be in range [0, data_range] (typically data_range=1)
    """
    ssim_map = _ssim_map(preds, targets, data_range, window_size, sigma)
    loss = 1.0 - ssim_map
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def ms_ssim_loss(preds: torch.Tensor,
                 targets: torch.Tensor,
                 data_range: float = 1.0,
                 weights: Optional[torch.Tensor] = None,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 reduction: str = "mean") -> torch.Tensor:
    """
    Multi-scale SSIM loss (1 - MS-SSIM)
    """
    if weights is None:
        # Classic 5-scale weights
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                               device=preds.device, dtype=preds.dtype)

    levels = len(weights)
    msssim_vals = []
    x, y = preds, targets
    for _ in range(levels):
        msssim_vals.append(_ssim_map(x, y, data_range, window_size, sigma).mean(dim=(1, 2, 3)))
        # Downsample by factor of 2
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)

    msssim_vals = torch.stack(msssim_vals, dim=1)  # (B, L)
    # Weighted geometric mean, 
    msssim = torch.exp((weights * torch.log(msssim_vals.clamp(min=1e-6))).sum(dim=1))
    loss = 1.0 - msssim  # (B,)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss.mean()  # 默认


# -------------------------
# 4) Focal Frequency Loss (FFL)
# -------------------------
class FocalFrequencyLoss(nn.Module):
    """
    Reference: Focal Frequency Loss for Image Reconstruction (CVPR 2021)
    Implementation: L2 loss on complex differences in frequency domain, weighted by normalized |Δ|^alpha for focusing.
    """
    def __init__(self, alpha: float = 1.0, eps: float = 1e-8, reduction: str = "mean",
                 use_log: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.use_log = use_log

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 频域：rfft2，对最后两个维度做 FFT
        F_pred = torch.fft.rfft2(preds, norm="ortho")
        F_tgt = torch.fft.rfft2(targets, norm="ortho")

        diff = F_pred - F_tgt
        mag = torch.abs(diff)

        # 聚焦权重：|Δ|^alpha，做通道内空间均值归一化
        if self.use_log:
            w = torch.log(mag + 1.0) ** self.alpha
        else:
            w = torch.pow(mag + self.eps, self.alpha)

        w = w / (w.mean(dim=(-2, -1), keepdim=True) + self.eps)

        # 复数 MSE：|Δ|^2 = Re^2 + Im^2
        loss_map = w * (diff.real ** 2 + diff.imag ** 2)

        if self.reduction == "mean":
            return loss_map.mean()
        if self.reduction == "sum":
            return loss_map.sum()
        return loss_map


# -------------------------
# 5) Combination Loss for Deblurring
# -------------------------
class DeblurCompositeLoss(nn.Module):
    """
    Combination: Charbonnier/L1 + Laplacian high-frequency + SSIM/MS-SSIM + FFL (optional)
    Typical weights: l_main=1.0, w_lap=0.1, w_ssim=0.05, w_ffl=0.0~0.2
    """
    def __init__(self,
                 use_charbonnier: bool = True,
                 l_main: float = 1.0,
                 charbonnier_eps: float = 1e-3,
                 w_lap: float = 0.1,
                 w_ssim: float = 0.05,
                 use_msssim: bool = False,
                 data_range: float = 1.0,
                 w_ffl: float = 0.0,
                 ffl_alpha: float = 1.0,
                 ffl_use_log: bool = False):
        super().__init__()
        self.use_charbonnier = use_charbonnier
        self.l_main = l_main
        self.charb_eps = charbonnier_eps
        self.w_lap = w_lap
        self.w_ssim = w_ssim
        self.use_msssim = use_msssim
        self.data_range = data_range
        self.w_ffl = w_ffl
        self.ffl = FocalFrequencyLoss(alpha=ffl_alpha, use_log=ffl_use_log)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # 主损失
        if self.use_charbonnier:
            losses["main"] = charbonnier_loss(preds, targets, eps=self.charb_eps, reduction="mean")
        else:
            losses["main"] = F.l1_loss(preds, targets)

        # 高频一致
        if self.w_lap > 0:
            losses["lap"] = laplacian_l1_loss(preds, targets)
        else:
            losses["lap"] = preds.new_tensor(0.0)

        # 结构一致（SSIM / MS-SSIM）
        if self.w_ssim > 0:
            if self.use_msssim:
                losses["ssim"] = ms_ssim_loss(preds, targets, data_range=self.data_range)
            else:
                losses["ssim"] = ssim_loss(preds, targets, data_range=self.data_range)
        else:
            losses["ssim"] = preds.new_tensor(0.0)

        # 频域（FFL）
        if self.w_ffl > 0:
            losses["ffl"] = self.ffl(preds, targets)
        else:
            losses["ffl"] = preds.new_tensor(0.0)

        total = (self.l_main * losses["main"]
                 + self.w_lap * losses["lap"]
                 + self.w_ssim * losses["ssim"]
                 + self.w_ffl * losses["ffl"])
        losses["total"] = total
        return losses["total"]


# -------------------------
# 6) 使用示例
# -------------------------
if __name__ == "__main__":
    # 假数据
    B, C, H, W = 2, 3, 128, 128
    preds = torch.rand(B, C, H, W, requires_grad=True).cuda() if torch.cuda.is_available() else torch.rand(B, C, H, W, requires_grad=True)
    targets = torch.rand_like(preds)

    # 典型配置：Charbonnier + Lap(0.1) + SSIM(0.05) + FFL(0.1, alpha=1)
    criterion = DeblurCompositeLoss(
        use_charbonnier=True,
        l_main=1.0,
        w_lap=0.1,
        w_ssim=0.05,
        use_msssim=False,      # 也可以 True
        data_range=1.0,        # 输入若是[0,255]，请设为255
        w_ffl=0.1,
        ffl_alpha=1.0,
        ffl_use_log=False
    )

    out = criterion(preds, targets)
    print({k: float(v.detach()) for k, v in out.items()})
    out["total"].backward()
