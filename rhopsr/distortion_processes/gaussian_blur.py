import math
import torch
import torch.nn.functional as F

# ---------- Small, fast Gaussian blur with reflect-padding ----------
def gaussian_kernel1d(sigma: float, radius: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Handle very small sigma as identity
    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def separable_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: (N, C, H, W)
    sigma: gaussian sigma; if <= 0, returns x directly
    """
    if sigma <= 0:
        return x

    dtype = x.dtype
    device = x.device

    # Heuristic: radius = ceil(3*sigma) covers >99% mass
    radius = int(math.ceil(3.0 * sigma))
    k1 = gaussian_kernel1d(sigma, radius, dtype=dtype, device=device)  # (K,)
    k2 = k1

    # Depthwise conv via groups=C. Build 1D conv kernels for H and W.
    C = x.shape[1]

    # Horizontal blur (W)
    k_w = k1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)  # (C,1,1,K)
    pad_w = (radius, radius, 0, 0)
    y = F.pad(x, pad_w, mode="reflect")
    y = F.conv2d(y, k_w, bias=None, stride=1, padding=0, groups=C)

    # Vertical blur (H)
    k_h = k2.view(1, 1, -1, 1).repeat(C, 1, 1, 1)  # (C,1,K,1)
    pad_h = (0, 0, radius, radius)
    y = F.pad(y, pad_h, mode="reflect")
    y = F.conv2d(y, k_h, bias=None, stride=1, padding=0, groups=C)

    return y