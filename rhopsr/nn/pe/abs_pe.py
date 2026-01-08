import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchPositionalEncoding(nn.Module):
    """Utility module that materializes 2D sinusoidal positional encodings for a patch grid."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, height, width, device=None):
        return positional_encoding_2d(self.embed_dim, height, width, device=device)

def positional_encoding_2d(d_model, height, width, device=None):
    """
    2D sinusoidal positional encoding on a (height x width) grid.

    Returns:
        pe: (1, height*width, d_model)
    """
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4 for 2D sin-cos positional encoding")

    d_model_half = d_model // 2

    y_pos = torch.arange(height, device=device).unsqueeze(1)  # (H, 1)
    x_pos = torch.arange(width, device=device).unsqueeze(1)   # (W, 1)

    div_term_x = torch.exp(
        torch.arange(0, d_model_half, 2, device=device) * -(math.log(10000.0) / d_model_half)
    )
    div_term_y = torch.exp(
        torch.arange(0, d_model_half, 2, device=device) * -(math.log(10000.0) / d_model_half)
    )

    # X-direction
    pe_x = torch.zeros(width, d_model_half, device=device)
    x_term = x_pos * div_term_x  # (W, d_model_half/2)
    pe_x[:, 0::2] = torch.sin(x_term)
    pe_x[:, 1::2] = torch.cos(x_term)

    # Y-direction
    pe_y = torch.zeros(height, d_model_half, device=device)
    y_term = y_pos * div_term_y  # (H, d_model_half/2)
    pe_y[:, 0::2] = torch.sin(y_term)
    pe_y[:, 1::2] = torch.cos(y_term)

    # Broadcast to 2D grid and concatenate
    pe_y_b = pe_y[:, None, :].expand(height, width, d_model_half)  # (H, W, d_model_half)
    pe_x_b = pe_x[None, :, :].expand(height, width, d_model_half)  # (H, W, d_model_half)
    pe = torch.cat([pe_y_b, pe_x_b], dim=-1)  # (H, W, d_model)

    pe = pe.view(1, height * width, d_model)
    return pe