import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_, to_2tuple

from einops import rearrange

class ChannelRMSNorm2d(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(C))

    def forward(self, x):  # x: (B, C, H, W)
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight[:, None, None]

class ConvMlp(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()

        hidden = max(1, num_feat // squeeze_factor)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, hidden, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        q_window_size,
        kv_window_size=None,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_window_size = to_2tuple(q_window_size)
        self.kv_window_size = to_2tuple(kv_window_size or q_window_size)

        head_dim = dim // num_heads
        self.qk_scale = qk_scale or head_dim ** -0.5

        qh, qw = self.q_window_size
        kh, kw = self.kv_window_size

        # Generalized relative-position bias table:
        # valid for self-attn (qh==kh, qw==kw) and cross-window attn
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((qh + kh - 1) * (qw + kw - 1), num_heads)
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_rel_pos_bias(self, rpi, dtype, device):
        # rpi: [Lq, Lk]
        Lq, Lk = rpi.shape
        bias = self.relative_position_bias_table[rpi.reshape(-1)]
        bias = bias.view(Lq, Lk, self.num_heads).permute(2, 0, 1)   # [nh, Lq, Lk]
        return bias.unsqueeze(0).to(dtype=dtype, device=device)     # [1, nh, Lq, Lk]

    def forward(self, q, k, v, rpi, mask=None):
        """
        q:    [BnW, Lq, C]
        k,v:  [BnW, Lk, C]
        rpi:  [Lq, Lk]
        mask: [nW, Lq, Lk] or None
        """
        BnW, Lq, C = q.shape
        BnW_k, Lk, Ck = k.shape
        BnW_v, Lk_v, Cv = v.shape

        if not (BnW == BnW_k == BnW_v):
            raise ValueError(f"Batch mismatch: q={q.shape}, k={k.shape}, v={v.shape}")
        if not (C == Ck == Cv):
            raise ValueError(f"Channel mismatch: q={q.shape}, k={k.shape}, v={v.shape}")
        if Lk != Lk_v:
            raise ValueError(f"Key/value length mismatch: k={k.shape}, v={v.shape}")

        head_dim = C // self.num_heads
        if C % self.num_heads != 0:
            raise ValueError(f"dim={C} must be divisible by num_heads={self.num_heads}")

        q = rearrange(q, 'bnw l (nh hd) -> bnw nh l hd', nh=self.num_heads, hd=head_dim)
        k = rearrange(k, 'bnw l (nh hd) -> bnw nh l hd', nh=self.num_heads, hd=head_dim)
        v = rearrange(v, 'bnw l (nh hd) -> bnw nh l hd', nh=self.num_heads, hd=head_dim)

        # [1, nh, Lq, Lk]
        attn_bias = self._get_rel_pos_bias(rpi, q.dtype, q.device)

        if mask is not None:
            # mask: [nW, Lq, Lk] -> [BnW, 1, Lq, Lk]
            nW = mask.shape[0]
            if BnW % nW != 0:
                raise ValueError(f"BnW={BnW} is not divisible by nW={nW}")
            B = BnW // nW

            mask = mask.to(dtype=q.dtype, device=q.device)
            mask = mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(BnW, 1, Lq, Lk)

            # broadcast [1, nh, Lq, Lk] + [BnW, 1, Lq, Lk]
            attn_bias = attn_bias + mask

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.qk_scale,
        )   # [BnW, nh, Lq, hd]

        x = rearrange(x, 'bnw nh l hd -> bnw l (nh hd)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):

    def __init__(self, dim, num_heads, window_size, shift_ratio,
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, 
                 qkv_bias=True, qk_scale=None,
                 mlp_ratio=4., attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = (int(window_size[0] * shift_ratio), int(window_size[1] * shift_ratio))
        self.mlp_ratio = mlp_ratio

        self.norm1 = ChannelRMSNorm2d(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, 
                                    q_window_size=window_size,
                                    kv_window_size=window_size,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)


        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.norm2 = ChannelRMSNorm2d(dim)
        self.proj = ConvMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x, rpi, mask):

        shortcut = x
        x = self.norm1(x)

        conv_x = self.conv_block(x)

        # cyclic shift
        if self.shift_size[0] > 0: # x.shape = b, c, h, w
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            mask = mask
        else:
            shifted_x = x
            mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # b, c, h, w -> b*nw, c, window_size[0], window_size[1]

        x_windows = x_windows.flatten(2).transpose(1, 2)  # b*nw, window_size[0]*window_size[1], c

        q_windows, k_windows, v_windows = self.qkv(x_windows).chunk(3, dim=-1)  # q, k, v: b*nw, c, window_size[0], window_size[1]

        o_windows = self.attn(q_windows, k_windows, v_windows, rpi=rpi, mask=mask)

        o_windows = o_windows.transpose(1, 2).reshape(-1, self.dim, self.window_size[0], self.window_size[1])  # b*nw, c, window_size[0], window_size[1]

        shifted_x = window_reverse(o_windows, self.window_size, x.shape[2], x.shape[3])  # b, c, h, w

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))
        else:
            attn_x = shifted_x

        x = shortcut + attn_x + conv_x * self.conv_scale

        x = x + self.proj(self.norm2(x))

        return x

class OCAB(nn.Module):

    def __init__(self, dim, num_heads, window_size, overlap_ratio, 
                 qkv_bias=True, qk_scale=None, 
                 mlp_ratio=2, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qk_scale = qk_scale or head_dim**-0.5

        self.overlap_size = (int(window_size[0] * overlap_ratio), int(window_size[1] * overlap_ratio))
        self.window_size_ext = (window_size[0] + self.overlap_size[0], window_size[1] + self.overlap_size[1])

        self.norm1 = ChannelRMSNorm2d(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn = WindowAttention(dim, num_heads=num_heads, 
                                    q_window_size=window_size,
                                    kv_window_size=self.window_size_ext,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.conv_block = nn.Conv2d(dim, dim, 1)

        self.norm2 = ChannelRMSNorm2d(dim)
        self.proj = ConvMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x, rpi):

        shortcut = x
        x = self.norm1(x)
        
        q, k, v = self.qkv(x).chunk(3, dim=1)

        # partition windows
        q_windows = window_partition(q, self.window_size).flatten(2).transpose(1, 2)  # b*nw, window_size[0]*window_size[1], c
        k_windows = partition_aligned_kv_windows(k, self.window_size, self.window_size_ext).flatten(2).transpose(1, 2)  # [B*nW, kv_ws*kv_ws, C]
        v_windows = partition_aligned_kv_windows(v, self.window_size, self.window_size_ext).flatten(2).transpose(1, 2)  # [B*nW, kv_ws*kv_ws, C]

        o_windows = self.attn(q_windows, k_windows, v_windows, rpi=rpi, mask=None)

        o_windows = o_windows.transpose(1, 2).reshape(-1, self.dim, self.window_size[0], self.window_size[1])  # b*nw, c, window_size[0], window_size[1]

        o_windows = self.conv_block(o_windows)

        x = window_reverse(o_windows, self.window_size, x.shape[2], x.shape[3])  # b, c, h, w

        x = x + shortcut

        x = x + self.proj(self.norm2(x))
        return x


class HATGroup(nn.Module):

    def __init__(self, hidden_size, num_heads, window_size, shift_ratio, depth,
                 compress_ratio, squeeze_factor,
                 conv_scale, overlap_ratio, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 proj_drop=0., attn_drop=0.):

        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # build HAB Blocks
        self.hybrid_attns = nn.ModuleList([
            HAB(
                dim=hidden_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_ratio=0 if (i % 2 == 0) else shift_ratio,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop,
                attn_drop=attn_drop) for i in range(depth)
        ])

        # build OCAB Blocks
        self.overlap_attn = OCAB(
                            dim=hidden_size,
                            num_heads=num_heads,
                            window_size=window_size,
                            overlap_ratio=overlap_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            mlp_ratio=mlp_ratio,
                            proj_drop=proj_drop,
                            attn_drop=attn_drop
        )

    def forward(self, x, rpi_sa, rpi_oca, mask):
        for hybrid_attn in self.hybrid_attns:
            x = hybrid_attn(x, rpi_sa, mask)

        x = self.overlap_attn(x, rpi_oca)

        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def _make_2d_coords(window_size: int, device=None) -> torch.Tensor:
    coords_h = torch.arange(window_size, device=device)
    coords_w = torch.arange(window_size, device=device)
    return torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]


def calculate_relative_position_index(
    query_window_size: int,
    key_window_size: int | None = None,
    *,
    reverse_subtract: bool = False,
    shift: int | tuple[int, int] | None = None,
    device='cpu',
) -> torch.Tensor:
    """
    Shared helper for relative position index.

    Args:
        query_window_size: window size for the query side.
        key_window_size: window size for the key side. Defaults to query_window_size.
        reverse_subtract:
            False -> query - key
            True  -> key - query
        shift:
            int or (shift_h, shift_w). If None, defaults to key_window_size - 1.

    Returns:
        relative_position_index: [query_window_size^2, key_window_size^2]
    """
    if key_window_size is None:
        key_window_size = query_window_size

    coords_q = _make_2d_coords(query_window_size, device=device)
    coords_k = _make_2d_coords(key_window_size, device=device)

    coords_q_flat = torch.flatten(coords_q, 1)  # [2, nq]
    coords_k_flat = torch.flatten(coords_k, 1)  # [2, nk]

    if reverse_subtract:
        # matches your OCA version: key - query
        relative_coords = coords_k_flat[:, None, :] - coords_q_flat[:, :, None]  # [2, nq, nk]
    else:
        # matches your SA version: query - key
        relative_coords = coords_q_flat[:, :, None] - coords_k_flat[:, None, :]  # [2, nq, nk]

    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [nq, nk, 2]

    if shift is None:
        shift_h = shift_w = key_window_size - 1
    elif isinstance(shift, tuple):
        shift_h, shift_w = shift
    else:
        shift_h = shift_w = shift

    relative_coords[:, :, 0] += shift_h
    relative_coords[:, :, 1] += shift_w
    relative_coords[:, :, 0] *= (query_window_size + key_window_size - 1)

    return relative_coords.sum(-1)

def calculate_attention_mask(
    x_size: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
    *,
    device='cpu',
) -> torch.Tensor:
    """
    Calculate attention mask for SW-MSA.

    Returns:
        attn_mask: [num_windows, window_size * window_size, window_size * window_size]
    """
    h, w = x_size

    # BCHW so it is compatible with window_partition(x), which expects [B, C, H, W]
    img_mask = torch.zeros((1, 1, h, w), device=device)  # [1, 1, H, W]

    # Optional safe path
    if shift_size == 0:
        mask_windows = window_partition(img_mask, window_size)          # [nw, 1, ws, ws]
        mask_windows = mask_windows.flatten(1)                          # [nw, ws*ws]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )

    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, :, h_slice, w_slice] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)              # [nw, 1, ws, ws]
    mask_windows = mask_windows.flatten(1)                              # [nw, ws*ws]

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # [nw, ws*ws, ws*ws]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def _get_window_and_stride(window_size, overlap_size=0):
    Wh, Ww = to_2tuple(window_size)
    Oh, Ow = to_2tuple(overlap_size)

    if not (0 <= Oh < Wh and 0 <= Ow < Ww):
        raise ValueError(
            f"overlap_size must satisfy 0 <= overlap < window_size, "
            f"got window_size={(Wh, Ww)}, overlap_size={(Oh, Ow)}"
        )

    Sh, Sw = Wh - Oh, Ww - Ow
    return (Wh, Ww), (Sh, Sw)

def window_partition(x, ws):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    cols = F.unfold(x, kernel_size=ws, stride=ws)              # [B, C*ws*ws, nW]
    windows = cols.transpose(1, 2)                             # [B, nW, C*ws*ws]
    windows = windows.reshape(-1, C, ws[0], ws[1])                   # [B*nW, C, ws, ws]
    return windows

def partition_aligned_kv_windows(x, q_ws, kv_ws):
    # one enlarged kv window per query window
    # padding keeps centers aligned with query-grid windows
    B, C, H, W = x.shape
    pad = ((kv_ws[0] - q_ws[0]) // 2, (kv_ws[1] - q_ws[1]) // 2)

    cols = F.unfold(
        x,
        kernel_size=kv_ws,
        stride=q_ws,        # SAME base grid as query windows
        padding=pad
    )                       # [B, C*kv_ws*kv_ws, nW]

    windows = cols.transpose(1, 2).reshape(-1, C, kv_ws[0], kv_ws[1])
    return windows


def window_reverse(windows: torch.Tensor, window_size, H: int, W: int):
    """
    Args:
        windows: [B * num_windows, C, Wh, Ww]
        window_size: int or (Wh, Ww)
        H, W: output spatial size

    Returns:
        x: [B, C, H, W]
    """
    Wh, Ww = to_2tuple(window_size)
    BnW, C, Wh_in, Ww_in = windows.shape

    if (Wh_in, Ww_in) != (Wh, Ww):
        raise ValueError(
            f"windows spatial size {(Wh_in, Ww_in)} does not match window_size {(Wh, Ww)}"
        )

    if H % Wh != 0 or W % Ww != 0:
        raise ValueError(
            f"H and W must be divisible by window_size, got "
            f"H={H}, W={W}, window_size={(Wh, Ww)}"
        )

    nH = H // Wh
    nW = W // Ww
    num_windows = nH * nW

    if BnW % num_windows != 0:
        raise ValueError(
            f"windows.shape[0]={BnW} is not divisible by num_windows={num_windows}"
        )

    B = BnW // num_windows

    # [B*nH*nW, C, Wh, Ww]
    # -> [B, nH, nW, C, Wh, Ww]
    # -> [B, C, nH, Wh, nW, Ww]
    # -> [B, C, H, W]
    return (
        windows.view(B, nH, nW, C, Wh, Ww)
               .permute(0, 3, 1, 4, 2, 5)
               .reshape(B, C, H, W)
    )