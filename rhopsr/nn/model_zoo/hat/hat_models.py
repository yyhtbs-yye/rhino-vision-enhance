import torch
import torch.nn as nn

from timm.layers import trunc_normal_, to_2tuple

from rhopsr.nn.model_zoo.hat.hat_modules import ChannelRMSNorm2d, Upsample, HATGroup, calculate_relative_position_index, calculate_attention_mask

class HATModel(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()

        if model_name == 'HAT-Tiny':
            self.model = HAT(patch_size=1, window_size=8, hidden_size=48, 
                             depths=(6, 6, 6, 6), 
                             num_heads=3, shift_ratio=0.5, overlap_ratio=0.5, 
                             compress_ratio=24, squeeze_factor=24,
                             conv_scale=0.01, mlp_ratio=2., **kwargs)
        elif model_name == 'HAT-Small':
            self.model = HAT(patch_size=1, window_size=8, hidden_size=96, 
                             depths=(6, 6, 6, 6, 6, 6), 
                             num_heads=6, shift_ratio=0.5, overlap_ratio=0.5, 
                             compress_ratio=24, squeeze_factor=24,
                             conv_scale=0.01, mlp_ratio=2., **kwargs)
        elif model_name == 'HAT-Medium':
            self.model = HAT(patch_size=1, window_size=16, hidden_size=180, 
                             depths=(6, 6, 6, 6, 6, 6), 
                             num_heads=9, shift_ratio=0.5, overlap_ratio=0.5,
                             compress_ratio=3, squeeze_factor=30,
                             conv_scale=0.01, mlp_ratio=2., **kwargs)
        elif model_name == 'HAT-Large':
            self.model = HAT(patch_size=1, window_size=16, hidden_size=180, 
                             depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6), 
                             num_heads=12, shift_ratio=0.5, overlap_ratio=0.5, 
                            compress_ratio=3, squeeze_factor=30,
                             conv_scale=0.01, mlp_ratio=2., **kwargs)
        self.model = HAT(**kwargs)

    def forward(self, x):
        return self.model(x)

class HAT(nn.Module):

    def __init__(self, patch_size=1, window_size=8,
                 in_channels=3, hidden_size=96, depths=(6, 6, 6, 6), num_heads=6, shift_ratio=0.5,
                 compress_ratio=24, squeeze_factor=24,
                 conv_scale=0.01, overlap_ratio=0.5,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 proj_drop=0., attn_drop=0., 
                 scale=4,
                 **kwargs):
        super(HAT, self).__init__()

        self.patch_size = patch_size
        self.window_size = to_2tuple(window_size)
        self.shift_ratio = shift_ratio
        self.overlap_ratio = overlap_ratio
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_layers = len(depths)
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.compress_ratio = compress_ratio
        self.squeeze_factor = squeeze_factor

        upsample_hidden_size = hidden_size // 2

        self.scale = scale

        # relative position index for SA
        relative_position_index_SA = calculate_relative_position_index(query_window_size=window_size,
                                                                       key_window_size=window_size,
                                                                       reverse_subtract=False,
                                                                       shift=window_size - 1,
        )

         # relative position index for OCA
        overlap_size = int(window_size * overlap_ratio)
        window_size_ext = window_size + overlap_size
        relative_position_index_OCA = calculate_relative_position_index(query_window_size=window_size,
                                                                        key_window_size=window_size_ext,
                                                                        reverse_subtract=False,
                                                                        shift=None,
        )

        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # 1. shallow feature extraction: 
        # TODO: understand why lift the n_chs to hidden_size?
        self.preproc = nn.Conv2d(self.in_channels, hidden_size, 3, 1, 1)

        self.patch_embed = nn.Conv2d(hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.patch_unembed = nn.PixelShuffle(patch_size)

        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            block = HATBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=to_2tuple(window_size),
                shift_ratio=shift_ratio,
                depth=depths[i_layer],
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                proj_drop=proj_drop,
                attn_drop=attn_drop)
            self.blocks.append(block)
        
        self.post_norm = ChannelRMSNorm2d(hidden_size)

        assert hidden_size % (patch_size ** 2) == 0, (
            f"hidden_size={hidden_size} must be divisible by patch_size^2={patch_size**2}"
        )
        self.conv_after_body = nn.Conv2d(hidden_size // (patch_size**2), hidden_size, 3, 1, 1)

        # 3. high quality image reconstruction:
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(hidden_size, upsample_hidden_size, 3, 1, 1), 
            nn.LeakyReLU(inplace=True)
        )
        
        self.upsample = Upsample(scale, upsample_hidden_size)
        
        self.post_conv = nn.Conv2d(upsample_hidden_size, self.out_channels, 3, 1, 1)

        self.mask_cached = dict()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        B, C, H, W = x.shape

        # Input size must be divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Input image size ({H}*{W}) should be divisible by patch size ({self.patch_size}*{self.patch_size})."
        )

        h_res, w_res = H // self.patch_size, W // self.patch_size

        assert h_res % self.window_size[0] == 0 and w_res % self.window_size[1] == 0, (
            f"After patching, feature size ({h_res}, {w_res}) must be divisible by "
            f"window_size {self.window_size}."
        )

        x = self.preproc(x)

        if (h_res, w_res) not in self.mask_cached:
            shift_size = (int(self.window_size[0] * self.shift_ratio), int(self.window_size[1] * self.shift_ratio))
            attn_mask = calculate_attention_mask((h_res, w_res), self.window_size, shift_size).to(x.device)
            self.mask_cached[(h_res, w_res)] = attn_mask
        else:
            attn_mask = self.mask_cached[(h_res, w_res)].to(x.device)
            
        h = self.patch_embed(x)  # h.shape = [B, hidden_size, H//patch_size, W//patch_size]

        for block in self.blocks:
            h = block(h, rpi_sa=self.relative_position_index_SA, 
                      rpi_oca=self.relative_position_index_OCA, 
                      mask=attn_mask)

        h = self.post_norm(h)                               # h.shape = [B, hidden_size, H//patch_size, W//patch_size]

        h = self.patch_unembed(h)

        x = self.conv_after_body(h) + x                     # x.shape = [B, hidden_size, H//patch_size, W//patch_size]

        x = self.conv_before_upsample(x)                    # x.shape = [B, upsample_hidden_size, H, W]
        
        x = self.upsample(x)                                # x.shape = [B, upsample_hidden_size, H*scale, W*scale]

        x = self.post_conv(x)                               # x.shape = [B, out_channels, H*scale, W*scale]

        return x
    

class HATBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, window_size, shift_ratio, depth, 
                 compress_ratio, squeeze_factor, conv_scale, overlap_ratio,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 proj_drop=0., attn_drop=0.):
        super(HATBlock, self).__init__()

        self.residual_group = HATGroup(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            shift_ratio=shift_ratio,
            depth=depth,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
            attn_drop=attn_drop)

        self.conv = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)

    def forward(self, x, rpi_sa, rpi_oca, mask):

        h = self.residual_group(x, rpi_sa=rpi_sa, 
                                rpi_oca=rpi_oca, 
                                mask=mask)
        
        h = self.conv(h)

        return h + x
    

if __name__ == "__main__":
    # Simple sanity check for initialization and forward pass.
    model = HAT(
        patch_size=4,
        window_size=8,
        in_channels=3,
        hidden_size=96,
        depths=(6, 6, 6, 6),
        num_heads=6,
        shift_ratio=0.5,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        proj_drop=0.,
        attn_drop=0.,
        scale=2
    )

    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    print(out.shape)