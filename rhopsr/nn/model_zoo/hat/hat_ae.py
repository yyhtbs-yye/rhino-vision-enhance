import math

import torch
import torch.nn as nn

from rhopsr.nn.model_zoo.hat.modules import HAT
from rhopsr.nn.model_zoo.ae.modules import Encoder, Decoder

from einops import rearrange

class HATAE(HAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        num_levels = int(math.log2(kwargs['patch_size']))
        self.patch_embed = Encoder(in_channels=kwargs['embed_dim'], num_levels=num_levels, num_res_blocks=2)
        self.decoder = Decoder(in_channels=kwargs['embed_dim'] * kwargs['patch_size'], num_levels=num_levels, num_res_blocks=2)
        self.proj = nn.Linear(kwargs['embed_dim'] * kwargs['patch_size'], kwargs['embed_dim'])
        self.xhat_out = nn.Conv2d(kwargs['embed_dim'], 3, kernel_size=1)
        self._init_ae_weights()

    def _init_ae_weights(self):
        """Initialize AE-specific modules without touching parent transformer weights."""

        def _init_fn(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.patch_embed.apply(_init_fn)
        self.xhat_out.apply(_init_fn)
        self.decoder.apply(_init_fn)
        self.proj.apply(_init_fn)

    def forward_features(self, x):

        u = self.patch_embed(x)
        x_size = (u.shape[2], u.shape[3])
        # Calculate attention mask and relative position index in advance to speed up inference. 
        # The original code is very time-consuming for large window size.
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        x = rearrange(u, 'b c h w -> b (h w) c')
        x = self.proj(x)

        xhat = self.xhat_out(self.decoder(u))
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x, xhat

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # for classical SR
        x = self.conv_first(x)
        x, xhat = self.forward_features(x)
        x = self.conv_after_body(x) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        xhat = xhat / self.img_range + self.mean

        return x, xhat


if __name__ == "__main__":
    # Simple sanity check for initialization and forward pass.
    model = HATAE(
        img_size=64,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 3, 6, 6),
        window_size=4,
        upscale=1,
        upsampler='pixelshuffle')

    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        sr, recon = model(dummy)

    print(f"SR output shape: {tuple(sr.shape)}")
    print(f"AE reconstruction shape: {tuple(recon.shape)}")
