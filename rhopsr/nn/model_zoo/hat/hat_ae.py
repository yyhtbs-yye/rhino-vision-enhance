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
        self.decoder = Decoder(in_channels=kwargs['embed_dim'], num_levels=num_levels, num_res_blocks=2)


    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference. 
        # The original code is very time-consuming for large window size.
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        u = self.patch_embed(x)
        x = rearrange(u, 'b c h w -> b (h w) c')

        xhat = self.decoder(u)
        
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
