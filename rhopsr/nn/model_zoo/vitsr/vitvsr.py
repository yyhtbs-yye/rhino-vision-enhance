import torch
import torch.nn as nn

from rhopsr.nn.model_zoo.vitsr.vitsr import (
    Patchify,
    PatchEmbedding,
    PatchPositionalEncoding,
    PatchUnembed,
)

class ViTVSR(nn.Module):
    """
    Pure ViT-based Video Super-Resolution with BasicVSR++-style grid propagation.

    Pipeline:
        feat_extract -> propagate (B-F-B-F) -> reconstruction -> PixelShuffle

    - feat_extract:
        Patchify + Linear patch embedding + 2D sinusoidal positional encoding
    - propagate:
        1st-order recurrent propagation using TemporalCrossAttentionBlock
        in multiple passes over time:
            backward -> forward -> backward -> forward
    - backbone:
        Stack of spatial TransformerBlock(s) (no temporal ops inside)
    - upsample:
        PatchUnembed to LR subpixel maps, then PixelShuffle ONCE
        (PixelShuffle is the dedicated Upsample layer, outside temporal loops)

    """

    def __init__(
        self,
        scale=4,
        in_channels=3,
        out_channels=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
        patch_size=4,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal PE"
        assert patch_size > 0, "patch_size must be > 0"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.scale = scale
        self.patch_size = patch_size

        # ---- feat_extract ----
        self.patchify = Patchify(patch_size)
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_encoding = PatchPositionalEncoding(embed_dim)

        # ---- alignment (temporal cross-attn) ----
        self.alignment = CrossAttentionTransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

        # ---- backbone (spatial ViT blocks) ----
        self.backbones = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                )
                for _ in range(depth)
            ])
            for __ in range(4)
        ])
        self.backbone_norm = nn.LayerNorm(embed_dim)

        # ---- reconstruction + upsample ----
        self.reconstruction = PatchUnembed(
            embed_dim=embed_dim,
            out_channels=out_channels,
            patch_size=patch_size,
            scale=scale,
        )
        # PixelShuffle as global Upsample layer (outside propagation loops)
        self.upsample = nn.PixelShuffle(scale)

    # -------------------------------------------------------------
    # feat_extract: per-frame ViT tokenization (BasicVSR++ "feat_extract")
    # -------------------------------------------------------------
    def feat_extract(self, x):
        """
        Args:
            x: (B, T, C_in, H, W)

        Returns:
            tokens_list: list length T with (B, N, D) per frame
            grids:       list length T with PatchGrid per frame
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got {x.shape}")

        B, T, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        tokens_list = []
        grids = []

        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)

            patches, grid = self.patchify(frame)  # (B, N, C*p^2), PatchGrid
            tokens = self.patch_embed(patches)    # (B, N, D)
            tokens = tokens + self.pos_encoding(grid, device=frame.device)

            tokens_list.append(tokens)
            grids.append(grid)

        return tokens_list, grids

    # -------------------------------------------------------------
    # backbone: pure spatial ViT encoder
    # -------------------------------------------------------------
    def _run_backbone(self, tokens, which):
        """
        Standard ViT encoder trunk (no temporal ops).
        tokens: (B, N, D)
        """
        x = tokens
        for blk in self.backbones[which]:
            x = blk(x)
        x = self.backbone_norm(x)
        return x

    # -------------------------------------------------------------
    # Single directional pass (1st-order recurrent) over the sequence
    # -------------------------------------------------------------
    def _propagate_once(self, tokens_list, cond_feats=None, 
                        is_backward=False, which=None,
    ):
        """
        One 1st-order recurrent pass (forward or backward) over the sequence.

        Args:
            tokens_list: list length T of (B, N, D), from feat_extract
            cond_feats:  optional list length T of (B, N, D),
                         used as per-frame conditioning (from previous passes).
                         If given, tokens_t is replaced by tokens_t + cond_feats[t].
            is_backward: if True, traverse T-1 -> 0, else 0 -> T-1.

        Returns:
            feats: list length T of (B, N, D), propagated in the chosen direction.
        """
        T = len(tokens_list)
        feats = [None] * T

        if cond_feats is not None and len(cond_feats) != T:
            raise ValueError("cond_feats must be None or have same length as tokens_list")

        indices = range(T - 1, -1, -1) if is_backward else range(T)

        prev_feat = None  # recurrent hidden state

        for i in indices:
            tokens_t = tokens_list[i]

            # Add conditioning from previous passes if available
            if cond_feats is not None and cond_feats[i] is not None:
                tokens_t = tokens_t + cond_feats[i]

            if prev_feat is None:
                # First frame in this directional pass: no temporal context yet
                fused = tokens_t
            else:
                # Recurrent alignment via cross-attention:
                #   aligned_prev = CA(prev_feat, tokens_t)
                # where:
                #   - queries from prev_feat (propagated state)
                #   - keys/values from tokens_t (current frame tokens)
                aligned_prev = self.alignment(prev_feat, tokens_t)
                fused = tokens_t + aligned_prev

            # Spatial backbone on fused tokens
            feat_t = self._run_backbone(fused, which)  # (B, N, D)

            feats[i] = feat_t
            prev_feat = feat_t

        return feats

    # -------------------------------------------------------------
    # Grid propagation: backward -> forward -> backward -> forward
    # -------------------------------------------------------------
    def propagate(self, tokens_list):
        """
        BasicVSR++-style grid propagation:

            backward -> forward -> backward -> forward

        Each pass is a 1st-order recurrent sweep that uses TemporalCrossAttention
        for alignment and the ViT backbone for spatial modeling. Passes after
        the first one are conditioned on the previous pass's per-frame features.
        """
        # 1st pass: backward (no conditioning)
        feats_bwd1 = self._propagate_once(tokens_list, cond_feats=None, is_backward=True, which=0)
        feats_fwd2 = self._propagate_once(tokens_list, cond_feats=feats_bwd1, is_backward=False, which=1)
        feats_bwd3 = self._propagate_once(tokens_list, cond_feats=feats_fwd2, is_backward=True, which=2)
        feats_fwd4 = self._propagate_once(tokens_list, cond_feats=feats_bwd3, is_backward=False, which=3)

        # Final features from the last forward pass
        return feats_fwd4

    # -------------------------------------------------------------
    # forward: full video SR pipeline
    # -------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (B, T, C_in, H, W) low-resolution video

        Returns:
            y: (B, T, C_out, H*scale, W*scale)

        Note:
            Spatial dims are internally cropped inside Patchify so that
            H and W are multiples of patch_size (no padding).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got {x.shape}")

        B, T, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        # 1) feat_extract for all frames
        tokens_list, grids = self.feat_extract(x)

        # 2) multi-pass grid propagation (B-F-B-F)
        feats = self.propagate(tokens_list)

        # 3) reconstruction to LR subpixel maps (no PixelShuffle yet)
        subpix_list = []
        for t in range(T):
            subpix = self.reconstruction(feats[t], grids[t])  # (B, C_out*r^2, Hc, Wc)
            subpix_list.append(subpix)

        # Stack over time: (B, T, C_out*r^2, Hc, Wc)
        subpix_seq = torch.stack(subpix_list, dim=1)
        B2, T2, Crr, Hc, Wc = subpix_seq.shape
        assert B2 == B and T2 == T

        # 4) PixelShuffle as global Upsample layer (outside propagation loops)
        subpix_seq = subpix_seq.view(B * T, Crr, Hc, Wc)  # (B*T, C_out*r^2, Hc, Wc)
        sr = self.upsample(subpix_seq)                    # (B*T, C_out, Hc*r, Wc*r)
        sr = sr.view(B, T, self.out_channels,
                     Hc * self.scale, Wc * self.scale,)

        return sr


if __name__ == "__main__":
    # Simple sanity check
    model = ViTVSR(
        scale=4,
        in_channels=3,
        out_channels=3,
        embed_dim=96,
        depth=2,
        num_heads=4,
        patch_size=8,
    )
    x = torch.randn(1, 5, 3, 64, 64)
    y = model(x)
    print("Input: ", x.shape)
    print("Output:", y.shape)
