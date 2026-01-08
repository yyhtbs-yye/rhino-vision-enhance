import torch
import torch.nn as nn
import torch.nn.functional as F

from rhopsr.nn.model_zoo.basicvsr.basicvsrpp import ConvResidualBlocks

from rhopsr.nn.model_zoo.vitsr.vit import ViT_models

class Upsampler(nn.Module):
    """Simple SR head that fuses ViT and stem features and upsamples to HR."""

    def __init__(self, hidden_size, scale):
        super().__init__()
        self.scale = scale

        # Fuse features coming from the CNN stem and the ViT core.
        self.fusion = ConvResidualBlocks(
            num_in_ch=hidden_size * 2,
            num_out_ch=hidden_size,
            num_block=3,
        )

        self.upsample = self._build_upsampler(hidden_size, scale)
        self.conv_hr = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)
        self.conv_last = nn.Conv2d(hidden_size, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if scale == 1:
            self.base_upsample = nn.Identity()
        else:
            self.base_upsample = nn.Upsample(
                scale_factor=scale, mode='bilinear', align_corners=False
            )

    def _build_upsampler(self, channels, scale):
        """Stack PixelShuffle blocks so any supported scale can be built."""
        if scale == 1:
            return nn.Identity()

        layers = []
        remaining = scale
        while remaining > 1:
            if remaining % 2 == 0:
                layers.append(nn.Conv2d(channels, channels * 4, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
                remaining //= 2
            elif remaining == 3:
                layers.append(nn.Conv2d(channels, channels * 9, 3, 1, 1))
                layers.append(nn.PixelShuffle(3))
                remaining = 1
            else:
                raise ValueError(f"Unsupported scale: {scale}")
        return nn.Sequential(*layers)

    def forward(self, x_lr, feats):
        spatial = feats.get('spatial')
        vit = feats.get('vit')
        if spatial is None or vit is None:
            raise ValueError("Upsampler expects 'spatial' and 'vit' entries in feats.")

        residual = torch.cat([spatial, vit], dim=1)
        residual = self.fusion(residual)
        residual = self.upsample(residual)
        residual = self.act(self.conv_hr(residual))
        residual = self.conv_last(residual)

        base = self.base_upsample(x_lr)
        return base + residual

class ViTSR(nn.Module):
    """
    One-pass image super-resolution model using ViT ViT as a backbone.

    - Stem: LR -> HR (learned upsampling)
    - Core: ViT ViT (unchanged, used as an image-to-image transformer)
    - Head: residual refinement (base upsample + refined residual)

    Args:
        vit_name:   key in ViT_models dict, e.g. 'ViT-B/16'
        scale:      SR scale factor (e.g. 2, 3, 4)
        hr_size:    spatial size of HR images (assumed square, hr_size x hr_size)
        num_classes: how many "classes" for the ViT label embedding.
                     We ignore labels, so 1 is enough.
    """
    def __init__(
        self,
        vit_name='ViT-B/16',
        scale=4,
        hr_size=256,
        num_classes=0
    ):
        super().__init__()
        assert vit_name in ViT_models, f"Unknown vit_name: {vit_name}"
        assert hr_size % scale == 0, "hr_size must be divisible by scale"

        self.scale = scale
        self.hr_size = hr_size
        self.lr_size = hr_size // scale

        self.hidden_size = 256

        # --- Core ViT backbone (unchanged) ---
        # We keep in_channels = 3, and set num_classes small since we ignore labels.
        self.vit_backbone = ViT_models[vit_name](
            input_size=self.lr_size,
            in_channels=self.hidden_size,
            num_classes=num_classes
        )

        # --- Stem & Head ---
        self.feat_extract = ConvResidualBlocks(3, self.hidden_size, 5)
        self.upsampler = Upsampler(self.hidden_size, scale=scale)

        # init only the custom CNN blocks per request
        self._init_cnn(self.feat_extract)
        self._init_cnn(self.upsampler)

    def _init_cnn(self, module):
        """Kaiming init for Conv2d layers to stabilize training."""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_lr):
        """
        x_lr: (N, 3, H_lr, W_lr) with H_lr = hr_size // scale
        returns: (N, 3, hr_size, hr_size)
        """
        # 1) Learned upsampling to HR grid
        latent = self.feat_extract(x_lr)  # (N, 3, hr_size, hr_size)

        # 2) Dummy t and y (we ignore them, but ViT API requires them)
        N = x_lr.shape[0]
        device = x_lr.device

        # t: scalar timestep per sample (we just fix it to zeros)
        t = torch.zeros(N, device=device)

        # y: class labels (we fix all to 0, valid since num_classes >= 1)
        y = torch.zeros(N, dtype=torch.long, device=device)

        # 3) Pass through ViT ViT backbone
        vit_out = self.vit_backbone(latent, t, y)  # (N, 3, hr_size, hr_size)

        feats = {
            'spatial': latent,
            'vit': vit_out,
        }

        # 4) Residual head refinement
        sr = self.upsampler(x_lr, feats)
        return sr

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose config
    vit_name = "ViT-B/16"
    scale = 4
    hr_size = 256
    num_classes = 1  # usually ViT wants >= 1, even if you ignore labels

    # Instantiate model
    model = ViTSR(
        vit_name=vit_name,
        scale=scale,
        hr_size=hr_size,
        num_classes=num_classes,
    ).to(device)
    model.eval()

    # Build a dummy LR input
    lr_h = hr_size // scale
    lr_w = hr_size // scale
    batch_size = 1

    x_lr = torch.randn(batch_size, 3, lr_h, lr_w, device=device)

    with torch.no_grad():
        sr = model(x_lr)

    print("=== ViTSR test ===")
    print(f"Scale factor: {scale}")
    print(f"HR size (configured): {hr_size} x {hr_size}")
    print(f"LR size (computed):   {lr_h} x {lr_w}")
    print(f"Input  tensor shape:  {tuple(x_lr.shape)}")
    print(f"Output tensor shape:  {tuple(sr.shape)}")

    # Optional: number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#params: {n_params/1e6:.2f}M")
