import torch
import torch.nn as nn

from diffusers import UNet2DModel, DDPMScheduler


class OpenAIImagePriorWrapper(nn.Module):
    """
    Minimal unconditional image-space diffusion wrapper.

    Design choices:
    - epsilon prediction only
    - DDPM forward noising
    - same public interfaces as the old wrapper
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        variant: str | None = None,
        use_safetensors: bool = True,
        image_range: str = "zero_one",          # "zero_one" or "minus_one_one"
        sample_posterior: bool = False,         # kept only for interface compatibility
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        layers_per_block: int = 2,
        block_out_channels: tuple[int, ...] = (128, 128, 256, 256),
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()

        self.image_range = image_range
        self.sample_posterior = sample_posterior
        self.original_size = original_size
        self.target_size = target_size
        self.crops_coords_top_left = crops_coords_top_left
        self.force_vae_upcast = force_vae_upcast
        self.adapter_name = adapter_name
        self.image_size = int(image_size)

        # If you already have a diffusers-format UNet checkpoint, you can load it.
        # Otherwise, build a simple image-space UNet from scratch.
        if model_id is not None:
            load_kwargs = dict(torch_dtype=torch_dtype, use_safetensors=use_safetensors)
            if variant is not None:
                load_kwargs["variant"] = variant

            try:
                self.unet = UNet2DModel.from_pretrained(model_id, subfolder="unet", **load_kwargs)
            except Exception:
                self.unet = UNet2DModel.from_pretrained(model_id, **load_kwargs)
        else:
            n_blocks = len(block_out_channels)
            self.unet = UNet2DModel(
                sample_size=self.image_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=("DownBlock2D",) * n_blocks,
                up_block_types=("UpBlock2D",) * n_blocks,
            )

        self.unet = self.unet.to(device=device, dtype=torch_dtype)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
            clip_sample=False,
        )

        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type != "epsilon":
            raise ValueError(
                f"This wrapper only supports epsilon prediction, got prediction_type={prediction_type!r}."
            )

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def _to_timestep_indices(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
        - float timesteps in [0, 1]
        - integer timesteps in [0, num_train_timesteps - 1]
        """
        timesteps = torch.as_tensor(timesteps, device=self.device)
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)

        if timesteps.dtype.is_floating_point:
            timesteps = torch.round(timesteps * (num_train_timesteps - 1)).long()
        else:
            timesteps = timesteps.long()

        timesteps = timesteps.clamp(0, num_train_timesteps - 1)
        return timesteps

    def _prepare_batch_timesteps(self, timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        timesteps = self._to_timestep_indices(timesteps)

        if timesteps.ndim == 0:
            timesteps = timesteps[None]

        if timesteps.shape[0] == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"timesteps batch mismatch: got {timesteps.shape[0]}, expected 1 or {batch_size}"
            )

        return timesteps.to(self.device)

    def set_teacher_mode(self) -> None:
        self.unet.eval()

    def set_lora_mode(self, train: bool = False) -> None:
        # Compatibility shim: no adapter branch in this simple version.
        self.unet.train(train)

    def denoiser_freeze_all(self) -> None:
        self.unet.requires_grad_(False)

    def denoiser_unfreeze_adaptor(self) -> None:
        # Compatibility shim: there is no adapter here, so unfreeze the full UNet.
        self.unet.requires_grad_(True)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns image-space model inputs (no VAE, no latent compression)
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B, 3, H, W], got {tuple(images.shape)}")

        h, w = images.shape[-2:]
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Expected images with spatial size {(self.image_size, self.image_size)}, got {(h, w)}"
            )

        x = images.to(device=self.device, dtype=self.dtype)

        if self.image_range == "zero_one":
            x = x * 2.0 - 1.0
        elif self.image_range == "minus_one_one":
            pass
        else:
            raise ValueError(f"Unsupported image_range={self.image_range!r}")

        x = x.clamp(-1.0, 1.0)
        return x

    def sample_noisy_latent(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noises: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)

        if noises is None:
            noises = torch.randn_like(latents)
        else:
            noises = noises.to(device=self.device, dtype=self.dtype)

        timesteps = self._prepare_batch_timesteps(timesteps, batch_size=latents.shape[0])
        return self.scheduler.add_noise(latents, noises, timesteps)

    def predict_score(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict | None = None,
        role: str = "teacher",  # "teacher" or "lora"; accepted for compatibility
    ) -> torch.Tensor:
        """
        Returns the raw epsilon prediction from the image-space UNet.
        `batch` is ignored because this wrapper is unconditional.
        """
        if role == "teacher":
            self.set_teacher_mode()
        elif role == "lora":
            self.set_lora_mode(train=self.training)
        else:
            raise ValueError(f"Unsupported role={role!r}")

        noisy_latents = noisy_latents.to(device=self.device, dtype=self.dtype)
        batch_size = noisy_latents.shape[0]
        timesteps = self._prepare_batch_timesteps(timesteps, batch_size=batch_size)

        model_input = self.scheduler.scale_model_input(noisy_latents, timesteps)

        model_pred = self.unet(
            model_input,
            timesteps,
            return_dict=False,
        )[0]

        return model_pred

    def forward(self, method_str: str, **kwargs):
        if method_str == "predict_score":
            return self.predict_score(**kwargs)
        if method_str == "encode":
            return self.encode(**kwargs)
        if method_str == "sample_noisy_latent":
            return self.sample_noisy_latent(**kwargs)
        if method_str == "denoiser_freeze_all":
            return self.denoiser_freeze_all()
        if method_str == "denoiser_unfreeze_adaptor":
            return self.denoiser_unfreeze_adaptor()
        raise ValueError(f"Unsupported method_str={method_str!r}")


# Optional alias so old imports can keep working.
SDXLPriorWrapper = OpenAIImagePriorWrapper