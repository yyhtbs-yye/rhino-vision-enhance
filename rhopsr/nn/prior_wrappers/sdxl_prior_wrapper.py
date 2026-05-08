import torch
import torch.nn as nn

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig

class SDXLPriorWrapper(nn.Module):
    """
    Frozen SDXL prior wrapper with one trainable LoRA adapter branch.

    Design choices:
    - epsilon prediction only
    - DDPM forward noising
    - clean null-prompt fallback
    - proper SDXL micro-conditioning
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        variant: str | None = "fp16",
        use_safetensors: bool = True,
        image_range: str = "zero_one", 
        sample_posterior: bool = False,
        original_size: tuple[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        force_vae_upcast: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        adapter_name: str = "reg",
    ):
        super().__init__()

        self.image_range = image_range
        self.sample_posterior = sample_posterior
        self.original_size = original_size
        self.target_size = target_size
        self.crops_coords_top_left = crops_coords_top_left
        self.force_vae_upcast = force_vae_upcast
        self.adapter_name = adapter_name

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=use_safetensors,
        ).to(device)

        # Use a DDPM scheduler built from the SDXL scheduler config so the
        # training-time noise schedule stays aligned with the checkpoint config.
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type != "epsilon":
            raise ValueError(
                f"This wrapper only supports epsilon prediction, got prediction_type={prediction_type!r}."
            )

        self.vae, self.denoiser = pipe.vae, pipe.unet

        if self.force_vae_upcast:
            self.vae = self.vae.to(dtype=torch.float32)

        self.vae.eval()
        self.denoiser.eval()

        # Freeze base VAE and UNet.
        self.vae.requires_grad_(False)
        self.denoiser.requires_grad_(False)

        # Add one trainable LoRA adapter on the UNet attention projections.
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.denoiser.add_adapter(lora_config, adapter_name=adapter_name)
        self.denoiser.disable_adapters()

        # Freeze everything, then unfreeze only this adapter.
        self._set_only_adapter_trainable(adapter_name)

        # Cache null-prompt conditioning once.
        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt="", prompt_2="", device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )

        self.register_buffer("_null_prompt_embeds", prompt_embeds.detach(), persistent=False)
        self.register_buffer("_null_pooled_prompt_embeds", pooled_prompt_embeds.detach(), persistent=False)

        # Free unused pipeline parts.
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.tokenizer
        del pipe.tokenizer_2
        del pipe

    @property
    def device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.denoiser.parameters()).dtype

    def _set_only_adapter_trainable(self, adapter_name: str) -> None:
        for name, param in self.denoiser.named_parameters():
            is_target_adapter = ("lora_" in name) and (f".{adapter_name}." in name)
            param.requires_grad_(is_target_adapter)

    def _to_timestep_indices(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
        - float timesteps in [0, 1]
        - integer timesteps in [0, num_train_timesteps - 1]
        """
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)

        if timesteps.dtype.is_floating_point:
            timesteps = torch.round(timesteps * (num_train_timesteps - 1)).long()
        else:
            timesteps = timesteps.long()

        timesteps = timesteps.clamp(0, num_train_timesteps - 1)
        return timesteps.to(self.device)

    def _build_add_time_ids(
        self,
        batch: dict,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "add_time_ids" in batch and batch["add_time_ids"] is not None:
            add_time_ids = batch["add_time_ids"].to(device=self.device, dtype=dtype)
            if add_time_ids.shape[0] == 1:
                add_time_ids = add_time_ids.expand(batch_size, -1)
            elif add_time_ids.shape[0] != batch_size:
                raise ValueError(
                    f"add_time_ids batch mismatch: got {add_time_ids.shape[0]}, expected 1 or {batch_size}"
                )
            return add_time_ids

        original_size = batch.get("original_size", self.original_size or (height, width))
        target_size = batch.get("target_size", self.target_size or (height, width))
        crops = batch.get("crops_coords_top_left", self.crops_coords_top_left)

        add_time_ids = torch.tensor(
            [[
                int(original_size[0]),
                int(original_size[1]),
                int(crops[0]),
                int(crops[1]),
                int(target_size[0]),
                int(target_size[1]),
            ]],
            device=self.device,
            dtype=dtype,
        )
        return add_time_ids.expand(batch_size, -1)

    def set_teacher_mode(self) -> None:
        self.denoiser.eval()
        self.denoiser.disable_adapters()

    def set_lora_mode(self, train: bool = False) -> None:
        self.denoiser.set_adapter(self.adapter_name)
        self.denoiser.enable_adapters()
        self.denoiser.train(train)

    def denoiser_freeze_all(self) -> None:
        self.denoiser.requires_grad_(False)

    def denoiser_unfreeze_adaptor(self) -> None:
        self._set_only_adapter_trainable(self.adapter_name)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns latents scaled by VAE scaling_factor
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B, 3, H, W], got {tuple(images.shape)}")

        h, w = images.shape[-2:]
        if h % 8 != 0 or w % 8 != 0: raise ValueError(f"Expected H and W divisible by 8, got {(h, w)}")

        x = images.to(device=self.device)

        if self.image_range == "zero_one":
            x = x * 2.0 - 1.0
        elif self.image_range == "minus_one_one":
            pass
        else:
            raise ValueError(f"Unsupported image_range={self.image_range!r}")

        x = x.clamp(-1.0, 1.0)

        vae_input = x.float() if self.force_vae_upcast else x.to(dtype=self.vae.dtype)

        posterior = self.vae.encode(vae_input).latent_dist
        latents = posterior.sample() if self.sample_posterior else posterior.mode()
        latents = latents * self.vae.config.scaling_factor

        return latents.to(device=self.device, dtype=self.dtype)

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

        timesteps = self._to_timestep_indices(timesteps)
        return self.scheduler.add_noise(latents, noises, timesteps)

    def predict_score(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict | None = None,
        role: str = "teacher",  # "teacher" or "lora"
    ) -> torch.Tensor:
        """
        Returns the raw epsilon prediction from the UNet.
        No x0 conversion. No v conversion.
        """
        if role == "teacher":
            self.set_teacher_mode()
        elif role == "lora":
            self.set_lora_mode(train=self.training)
        else:
            raise ValueError(f"Unsupported role={role!r}")

        # The UNet forward expects inputs in the wrapper's device and dtype, so move and cast here.
        noisy_latents = noisy_latents.to(device=self.device, dtype=self.dtype)
        timesteps = self._to_timestep_indices(timesteps)

        batch_size = noisy_latents.shape[0]
        latent_h, latent_w = noisy_latents.shape[-2:]
        height, width = latent_h * 8, latent_w * 8

        prompt_embeds = batch.get("prompt_embeds", self._null_prompt_embeds)
        pooled_prompt_embeds = batch.get("pooled_prompt_embeds", self._null_pooled_prompt_embeds)

        add_time_ids = self._build_add_time_ids(
            batch=batch, batch_size=batch_size, height=height, width=width, dtype=pooled_prompt_embeds.dtype,
        )

        # Keeps the wrapper scheduler-compatible.
        model_input = self.scheduler.scale_model_input(noisy_latents, timesteps)

        model_pred = self.denoiser(
            model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
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