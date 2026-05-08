import torch
import torch.nn as nn

from diffusers import DDPMScheduler, PixArtAlphaPipeline
from peft import LoraConfig, inject_adapter_in_model


class PixArtPriorWrapper(nn.Module):
    """
    Frozen PixArt-Alpha wrapper with one trainable LoRA adapter branch.

    Design choices:
    - epsilon prediction only
    - DDPM forward noising
    - clean null-prompt fallback
    - correct PixArt conditioning:
        * prompt_embeds + prompt_attention_mask
        * optional resolution/aspect_ratio only when the checkpoint uses them
    """

    def __init__(
        self,
        model_id: str = "PixArt-alpha/PixArt-XL-2-256x256",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        variant: str | None = None,
        use_safetensors: bool = True,
        image_range: str = "zero_one",          # "zero_one" or "minus_one_one"
        sample_posterior: bool = False,
        force_vae_upcast: bool = False,
        max_sequence_length: int = 120,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        adapter_name: str = "reg",
    ):
        super().__init__()

        self.image_range = image_range
        self.sample_posterior = sample_posterior
        self.force_vae_upcast = force_vae_upcast
        self.max_sequence_length = max_sequence_length
        self.adapter_name = adapter_name

        pipe_kwargs = dict(
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        )
        if variant is not None:
            pipe_kwargs["variant"] = variant

        pipe = PixArtAlphaPipeline.from_pretrained(model_id, **pipe_kwargs).to(device)

        # Keep training-time forward noising aligned with the checkpoint scheduler config.
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type != "epsilon":
            raise ValueError(
                f"This wrapper only supports epsilon prediction, got prediction_type={prediction_type!r}."
            )

        self.vae = pipe.vae
        self.denoiser = pipe.transformer

        if self.force_vae_upcast:
            self.vae = self.vae.to(dtype=torch.float32)

        self.vae.eval()
        self.denoiser.eval()

        # Freeze base VAE and transformer.
        self.vae.requires_grad_(False)
        self.denoiser.requires_grad_(False)

        # Inject one trainable LoRA adapter into the PixArt attention projections.
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        inject_adapter_in_model(lora_config, self.denoiser, adapter_name=adapter_name)

        # Freeze everything, then unfreeze only this adapter.
        self._set_only_adapter_trainable(adapter_name)
        self._set_adapter(adapter_name)
        self._set_adapters_enabled(False)

        # Cache null-prompt conditioning once.
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
                prompt="",
                do_classifier_free_guidance=False,
                negative_prompt="",
                num_images_per_prompt=1,
                device=device,
                clean_caption=False,
                max_sequence_length=max_sequence_length,
            )

        self.register_buffer("_null_prompt_embeds", prompt_embeds.detach(), persistent=False)
        self.register_buffer("_null_prompt_attention_mask", prompt_attention_mask.detach(), persistent=False)

        # Free unused pipeline parts.
        del pipe.text_encoder
        del pipe.tokenizer
        del pipe

    @property
    def device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.denoiser.parameters()).dtype

    def _iter_tuner_layers(self):
        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.denoiser.named_modules():
            if isinstance(module, BaseTunerLayer):
                yield module

    def _set_adapter(self, adapter_name: str) -> None:
        found = False
        for module in self._iter_tuner_layers():
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_name)
            else:
                module.active_adapter = adapter_name
            found = True

        if not found:
            raise RuntimeError("No PEFT tuner layers were found on the PixArt transformer.")

    def _set_adapters_enabled(self, enabled: bool) -> None:
        for module in self._iter_tuner_layers():
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=enabled)
            else:
                # Older PEFT fallback.
                module.disable_adapters = not enabled

    def _set_only_adapter_trainable(self, adapter_name: str) -> None:
        tag = f".{adapter_name}."
        for name, param in self.denoiser.named_parameters():
            is_target_adapter = ("lora_" in name) and (tag in name)
            param.requires_grad_(is_target_adapter)

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

    def set_teacher_mode(self) -> None:
        self.denoiser.eval()
        self._set_adapters_enabled(False)

    def set_lora_mode(self, train: bool = False) -> None:
        self._set_adapter(self.adapter_name)
        self._set_adapters_enabled(True)
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
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"Expected H and W divisible by 8, got {(h, w)}")

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

        timesteps = self._prepare_batch_timesteps(timesteps, batch_size=latents.shape[0])
        return self.scheduler.add_noise(latents, noises, timesteps)

    def predict_score(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict | None = None,
        role: str = "teacher",  # "teacher" or "lora"
    ) -> torch.Tensor:
        """
        Returns the raw epsilon prediction from the PixArt transformer.
        No x0 conversion. No v conversion.

        Expected text conditioning in `batch`:
          - prompt_embeds: [B, S, C] or [1, S, C]
          - prompt_attention_mask: [B, S] or [1, S]

        Optional micro-conditioning in `batch`:
          - added_cond_kwargs={"resolution": ..., "aspect_ratio": ...}
          - or resolution / aspect_ratio directly
        """
        batch = batch or {}

        if role == "teacher":
            self.set_teacher_mode()
        elif role == "lora":
            self.set_lora_mode(train=self.training)
        else:
            raise ValueError(f"Unsupported role={role!r}")

        noisy_latents = noisy_latents.to(device=self.device, dtype=self.dtype)
        batch_size = noisy_latents.shape[0]

        latent_h, latent_w = noisy_latents.shape[-2:]
        height, width = latent_h * 8, latent_w * 8

        prompt_embeds = batch.get("prompt_embeds", self._null_prompt_embeds)
        prompt_attention_mask = batch.get("prompt_attention_mask", self._null_prompt_attention_mask)

        added_cond_kwargs = {"resolution": 256 * torch.ones(batch_size, height, width, dtype=torch.long).to(self.device), 
                             "aspect_ratio": torch.ones(batch_size, 1, dtype=self.dtype).to(self.device)}

        model_input = self.scheduler.scale_model_input(noisy_latents, timesteps)

        model_pred = self.denoiser(
            model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # PixArt checkpoints can predict learned sigma alongside epsilon.
        # For epsilon-target training, keep only the epsilon half.
        latent_channels = noisy_latents.shape[1]
        out_channels = int(getattr(self.denoiser.config, "out_channels", model_pred.shape[1]))

        if out_channels // 2 == latent_channels:
            model_pred = model_pred.chunk(2, dim=1)[0]
        elif out_channels != latent_channels:
            raise ValueError(
                "Unsupported PixArt output channel configuration: "
                f"transformer out_channels={out_channels}, latent_channels={latent_channels}"
            )

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
