import torch
import torch.nn as nn

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig

class PriorWrapper(nn.Module, ):

    def __init__(self, prior_cfg):
        super().__init__()


    def apply_lora(self, lora_cfg):
        # Create a LoRA configuration and apply it to the UNet
        config = LoraConfig(
            r=lora_cfg.get('r', 4),
            lora_alpha=lora_cfg.get('lora_alpha', 16),
            init_scale=lora_cfg.get('init_scale', 0.01),
            target_modules=lora_cfg.get('target_modules', ['to_k', 'to_q', 'to_v', 'to_out.0']),
        )
        for name, param in self.unet.named_parameters():
            if any(target in name for target in config.target_modules):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)