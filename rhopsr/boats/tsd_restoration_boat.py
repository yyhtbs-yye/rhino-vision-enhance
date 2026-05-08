import torch
import torch.nn as nn
import torch.nn.functional as F
from rhcore.boats.base_boat import BaseBoat
from rhtrain.utils.ddp_utils import move_to_device
from rhopsr.boats.mixins.uncond_generate_mixin import UncondGenerateMixin
from rhopsr.boats.target_score_distillation.dasm import DASMConfig, DistributionAwareSamplingModule

class TargetScoreDistillationRestorationBoat(BaseBoat, UncondGenerateMixin):

    def __init__(self, config=None):
        super().__init__(config=config or {})

        boat_cfg = config.get('boat', {})
        self.config = config

        self.gt_name = boat_cfg.get('gt_name', 'gt')
        self.lq_name = boat_cfg.get('lq_name', 'lq')

        self.lambda_weights = boat_cfg.get('hyperparameters', {}).get(
            'lambda_weights',
            {
                'pixel_loss': 1.0,
                'lpips_loss': 0.5,
                'vsd_loss': 0.8,
                'tsd_all_loss': 0.2,
                'adaptor_loss': 1.0,
            }
        )

        self.prior_cfg = boat_cfg.get('prior', {})

        self.ordered_groups = boat_cfg.get('ordered_groups', None)
        if self.ordered_groups is None or len(self.ordered_groups) == 0:
            self.ordered_groups = [
                {
                    'boat_loss_method_str': 'student_step_calc_losses',
                    'target_loss_name': 'student_loss',
                    'models': ['net'],
                    'optimizers': ['net'],
                    'train_interval': 1,
                },
                {
                    'boat_loss_method_str': 'prior_adapter_step_calc_losses',
                    'target_loss_name': 'prior_adapter_loss',
                    'models': ['teacher'],
                    'optimizers': ['teacher'],
                    'train_interval': 1,
                },
            ]

        self.dasm = DistributionAwareSamplingModule(
            DASMConfig(
                enable_after_step=100,   # optional warm-up
                lambda_vsd=0.7,
                lambda_tsm=0.3,
                total_steps=4,
                start_index_min=50,
                start_index_max=950,
                step_size=50,
                use_random_bias=True,
                trajectory_weights=(1.0, 0.3, 0.3, 0.3),
                teacher_role="teacher",
                adapter_role="lora",
            )
        )

    def student_step_calc_losses(self, batch):
        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        student = self.models['net']
        teacher = self.models['teacher']

        losses = {}

        # Forward pass through the student model to get the restored image
        restored = student(lq)

        # Reconstruction loss in pixel space
        losses['pixel_loss'] = self.losses['pixel_loss'](restored, gt)
        pixel_loss = self.lambda_weights['pixel_loss'] * losses['pixel_loss']

        # Freeze the teacher's weights, include the adapter if it exists.
        # This ensures that the teacher provides stable guidance to the student.
        teacher('denoiser_freeze_all')
        
        # Use the DASM module to compute the surrogate losses for VSD and TSM
        dasm_out = self.dasm.student_surrogate_losses(
            teacher=teacher,
            prediction=restored,
            target=gt,
            batch=batch,
            global_step=self.get_global_step(),
        )

        # Combine the surrogate losses with the pixel loss to form the total student loss
        losses["vsd_loss"] = dasm_out["vsd_surrogate_loss"]
        losses["tsm_loss"] = dasm_out["tsm_surrogate_loss"]

        losses["prior_regularization_loss"] = (
            self.lambda_weights["tsd_all_loss"] * dasm_out["tsd_surrogate_loss"]
        )

        losses['student_loss'] = pixel_loss + losses['prior_regularization_loss']

        return losses

    def prior_adapter_step_calc_losses(self, batch):
        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        teacher = self.models['teacher']

        losses = {}

        with torch.no_grad():
            restored = self.models['net'](lq)
            restored_latents = teacher('encode', images=restored)

        batch_size = gt.size(0)
        t = torch.empty(batch_size, device=gt.device).uniform_(
            self.prior_cfg.get('t_min', 0.02),
            self.prior_cfg.get('t_max', 0.12),
        )

        noises = torch.randn_like(restored_latents)

        noisy_restored_latents = teacher('sample_noisy_latent', latents=restored_latents, timesteps=t, noises=noises)

        target_scores = noises

        teacher('denoiser_unfreeze_adaptor')

        adapter_prediction = teacher('predict_score', noisy_latents=noisy_restored_latents, timesteps=t, batch=batch, role="lora")

        losses['adaptor_loss'] = F.mse_loss(
            adapter_prediction, target_scores, reduction='mean'
        )

        losses['prior_adapter_loss'] = (
            self.lambda_weights['adaptor_loss'] * losses['adaptor_loss']
        )

        return losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):
        batch = move_to_device(batch, self.device)

        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        input = batch[self.lq_name]
        net = self.maybe_get_ema('net')

        sr_img = self.generate(net, input)

        valid_output = {
            'input': sr_img,
            'target': gt,
        }

        metrics = self.calc_metrics(valid_output)

        named_imgs = {
            'low quality': lq,
            'high quality': gt,
            'enhanced': sr_img,
        }

        return metrics, named_imgs