import torch
import torch.nn.functional as F

from rhcore.utils.build_components import build_module

from rhtrain.utils.ddp_utils import move_to_device

from rhopsr.boats.base_restoration_boat import BaseRestorationBoat
from rhopsr.nn.utils.gaussian_blur import separable_gaussian_blur
from rhopsr.schedulers.progressive_kimgs_curriculum import ProgressiveKimgsCurriculum

class ProgressiveGaussianDeblurRestorationBoat(BaseRestorationBoat):
    """
    Drop-in replacement boat that applies a PGGAN-like kimgs curriculum
    by progressively sharpening the training target via low-pass filtering.

    Config (example):
    curriculum = {
        "sigmas": [6.0, 3.0, 1.5, 0.75, 0.0],
        "stage_kimgs": [200, 200, 400, 400, 600],  # total 1.8M images
        "fade_fraction": 0.5
    }
    """

    def __init__(self, config={}):
        super().__init__(config)

        hps = self.boat_config.get('hyperparameters', {})

        # Curriculum config
        curr_cfg = hps.get('curriculum', None)

        assert curr_cfg is not None, "ProgressiveGaussianDeblurRestorationBoat requires a 'curriculum' config in 'hyperparameters'."

        self.num_stages = len(curr_cfg['sigmas'])
        self.curriculum = ProgressiveKimgsCurriculum(
            sigmas=curr_cfg['sigmas'],
            stage_kimgs=curr_cfg['stage_kimgs'],
            fade_fraction=curr_cfg.get('fade_fraction', 0.5),
        )

        self.upscale = build_module(self.boat_config.get('upscale', {}))

        # Track world size to estimate global kimgs in DDP
        self._world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                self._world_size = torch.distributed.get_world_size()
            except Exception:
                self._world_size = 1

    def predict(self, x):
        
        copy = x

        _, _, alpha, T = self.curriculum.current_blend()

        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
        # disable grad 
        with torch.no_grad():
            # Forward except last stage, a discrete difference equation t=0, 1, ..., T-1
            for t in range(T): 
                r = network_in_use(x, copy, t, mode='residual')
                x = x + r
        # Last stage with blending, time T
        restored = network_in_use(x, copy, T, mode='blend', alpha=alpha)

        return restored
    
    def training_calc_losses(self, batch):
        gt = batch['gt']
        lq = separable_gaussian_blur(gt, sigma=8.0)
        batch_size = gt.size(0)

        # Advance curriculum by global images processed
        self.curriculum.add_images(n_images=batch_size, world_size=self._world_size)
        prev_sigma, curr_sigma, alpha, T = self.curriculum.current_blend()

        x = lq
        # Forward except last stage, a discrete difference equation t=0, 1, ..., T-1
        for t in range(T): 
            r = self.models['net'](x, lq, t, mode='residual')
            x = x + r
        # Last stage with blending, time T
        restored = self.models['net'](x, lq, T, mode='blend', alpha=alpha)

        # Build progressively-sharpened target:
        #   target = (1-alpha)*blur(gt, prev_sigma) + alpha*blur(gt, curr_sigma)
        with torch.no_grad():
            tgt_prev = separable_gaussian_blur(gt, prev_sigma)
            tgt_curr = separable_gaussian_blur(gt, curr_sigma)
            target = (1.0 - alpha) * tgt_prev + alpha * tgt_curr
            # Stop grads & keep numerics clean
            target = target.detach()

        train_output = {
            'preds': restored,
            'targets': target,
            'weights': torch.ones(batch_size, device=self.device),
            # You may want the raw gt in losses/metrics; keep it around:
            'gt': gt,
            'lq': lq,
            # Expose curriculum state for logging
            'curriculum_prev_sigma': torch.tensor(prev_sigma, device=self.device),
            'curriculum_curr_sigma': torch.tensor(curr_sigma, device=self.device),
            'curriculum_alpha': torch.tensor(alpha, device=self.device),
            'curriculum_kimgs': torch.tensor(self.curriculum.kimgs, device=self.device),
            **batch
        }

        losses = {'total_loss': torch.tensor(0.0, device=self.device)}
        # Your loss module should read 'preds' vs 'targets'
        losses['total_loss'] = self.losses['net'](train_output)

        # Optional: add a small consistency term toward raw gt after halfway through training
        # to gently bias the model as it nears the final stage. Uncomment if desired.
        # if self.curriculum.kimgs > 0.5 * self.curriculum.total_kimgs:
        #     with torch.no_grad():
        #         w = 0.05
        #     losses['gt_bias'] = w * F.l1_loss(restored, gt)
        #     losses['total_loss'] += losses['gt_bias']

        # You can also attach debug info for loggers
        losses.update({
            'curriculum_prev_sigma': train_output['curriculum_prev_sigma'],
            'curriculum_curr_sigma': train_output['curriculum_curr_sigma'],
            'curriculum_alpha': train_output['curriculum_alpha'],
            'curriculum_kimgs': train_output['curriculum_kimgs'],
        })

        return losses

    # ---------- Validation (shows where we are in the curriculum) ----------
    def validation_step(self, batch, batch_idx, epoch):
        batch = move_to_device(batch, self.device)
        gt = batch['gt']
        lq = separable_gaussian_blur(gt, sigma=8.0)

        with torch.no_grad():
            restored = self.predict(lq)

            # Also show the *current* progressive target for visibility
            prev_sigma, curr_sigma, alpha,_ = self.curriculum.current_blend()
            progressive_target = (1.0 - alpha) * separable_gaussian_blur(gt, prev_sigma) + \
                                 alpha * separable_gaussian_blur(gt, curr_sigma)

            valid_output = {'preds': restored, 'targets': gt}
            metrics = self._calc_metrics(valid_output)

            named_imgs = {
                'high res (gt)': gt,
                'progressive target': progressive_target,
                'restored': restored,
                'input (lq)': lq,
            }

        return metrics, named_imgs

    # ---------- (Optional) Save/Load curriculum state with checkpoints ----------
    def state_dict(self):
        s = super().state_dict()
        s['curriculum_state'] = self.curriculum.state_dict()
        return s

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        cur = state_dict.get('curriculum_state', None)
        if cur is not None:
            self.curriculum.load_state_dict(cur)
