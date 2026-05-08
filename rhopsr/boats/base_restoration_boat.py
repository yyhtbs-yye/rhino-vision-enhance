import torch

from rhcore.boats.base_boat import BaseBoat
from rhtrain.utils.ddp_utils import move_to_device
from einops import rearrange

class BaseRestorationBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        self.gt_name = config['boat'].get('gt_name', 'gt')
        self.lq_name = config['boat'].get('lq_name', 'lq')

        if self.ordered_groups is None:
            self.ordered_groups = [
                {
                    'boat_loss_method_str': 'd_step_calc_losses',
                    'target_loss_name': 'd_loss',
                    'models': ['critic'],
                    'optimizers': ['critic'],
                    'train_interval': 1,
                },
                {
                    'boat_loss_method_str': 'superresolution_calc_losses',
                    'target_loss_name': 'g_loss',
                    'models': ['net'],
                    'optimizers': ['net'],
                    'train_interval': 1,
                },
            ]

        self.loss_debug = True

        self.log_excludes = ['mu_r0', 'mu_rD', 'mu_f0', 'mu_fD', 'mD', 'mu_fDG', ]

    @torch.no_grad()
    def generate(self, batch, use_clamp=True):

        lq = batch[self.lq_name]

        net = self.maybe_get_ema('net')
        gen_images = net(lq)
        
        return gen_images

    def superresolution_calc_losses(self, batch):

        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        gen_images = self.models['net'](lq)

        # Pixel loss
        losses['net'] = self.losses['net'](gen_images, gt)

        # Adversarial loss
        g_real = self.models['critic'](gen_images)

        losses['g_adv'] = self.losses['critic']({'real': g_real, 'fake': None, **batch})

        losses['total_loss'] += losses['net']
        losses['total_loss'] += losses['g_adv']

        return losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        with torch.no_grad():
            
            gen_images = self.generate(lq)

            if gt.ndim == 5:
                gt = rearrange(gt, 'b t c h w -> (b t) c h w')
                gen_images = rearrange(gen_images, 'b t c h w -> (b t) c h w')
                lq = rearrange(lq, 'b t c h w -> (b t) c h w')


            valid_output = {'preds': gen_images, 'targets': gt,}

            metrics = self.calc_metrics(valid_output)

            named_imgs = {'high res': gt, 
                          'super res': gen_images, 
                          'low res': lq,}

        return metrics, named_imgs
    