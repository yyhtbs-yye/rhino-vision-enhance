import torch

from rhcore.boats.base_boat import BaseBoat
from rhtrain.utils.ddp_utils import move_to_device

class AdversarialPriorRestorationBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        self.gt_name = config['boat'].get('gt_name', 'gt')
        self.lq_name = config['boat'].get('lq_name', 'lq')

        self.lambda_weights = config['boat'].get(
            'hyperparameters', {}).get(
                'lambda_weights', 
                    {'pixel_loss': 1.0, 
                    'lpips_loss': 0.5, 
                    'prior_loss': 0.1,
                    'g_adv': 0.02,
                    'd_adv': 0.1,
                    }
            )
        
        self.prior_cfg = config['boat'].get('prior', {})

        self.ordered_groups = config['boat'].get('ordered_groups', None)
        if self.ordered_groups is None or len(self.ordered_groups) == 0:
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
        from rhopsr.nn.prior_wrappers.sdxl_prior_wrapper import SDXLPriorWrapper
        self.pretrained['prior'] = SDXLPriorWrapper()

    @torch.no_grad()
    def generate(self, batch, use_clamp=True):

        lq = batch[self.lq_name]

        net = self.maybe_get_ema('net')
        gen_images = net(lq)
        
        return gen_images


    def _sample_prior_timesteps(self, batch_size, device):
        t_min = self.prior_cfg.get('t_min', 0.02)
        t_max = self.prior_cfg.get('t_max', 0.12)

        return torch.empty(batch_size, device=device).uniform_(t_min, t_max)
    

    def superresolution_calc_losses(self, batch):

        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        losses = {}

        gen_images = self.models['net'](lq)

        # Pixel loss
        losses['pixel_loss'] = self.losses['pixel_loss'](gen_images, gt)

        # Perceptual loss
        losses['lpips_loss'] = self.losses['lpips_loss'](gen_images, gt, normalize=True).mean()

        # Adversarial loss
        g_real = self.models['critic'](gen_images)
        losses['g_adv'] = self.losses['critic'](g_real, None)

        # Prior loss
        batch_size = gt.size(0)
        t = self._sample_prior_timesteps(batch_size, gt.device)

        with torch.no_grad():
            pred_gt, noises = self.pretrained['prior'].calc_target(gt, t)
            pred_gt = pred_gt.detach()
        
        pred_gen, _ = self.pretrained['prior'].calc_target(gen_images, t, noises)

        losses['prior_loss'] = self.losses['prior_loss'](pred_gen, pred_gt)

        losses['g_loss'] = (self.lambda_weights['pixel_loss'] * losses['pixel_loss'] +
                            self.lambda_weights['lpips_loss'] * losses['lpips_loss'] +
                            self.lambda_weights['g_adv'] * losses['g_adv'] +
                            self.lambda_weights['prior_loss'] * losses['prior_loss'])

        return losses


    def d_step_calc_losses(self, batch):
        
        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        losses = {}

        with torch.no_grad():
            gen_images = self.models['net'](lq)

        d_real = self.models['critic'](gt)
        d_fake = self.models['critic'](gen_images)

        losses['d_loss'] = self.lambda_weights['d_adv'] * self.losses['critic'](d_real, d_fake)

        return losses


    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        with torch.no_grad():
            
            sr_img = self.generate(batch)

            valid_output = {'input': sr_img, 'target': gt}

            metrics = self.calc_metrics(valid_output)

            named_imgs = {'low quality': lq,
                          'high quality': gt, 
                          'enhanced': sr_img, 
                          }

        return metrics, named_imgs
    