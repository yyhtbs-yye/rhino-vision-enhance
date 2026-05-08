import torch

from rhcore.boats.base_boat import BaseBoat
from rhcore.utils.build_components import build_module
from rhtrain.utils.ddp_utils import move_to_device

class AdversarialRestorationBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        self.config = config
        self.gt_name = config['boat'].get('gt_name', 'gt')
        self.lq_name = config['boat'].get('lq_name', 'lq')

        self.lambda_weights = config['boat'].get(
            'hyperparameters', {}).get(
                'lambda_weights', 
                    {'pixel_loss': 1.0, 
                    'lpips_loss': 0.5, 
                    'g_adv': 0.02,
                    'd_adv': 0.1,
                    }
            )

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
        
        # Build loss fading modules based on hyperparameters
        self.build_loss_fading()

    @torch.no_grad()
    def generate(self, batch, use_clamp=True):

        lq = batch[self.lq_name]

        net = self.maybe_get_ema('net')
        gen_images = net(lq)
        
        return gen_images

    def superresolution_calc_losses(self, batch):

        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        losses = {}

        gen_images = self.models['net'](lq)

        # Pixel loss
        losses['pixel_loss'] = self.losses['pixel_loss'](gen_images, gt)

        # Perceptual loss
        w_lpips = self.lambda_weights['lpips_loss'] * (self.loss_fadeins['lpips_fadein'](self.global_step()))

        losses['lpips_loss'] = self.losses['lpips_loss'](gen_images, gt, normalize=True).mean()
        losses['w_lpips'] = w_lpips

        # Adversarial loss
        w_adv = self.lambda_weights['g_adv'] * (self.loss_fadeins['adv_fadein'](self.global_step()))

        g_real = self.models['critic'](gen_images)
        losses['g_adv'] = self.losses['critic'](g_real, None)
        losses['w_adv'] = w_adv

        # Total generator loss
        losses['g_loss'] = (self.lambda_weights['pixel_loss'] * losses['pixel_loss'] +
                            w_lpips * losses['lpips_loss'] +
                            w_adv * losses['g_adv'])
        

        return losses

    def d_step_calc_losses(self, batch):
        
        gt = batch[self.gt_name]
        lq = batch[self.lq_name]

        losses = {}

        with torch.no_grad():
            gen_images = self.models['net'](lq)

        d_real = self.models['critic'](gt)
        d_fake = self.models['critic'](gen_images)

        w_adv = self.lambda_weights['d_adv'] * (self.loss_fadeins['adv_fadein'](self.global_step()))

        losses['d_loss'] = w_adv * self.losses['critic'](d_real, d_fake)

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
    

    def build_loss_fading(self):

        hyperparameters = self.config['boat'].get('hyperparameters', {})

        self.loss_fadeins = {it: build_module(hyperparameters[it]) 
                       for it in hyperparameters if 'fadein' in it}