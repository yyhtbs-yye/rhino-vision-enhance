import torch

from rhcore.boats.base_boat import BaseBoat
from rhtrain.utils.ddp_utils import move_to_device

from einops import rearrange

class BaseRestorationBoat(BaseBoat):

    def maybe_get_ema(self, name):
        return self.models[f'{name}_ema'] if f'{name}_ema' in self.models and self.use_ema else self.models[name]

    @torch.no_grad()
    def predict(self, lq):
        
        net = self.maybe_get_ema('net')
            
        preds = net(lq)
        
        return preds

    def training_calc_losses(self, batch):

        gt = batch['gt']
        lq = batch['lq']

        preds = self.models['net'](lq)
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](preds, gt)
        losses['total_loss'] += losses['net']

        return losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']
        lq = batch['lq']

        with torch.no_grad():
            
            preds = self.predict(lq)

            if gt.ndim == 5:
                gt = rearrange(gt, 'b t c h w -> (b t) c h w')
                preds = rearrange(preds, 'b t c h w -> (b t) c h w')
                lq = rearrange(lq, 'b t c h w -> (b t) c h w')


            valid_output = {'preds': preds, 'targets': gt,}

            metrics = self.calc_metrics(valid_output)

            named_imgs = {'high res': gt, 
                          'super res': preds, 
                          'low res': lq,}

        return metrics, named_imgs
    