import torch

from rhcore.boats.base_boat import BaseBoat
from rhcore.utils.build_components import build_module
from rhtrain.utils.ddp_utils import move_to_device

class BaseRestorationBoat(BaseBoat):

    def predict(self, lq):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
            
        restored = network_in_use(lq)
        
        return restored

    def training_calc_losses(self, batch):

        gt = batch['gt']
        lq = batch['lq']

        batch_size = gt.size(0)

        restored = self.models['net'](lq)
        
        train_output = {
            'preds': restored,
            'targets': gt,
            'weights': torch.ones(batch_size, device=self.device),
            **batch
        }
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses

    def training_step(self, batch, batch_idx, epoch, *, scaler=None):
        
        active_keys = list(self.optimizers.keys())
        
        micro_batches = self._split_batch(batch, self.total_micro_steps)

        self._zero_grad(active_keys, set_to_none=True)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_losses = self.training_calc_losses(micro_batch)
            micro_losses_list.append(micro_losses)

            if isinstance(self.target_loss_key, str):
                micrompathloss = micro_losses[self.target_loss_key] / self.total_micro_steps
            elif isinstance(self.target_loss_key, list) or isinstance(self.target_loss_key, tuple):
                micrompathloss = [micro_losses[k] / self.total_micro_steps for k in self.target_loss_key]

            self.training_backpropagation(micrompathloss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, active_keys)
        
        self._update_ema()

        self.training_lr_scheduling_step(active_keys)

        return self._aggregate_loss_dicts(micro_losses_list)

    # ------------------------------------ Visualization ---------------------------------------------

    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']
        lq = batch['lq']

        with torch.no_grad():

            restored = self.predict(lq)

            valid_output = {'preds': restored, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'high res': gt, 'super res': restored, 'low res': lq,}

        return metrics, named_imgs
    