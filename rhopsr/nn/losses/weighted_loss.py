import torch
import torch.nn as nn

class WeightedLoss(nn.Module):
    def __init__(self, base_loss_fn_str: str):

        super(WeightedLoss, self).__init__()

        loss_map = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'smooth_l1': nn.SmoothL1Loss,
            'bce': nn.BCEWithLogitsLoss,  # or nn.BCELoss
            'ce': nn.CrossEntropyLoss
        }

        if base_loss_fn_str not in loss_map:
            raise ValueError(f"Unsupported loss function: {base_loss_fn_str}")

        # Instantiate the loss function with reduction='none' for per-element loss
        self.base_loss_fn = loss_map[base_loss_fn_str](reduction='none')

    def forward(self, preds, targets, weights=None):
        """
        Args:
            preds (Tensor): Model predictions.
            targets (Tensor): Ground truth.
            weights (Tensor): Sample weights (shape: batch_size,)
        """
        # Compute per-element loss

        elementwise_loss = self.base_loss_fn(preds, targets)

        # Reduce over all but the batch dimension
        if elementwise_loss.dim() > 1:
            loss_per_sample = elementwise_loss.view(elementwise_loss.size(0), -1).mean(dim=1)
        else:
            loss_per_sample = elementwise_loss  # For 1D outputs

        if weights is None:
            # If no weights are provided, use uniform weights of 1
            weighted_loss = torch.mean(loss_per_sample)
        else:
            # Apply weights
            weighted_loss = torch.mean(loss_per_sample * weights)

        return weighted_loss
