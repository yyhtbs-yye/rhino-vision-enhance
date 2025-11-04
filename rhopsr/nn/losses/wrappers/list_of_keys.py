import torch
import torch.nn as nn
from rhcore.utils.build_components import build_module

class ListOfKeys(nn.Module):

    def __init__(self, module, keys=["preds", "targets", "weights"]):
        super().__init__()

        self.module = module
        self.keys = keys

    def forward(self, results) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        loss_val = self.module(*[results.get(key, None) for key in self.keys])

        return loss_val
