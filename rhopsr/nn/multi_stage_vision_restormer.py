import copy
import torch
import torch.nn as nn
from rhopsr.nn.vision_restormer import VisionRestormer

from rhcore.utils.build_components import build_module

class MultiStageVisionRestormer(nn.Module):
    """
    Minimal outer wrapper:
      - Holds a ModuleList of VisionRestormer, one per stage index t.
      - Exposes self.models['net'] so you can call:
            r = self.models['net'](x, t, mode='residual')
            out = self.models['net'](x, T, mode='blend', alpha=alpha)

    Assumptions:
      - Each VisionRestormer in self.nets ONLY accepts `x` and returns a residual.
      - Blending is performed here as: out = x + alpha * residual_T(x).
    """
    def __init__(self, base_config: dict, num_stages: int):
        super().__init__()
        self.nets = nn.ModuleList([
            build_module(copy.deepcopy(base_config)) for _ in range(num_stages)])
        # Expose the exact interface expected by caller:
        self.models = {'net': self}

    def forward(self, x: torch.Tensor, base: torch.Tensor, t: int = None, alpha: float = 1.0, **kwargs):
        """
        Routed call used by the outside loop.

        Args (by convention from caller):
            x: input tensor
            t: stage index (int)
            alpha: blend factor used only when mode='blend'
            mode (in kwargs): 'residual' or 'blend' (defaults to 'residual')

        Returns:
            - If mode='residual': residual r_t(x)
            - If mode='blend'   : x + alpha * r_t(x)
        """
        mode = kwargs.get('mode', 'residual')
        net = self.nets[t]            # assume valid t, no checks per request
        r = net(torch.cat((x, base), dim=1))                    # each net only takes x and returns residual

        if mode == 'blend':
            return x + alpha * r
        # default: residual branch
        return r

# -------------- tiny usage example --------------
if __name__ == "__main__":
    # assumes VisionRestormer is defined/imported
    config = {
        "shared": {"default_norm": "ln", "default_acti": "relu", "output_norm": "none", "output_acti": "none"},
        "norm": {"path": "torch.nn", "name": "Identity", "params": {}},
        "pos_drop": {"path": "torch.nn", "name": "Identity", "params": {}},
        "stem": {"path": "rhopsr.nn.agile_gpt.unet", "name": "UNetStem", "params": {"in_ch": 3, "out_ch": 64, "n_convs": 2}},
        "body": {"path": "rhopsr.nn.agile_gpt.unet", "name": "UNetBodyNoResize",
                 "params": {"channels": 64, "depth": 3, "num_blocks_per_level": 2, "dilations": [1, 2, 3]}},
        "head": {"path": "rhopsr.nn.agile_gpt.unet", "name": "UNetHead", "params": {"in_ch": 64, "out_ch": 3, "residual": True}},
    }

    T = 3
    model = MultiStageVisionRestormer(config, num_stages=T+1)

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        for t in range(T):
            r = model.models['net'](x, t, mode='residual')
            x = x + r
        restored = model.models['net'](x, T, mode='blend', alpha=0.7)

    print(restored.shape)  # [2, 3, 128, 128]
