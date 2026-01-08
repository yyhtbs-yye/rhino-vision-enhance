import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from rhcore.utils.build_components import build_module
from rhcore.nn.ops.channel_layer_norm import ChannelLayerNorm2d

class VisionRestormer(nn.Module):

    def __init__(self, config): 
        
        super(VisionRestormer, self).__init__()

        shared_config = config['shared']
        self.default_norm = shared_config['default_norm']
        self.default_acti = shared_config['default_acti']
        self.output_norm = shared_config['output_norm']
        self.output_acti = shared_config['output_acti']

        self.norm = build_module(config['norm'])
        self.pos_drop = build_module(config['pos_drop'])

        config['stem']['norm'] = self.default_norm
        config['stem']['acti'] = self.default_acti
        self.stem = build_module(config['stem'])

        config['body']['norm'] = self.default_norm
        config['body']['acti'] = self.default_acti
        self.body = build_module(config['body'])


        config['head']['norm'] = self.output_norm
        config['head']['acti'] = self.output_acti
        self.head = build_module(config['head'])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, ChannelLayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):

        x = self.stem(x, **kwargs)                    # B C H W

        x = self.pos_drop(x)                        # B C H W

        x = self.body(x, **kwargs)                    # B C H W

        x = self.norm(x)                            # B C H W

        x = self.head(x, **args)                    # B C H W

        return x

if __name__ == "__main__":
    # Simple test
    config = {
        "shared": {
            "default_norm": "ln",        # or "bn" / "gn" / callable / module
            "default_acti": "relu",
            "output_norm": "none",
            "output_acti": "none",
        },
        "norm":      {"path": "torch.nn", 
                      "name": "Identity",
                      "params": {}},   # overall post-body norm (can be "Identity")

        "pos_drop":  {"path": "torch.nn",
                      "name": "Identity",
                      "params": {}},   # or Dropout2d/DropPath from your registry

        "stem": {"path": "rhopsr.nn.agile_gpt.unet",
                 "name": "UNetStem", 
                 "params": {
                     "in_ch": 3, "out_ch": 64, "n_convs": 2}
                 },

        "body": {"path": "rhopsr.nn.agile_gpt.unet",
                 "name": "UNetBodyNoResize", 
                 "params": {
                    "channels": 64, "depth": 3,
                    "num_blocks_per_level": 2, "dilations": [1, 2, 3]},
                 },
                
        "head": {"path": "rhopsr.nn.agile_gpt.unet",
                 "name": "UNetHead", 
                 "params": {"in_ch": 64, "out_ch": 3, "residual": True},
                 },
    }

    model = VisionRestormer(config)
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(y.shape)  # expect [2, 3, 128, 128]