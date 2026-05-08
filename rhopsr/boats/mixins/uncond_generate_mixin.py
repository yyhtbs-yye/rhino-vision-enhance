import torch

class UncondGenerateMixin:

    @torch.no_grad()
    def generate(self, net, input, use_clamp=True, clamp_min=0.0, clamp_max=1.0):

        output = net(input)
        
        if use_clamp:
            output = output.clamp(min=clamp_min, max=clamp_max)
        
        return output