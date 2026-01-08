import torch
import torch.nn as nn
import torch.nn.functional as F

from rhopsr.nn.model_zoo.basicvsr.optical_flow.utils import flow_warp
from rhopsr.nn.model_zoo.basicvsr.optical_flow.spynet import SpyNet

from rhopsr.nn.model_zoo.basicvsr.basicvsrpp import ConvResidualBlocks, Upsampler

class BasicVSRMinusMinus(nn.Module):

    def __init__(self, mid_channels=64, num_blocks=7, scale=4, spynet_path=None):

        super().__init__()
        self.mid_channels = mid_channels

        # optical flow
        self.spynet = SpyNet(spynet_path)
        self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.backbone[module] = ConvResidualBlocks(2 * mid_channels, mid_channels, num_blocks)
        
        self.upsampler = Upsampler(mid_channels, scale=scale)

        # self.reset_parameters()

    def reset_parameters(self):
        for name, m in self.named_modules():
            # init weights except for modules under "spynet"
            if isinstance(m, nn.Conv2d) and 'spynet' not in name:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def compute_flow(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop1 = flows.new_zeros(n, self.mid_channels, h, w)

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            feat_prop1_align = feat_prop1
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]

                feat_prop1_align = flow_warp(feat_prop1, flow_n1.permute(0, 2, 3, 1))

            # concatenate and residual blocks
            feat = [feat_current, feat_prop1_align]

            feat = torch.cat(feat, dim=1)
            feat_prop1 = feat_current + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop1)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()

        feats = {}
        # compute spatial features
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        flows_forward, flows_backward = self.compute_flow(lqs)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)

        return self.upsampler(lqs, feats)
