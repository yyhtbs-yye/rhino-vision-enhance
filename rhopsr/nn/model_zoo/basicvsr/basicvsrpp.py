import torch
import torch.nn as nn
import torch.nn.functional as F

from .optical_flow.utils import flow_warp
from .optical_flow.spynet import SpyNet

from rhcore.nn.utils.make_layers import make_layers
from .basics.residual_blocks import ResidualBlockNoBN

from .basics.dcn_blocks import SecondOrderDeformableAlignment

class ConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), 
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layers(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

class Upsampler(nn.Module):

    def __init__(self, mid_channels=64, scale=4):
        super().__init__()
        # upsampling module
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lqs, feats):

        B, T, C_in, H, W = lqs.shape

        # ----- build mapping index (same logic as original) -----
        num_outputs = len(feats['spatial'])
        mapping_idx = list(range(num_outputs))
        mapping_idx += mapping_idx[::-1]          # [0..n-1, n-1..0]
        mapping_idx = mapping_idx[:T]             # only need first T entries

        # ----- spatial branch: (B, num_outputs, C_s, H, W) -> (B, T, C_s, H, W) -----
        spatial = torch.stack(feats['spatial'], dim=1)        # (B, N, C_s, H, W)
        spatial = spatial[:, mapping_idx, ...]                # (B, T, C_s, H, W)

        # ----- non-spatial features: stack along time -----
        # assumes each feats[k] (k != 'spatial') is a list length >= T of (B, C_k, H, W)
        other_feats = [
            torch.stack(feats[k][:T], dim=1)                  # (B, T, C_k, H, W)
            for k in feats if k != 'spatial'
        ]

        # ----- concatenate all features along channel dim -----
        # result: (B, T, C_total, H, W)
        hr = torch.cat([spatial] + other_feats, dim=2)

        # ----- flatten (B, T) into batch to run CNN in parallel -----
        B_, T_, C_total, H_, W_ = hr.shape
        hr = hr.view(B_ * T_, C_total, H_, W_)                # (B*T, C_total, H, W)

        # ----- reconstruction & upsampling (same as original, but batched) -----
        hr = self.reconstruction(hr)
        hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
        hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)

        # ----- residual connection from lqs -----
        lqs_flat = lqs.view(B_ * T_, C_in, H, W)              # (B*T, C_in, H, W)
        hr = hr + self.img_upsample(lqs_flat)                 # (B*T, C_out, H_up, W_up)

        # ----- reshape back to (B, T, C_out, H_up, W_up) -----
        _, C_out, H_up, W_up = hr.shape
        hr = hr.view(B_, T_, C_out, H_up, W_up)

        return hr

class BasicVSRPlusPlus(nn.Module):

    def __init__(self, mid_channels=64, num_blocks=7,
                 max_residue_magnitude=10, scale=4, spynet_path=None):

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
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deformable_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)
        
        self.upsampler = Upsampler(mid_channels, scale=scale)

        # self.reset_parameters()

        if False:
            mode = 'default'
            self.spynet = torch.compile(self.spynet, mode=mode)
            self.feat_extract = torch.compile(self.feat_extract, mode=mode)
            self.upsampler = torch.compile(self.upsampler, mode=mode)
            for module in self.backbone:
                self.backbone[module] = torch.compile(self.backbone[module], mode=mode)

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
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

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
