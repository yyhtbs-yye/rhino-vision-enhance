import torch
import torch.nn as nn
import torch.nn.functional as F

from rhopsr.nn.model_zoo.psrt.psrt_arch import PSRT

from rhcore.utils.build_components import build_module

class ExvPSRT(PSRT):
    """PSRT-Recurrent network structure with External Image Super-Resolution Module."""

    def __init__(self, in_channels=3, mid_channels=64,
                 embed_dim=120, depths=(6, 6, 6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6, 6, 6), window_size=(3, 8, 8), 
                 num_frames=3, img_size = 64, patch_size=1,
                 cpu_cache_length=100, is_low_res_input=True, spynet_path=None,
                 spatial_ext_config=None, spatial_ext_path=None):

        super().__init__(in_channels=in_channels,
                         mid_channels=mid_channels,
                         embed_dim=embed_dim,
                         depths=depths,
                         num_heads=num_heads,
                         window_size=window_size,
                         num_frames=num_frames,
                         img_size=img_size,
                         patch_size=patch_size,
                         cpu_cache_length=cpu_cache_length,
                         is_low_res_input=is_low_res_input,
                         spynet_path=spynet_path)
        self.window_size = window_size
        
        self.spatial_ext_config = spatial_ext_config
        self.spatial_ext = build_module(spatial_ext_config)
        if spatial_ext_path is not None:
            loadnet = torch.load(spatial_ext_path)['params_ema']
            self.spatial_ext.load_state_dict(loadnet, strict=True)
        
        # freeze spatial feature extractor
        for param in self.spatial_ext.parameters():
            param.requires_grad = False
        self.spatial_ext.eval()

    def forward(self, lqs):

        if not self.training:
            return self.eval_forward(lqs)

        n, t, c, h, w = lqs.size()
        orig_h, orig_w = h, w

        spatial_window_size = self.spatial_ext_config['params'].get('window_size', 1)
        
        window_size = self.window_size
        win_h = window_size[1] if isinstance(window_size, (list, tuple)) else window_size
        win_w = window_size[2] if isinstance(window_size, (list, tuple)) else window_size

        win_h = max(win_h, spatial_window_size)
        win_w = max(win_w, spatial_window_size)

        pad_h = (win_h - h % win_h) % win_h
        pad_w = (win_w - w % win_w) % win_w
        if pad_h > 0 or pad_w > 0:
            lqs = lqs.view(-1, c, h, w)
            # Pad inputs so H/W are divisible by window size.
            lqs = F.pad(lqs, (0, pad_w, 0, pad_h), mode='reflect')
            h += pad_h
            w += pad_w
            lqs = lqs.view(n, t, c, h, w)

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                _, feat = self.spatial_ext(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            _, feats_ = self.spatial_ext(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

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
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        out = self.upsample(lqs, feats)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :, :orig_h * 4, :orig_w * 4].contiguous()

        return out

    def eval_forward(
        self,
        lqs: torch.Tensor,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        offload_flows_to_cpu: bool = True,
        empty_cache: bool = True,
    ) -> torch.Tensor:
        """
        Ultra-VRAM-lightweight eval forward:
          - inference_mode
          - force CPU feature caching
          - optional AMP + flow offload (CPU)
          - aggressive delete + empty_cache

        Expected input: (n, t, c, h, w)
        """
        self.eval()
        device = lqs.device

        # ---- shape/padding (same logic as forward) ----
        n, t, c, h, w = lqs.size()
        orig_h, orig_w = h, w

        spatial_window_size = self.spatial_ext_config["params"].get("window_size", 1)

        window_size = self.window_size
        win_h = window_size[1] if isinstance(window_size, (list, tuple)) else window_size
        win_w = window_size[2] if isinstance(window_size, (list, tuple)) else window_size

        win_h = max(win_h, spatial_window_size)
        win_w = max(win_w, spatial_window_size)

        pad_h = (win_h - h % win_h) % win_h
        pad_w = (win_w - w % win_w) % win_w

        if pad_h > 0 or pad_w > 0:
            lqs_ = lqs.view(-1, c, h, w)
            lqs_ = F.pad(lqs_, (0, pad_w, 0, pad_h), mode="reflect")
            h += pad_h
            w += pad_w
            lqs = lqs_.view(n, t, c, h, w)
            del lqs_
            if empty_cache and device.type == "cuda":
                torch.cuda.empty_cache()

        # In eval, always prefer CPU caching for lowest VRAM peak.
        self.cpu_cache = True

        # Avoid clone() in eval; keep a view/reference unless downsampling is needed.
        if self.is_low_res_input:
            lqs_downsample = lqs
        else:
            # downsample only if needed
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w),
                scale_factor=0.25,
                mode="bicubic",
                align_corners=False if "align_corners" in F.interpolate.__code__.co_varnames else None,  # safe-ish
            ).view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        with torch.inference_mode():
            # ---- spatial features: compute per-frame, offload to CPU immediately ----
            feats = {"spatial": []}

            for i in range(t):
                # Keep the slice ephemeral; ensure contiguous to avoid hidden copies later.
                frame = lqs[:, i, :, :, :].contiguous()

                _, feat = self.spatial_ext(frame)

                # Force CPU storage (this is the biggest VRAM saver).
                feat = feat.to("cpu", non_blocking=True)
                feats["spatial"].append(feat)

                del frame, feat
                if empty_cache and device.type == "cuda":
                    torch.cuda.empty_cache()

            # ---- optical flow ----
            # Your original asserts h/w >= 64 for low-res inputs; keep it.
            assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
                "The height and width of low-res inputs must be at least 64, "
                f"but got {lqs_downsample.size(3)} and {lqs_downsample.size(4)}."
            )

            flows_forward, flows_backward = self.compute_flow(lqs_downsample)

            # We will not need lqs_downsample anymore.
            if lqs_downsample is not lqs:
                del lqs_downsample
            if empty_cache and device.type == "cuda":
                torch.cuda.empty_cache()

            # Optionally offload flows to CPU to reduce VRAM peak.
            if offload_flows_to_cpu:
                if flows_forward is not None:
                    flows_forward = flows_forward.to("cpu", non_blocking=True)
                flows_backward = flows_backward.to("cpu", non_blocking=True)
                if empty_cache and device.type == "cuda":
                    torch.cuda.empty_cache()

            # ---- feature propagation (two iters, both directions) ----
            for iter_ in (1, 2):
                for direction in ("backward", "forward"):
                    module = f"{direction}_{iter_}"
                    feats[module] = []

                    if direction == "backward":
                        flows = flows_backward
                    else:
                        if flows_forward is not None:
                            flows = flows_forward
                        else:
                            # flip on CPU if needed (still cheaper than holding forward flows on GPU)
                            flows = flows_backward.flip(1)

                    # If flows are on CPU, bring them to GPU only for this propagation pass.
                    if offload_flows_to_cpu and flows.device.type == "cpu":
                        flows_gpu = flows.to(device, non_blocking=True)
                        feats = self.propagate(feats, flows_gpu, module)
                        del flows_gpu
                    else:
                        feats = self.propagate(feats, flows, module)

                    # aggressively free intermediates between passes
                    del flows
                    if empty_cache and device.type == "cuda":
                        torch.cuda.empty_cache()

            # ---- upsample ----
            # Note: feats['spatial'] is on CPU; this matches your cpu_cache pathway expectation.
            out = self.upsample(lqs, feats)

            # crop padding
            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :, : orig_h * 4, : orig_w * 4].contiguous()

            # last cleanup (doesn't affect out)
            del feats, flows_forward, flows_backward
            if empty_cache and device.type == "cuda":
                torch.cuda.empty_cache()

            out = out.to(device, non_blocking=True)

            return out

if __name__ == '__main__':
    #upscale = 4
    window_size = [3, 8, 8]
    img_size=64

    model = ExvPSRT(
        mid_channels = 64,
        embed_dim=120,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        window_size=window_size,
        num_frames = 3,
        img_size = img_size,
        patch_size = 1,
        cpu_cache_length = 100,
        is_low_res_input = True,
        spynet_path = 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    )

    print(model)
    print("flops",model.flops() / 1e9 + 'G')

    x = torch.randn((1, 5, 3, img_size, img_size))
    x = model(x)
    print(x.shape)
