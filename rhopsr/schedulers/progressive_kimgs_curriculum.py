class ProgressiveKimgsCurriculum:
    """
    Progressive curriculum over blur sigma values, controlled by kimgs (thousands of images).
    - sigmas: list length S (e.g., [8, 4, 2, 1, 0])
    - stage_kimgs: list length S (kimgs spent in each sigma stage)
    - fade_fraction: fraction of each *new* stage spent fading from previous sigma to current sigma
                     (0.0 disables fade; 0.5 means first half of each stage is a linear fade)
    """
    def __init__(self, sigmas, stage_kimgs, fade_fraction=0.5):
        assert len(sigmas) == len(stage_kimgs), "sigmas and stage_kimgs must have same length"
        assert 0.0 <= fade_fraction <= 1.0
        self.sigmas = sigmas
        self.stage_kimgs = stage_kimgs
        self.fade_fraction = fade_fraction

        # Precompute cumulative kimgs per stage
        self.cum_kimgs = []
        total = 0.0
        for L in stage_kimgs:
            total += L
            self.cum_kimgs.append(total)

        self.total_kimgs = total
        self.kimgs = 0.0        # running total
        self.stage_idx = 0      # current stage index

    def add_images(self, n_images, world_size=1):
        """
        Increment kimgs by (#images processed * world_size) / 1000.0
        world_size tries to reflect global progress w/ DDP.
        """
        self.kimgs += (n_images * world_size) / 1000.0
        self.kimgs = min(self.kimgs, self.total_kimgs)

        # Update stage index
        k = self.kimgs
        idx = 0
        while idx < len(self.cum_kimgs) and k > self.cum_kimgs[idx] + 1e-9:
            idx += 1
        self.stage_idx = min(idx, len(self.sigmas) - 1)

    def _stage_bounds(self, i):
        """Return (start_k, end_k) in kimgs for stage i."""
        start_k = 0.0 if i == 0 else self.cum_kimgs[i - 1]
        end_k = self.cum_kimgs[i]
        return start_k, end_k

    def current_blend(self):
        """
        Returns (prev_sigma, curr_sigma, alpha)
        - During the first fade_fraction of stage i>0, alpha goes from 0->1 linearly.
        - Otherwise alpha=1 (fully at curr_sigma).
        For stage 0, prev_sigma=curr_sigma, alpha=1.
        """
        i = self.stage_idx
        curr_sigma = self.sigmas[i]
        if i == 0:
            return curr_sigma, curr_sigma, 1.0, self.stage_idx

        prev_sigma = self.sigmas[i - 1]
        start_k, end_k = self._stage_bounds(i)
        stage_len = max(end_k - start_k, 1e-9)
        fade_len = self.fade_fraction * stage_len

        if fade_len <= 1e-9:
            return prev_sigma, curr_sigma, 1.0, self.stage_idx  # no fading

        # Position inside stage
        k_in = max(0.0, min(self.kimgs - start_k, stage_len))
        if k_in <= fade_len:
            alpha = k_in / fade_len  # 0->1 across fade
        else:
            alpha = 1.0
        return prev_sigma, curr_sigma, float(alpha), self.stage_idx

    def state_dict(self):
        return {
            "kimgs": self.kimgs,
            "stage_idx": self.stage_idx,
        }

    def load_state_dict(self, d):
        self.kimgs = float(d.get("kimgs", 0.0))
        self.stage_idx = int(d.get("stage_idx", 0))
