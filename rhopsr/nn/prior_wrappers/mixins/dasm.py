from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DASMConfig:
    enable_after_step: int = 0
    lambda_vsd: float = 0.7
    lambda_tsm: float = 0.3

    total_steps: int = 4
    start_index_min: int = 50
    start_index_max: int = 950
    step_size: int = 50
    use_random_bias: bool = True

    trajectory_weights: tuple = (1.0, 0.3, 0.3, 0.3)
    teacher_role: str = "teacher"
    adapter_role: str = "lora"


class DistributionAwareSamplingModule:
    def __init__(self, cfg: DASMConfig = DASMConfig()):
        self.cfg = cfg

    def _alpha_bar(self, teacher, t, x):
        a = self._scheduler(teacher).alphas_cumprod.to(x.device)[t.long()]
        return a.view(-1, *([1] * (x.ndim - 1))).to(x.dtype)

    def _weight(self, teacher, t, x):
        return 1.0 - self._alpha_bar(teacher, t, x)

    def _step(self, teacher, x_t, eps, t, t_next):
        m = self._module(teacher)
        if hasattr(m, "dasm_step"):
            return m.dasm_step(
                noisy_latents=x_t,
                prediction=eps,
                current_t=t,
                next_t=t_next,
            )

        a_t = self._alpha_bar(teacher, t, x_t)
        a_next = self._alpha_bar(teacher, t_next, x_t)

        x0 = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        return a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps

    def _trajectory(self, teacher, batch_size, device, steps):
        sched = self._scheduler(teacher)
        ts = sched.timesteps.to(device)

        start = torch.randint(
            self.cfg.start_index_min,
            self.cfg.start_index_max + 1,
            (batch_size,),
            device=device,
        )

        bias = 0
        if self.cfg.use_random_bias:
            half = self.cfg.step_size // 2
            bias = torch.randint(-half, half + 1, (batch_size,), device=device)

        end = (start + self.cfg.step_size * steps + bias).clamp(0, len(ts) - 1)

        grid = torch.linspace(0, 1, steps + 1, device=device)[:, None]
        idx = torch.round(start[None].float() + (end - start)[None].float() * grid).long()
        traj = ts[idx]

        w = torch.tensor(self.cfg.trajectory_weights, device=device, dtype=torch.float32)
        if len(w) < steps:
            w = torch.cat([w, w[-1].repeat(steps - len(w))])
        w = w[:steps]

        return [traj[i] for i in range(steps)], [traj[i + 1] for i in range(steps - 1)], w

    def _surrogate(self, z, g):
        return 0.5 * F.mse_loss(z.float(), (z - g).detach().float(), reduction="mean")

    def student_surrogate_losses(
        self,
        *,
        teacher,
        prediction,
        target,
        batch=None,
        global_step=None,
        noises=None,
    ):
        
        # Encode target/gt and prediction/restored images to latents using the teacher prior
        with torch.no_grad():
            target_latents = teacher('encode', images=target)

        student_latents = teacher('encode', images=prediction)

        batch = {} if batch is None else batch
        steps = self.cfg.total_steps if (global_step is None or global_step >= self.cfg.enable_after_step) else 1
        noises = torch.randn_like(student_latents) if noises is None else noises

        with torch.no_grad():
            t_cur, t_next, step_w = self._trajectory(
                teacher, student_latents.shape[0], student_latents.device, steps
            )

            x_student = teacher("sample_noisy_latent",
                latents=student_latents,
                timesteps=t_cur[0],
                noises=noises,
            )
            x_target = teacher("sample_noisy_latent",
                latents=target_latents,
                timesteps=t_cur[0],
                noises=noises,
            )

            g_vsd = torch.zeros_like(student_latents)
            g_tsm = torch.zeros_like(student_latents)

            for i, t in enumerate(t_cur):
                eps_teacher_student = teacher("predict_score",
                                              noisy_latents=x_student,
                                              timesteps=t,
                                              batch=batch,
                                              role=self.cfg.teacher_role,
                )
                eps_teacher_target = teacher("predict_score",
                                             noisy_latents=x_target,
                                             timesteps=t, batch=batch,
                                             role=self.cfg.teacher_role,
                )
                eps_adapter_student = teacher("predict_score",
                                              noisy_latents=x_student,
                                              timesteps=t,
                                              batch=batch,
                                              role=self.cfg.adapter_role,
                )

                w = self._weight(teacher, t, student_latents) * step_w[i]
                g_vsd = g_vsd + (eps_teacher_student - eps_adapter_student) * w
                g_tsm = g_tsm + (eps_teacher_student - eps_teacher_target) * w

                if i < steps - 1:
                    x_student = self._step(teacher, x_student, eps_adapter_student, t, t_next[i])
                    x_target = self._step(teacher, x_target, eps_teacher_target, t, t_next[i])

        loss_vsd = self._surrogate(student_latents, g_vsd)
        loss_tsm = self._surrogate(student_latents, g_tsm)
        loss_tsd = self.cfg.lambda_vsd * loss_vsd + self.cfg.lambda_tsm * loss_tsm

        return {
            "vsd_surrogate_loss": loss_vsd,
            "tsm_surrogate_loss": loss_tsm,
            "tsd_surrogate_loss": loss_tsd,
            "grad_vsd": g_vsd,
            "grad_tsm": g_tsm,
            "grad_total": self.cfg.lambda_vsd * g_vsd + self.cfg.lambda_tsm * g_tsm,
        }