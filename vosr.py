import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import random
import torch.distributions as D


def _beta_ratio(size, alpha: float, beta: float, device="cpu"):
    return D.Beta(alpha, beta).sample(size).to(device)


def stopgrad(x):
    return x.detach().clone()


def _zero_like(x):
    """Create zero tensor(s) matching x, which can be Tensor or list of Tensors."""
    if isinstance(x, (list, tuple)):
        return [torch.zeros_like(xi) for xi in x]
    return torch.zeros_like(x)


def _set_indices(x, indices, src):
    """x[indices] = src[indices] for Tensor or list of Tensors."""
    if isinstance(x, (list, tuple)):
        for i in range(len(x)):
            x[i][indices] = src[i][indices]
    else:
        x[indices] = src[indices]


class VOSR(nn.Module):
    def __init__(
        self,
        time_dist        = ['lognorm', -0.4, 1.0],
        cfg_ratio        = 0.10,
        cfg_scale        = 2.0,
        interp_type      = 'lin',
        u_weight         = 1,
        a = 1.0,
        b = 1.0,
        accelerator = None,
        t_start:                float = 0.0,
        t_end:                  float = 1.0,
        args=None,
    ):
        super().__init__()
        self.time_dist   = time_dist
        self.interp_type = interp_type
        self.cfg_ratio   = cfg_ratio
        self.cfg_scale   = cfg_scale
        self.args = args
        self.u_weight = u_weight
        self.device = accelerator.device
        self.a = a
        self.b = b
        self.t_start = t_start
        self.t_end = t_end



    def interpolate(self, cond, uncond, alpha, interp_type='linear'):
        if interp_type == 'sph':
            return alpha * cond + (1 - alpha**2)**0.5 * uncond
        elif interp_type == 'lin':
            return alpha * cond + (1 - alpha) * uncond

    def sample_t_r_v1(self, B, device):
        samples = torch.rand(B, 2, device=device)
        t_raw, r_raw = samples[:, 0], samples[:, 1]
        t = torch.maximum(t_raw, r_raw)
        g = _beta_ratio((B,), self.a, self.b, device)
        r = t.detach() * (1. - g)
        return t, r

    def _prepare_cfg_conditions(self, B, device, lq, z, cond_strength_aelq):
        """Prepare lq_weak/lq_noised/lq_mixed and z_weak/z_noised/z_mixed for CFG training.
        Fixed: use_aelq=True, use_venc=True, weak_cond_strength_venc=0, cond_strength_venc=1.
        """
        cfg_mask = torch.rand(B, device=device) < self.cfg_ratio
        cfg_indices = (cfg_mask > 0).nonzero(as_tuple=True)[0]

        weak_cond_strength_aelq = random.uniform(self.args.weak_cond_strength_aelq_list[0], self.args.weak_cond_strength_aelq_list[1])
        lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_cond_strength_aelq, self.interp_type)
        lq_noised = self.interpolate(lq, torch.randn_like(lq), cond_strength_aelq, interp_type='sph')
        lq_mixed = lq_noised.clone()
        lq_mixed[cfg_indices] = lq_weak[cfg_indices]

        # weak_cond_strength_venc=0 => z_weak is zeros
        z_weak = _zero_like(z)
        # cond_strength_venc=1, sph interp with alpha=1 => z_noised = z
        z_noised = z
        z_mixed = _zero_like(z)  # will be overwritten below
        if isinstance(z, (list, tuple)):
            z_mixed = [zi.clone() for zi in z]
        else:
            z_mixed = z.clone()
        _set_indices(z_mixed, cfg_indices, z_weak)

        return cfg_indices, lq_weak, lq_noised, lq_mixed, z_weak, z_noised, z_mixed

    def _prepare_cfg_conditions_distill(self, B, device, lq, z, cond_strength_aelq):
        """Same as _prepare_cfg_conditions but uses args.weak_cond_strength_aelq (scalar) for distill losses."""
        cfg_mask = torch.rand(B, device=device) < self.cfg_ratio
        cfg_indices = (cfg_mask > 0).nonzero(as_tuple=True)[0]

        lq_weak = self.interpolate(lq, torch.zeros_like(lq), self.args.weak_cond_strength_aelq, self.interp_type)
        lq_noised = self.interpolate(lq, torch.randn_like(lq), cond_strength_aelq, interp_type='sph')
        lq_mixed = lq_noised.clone()
        lq_mixed[cfg_indices] = lq_weak[cfg_indices]

        z_weak = _zero_like(z)
        z_noised = z
        if isinstance(z, (list, tuple)):
            z_mixed = [zi.clone() for zi in z]
        else:
            z_mixed = z.clone()
        _set_indices(z_mixed, cfg_indices, z_weak)

        return cfg_indices, lq_weak, lq_noised, lq_mixed, z_weak, z_noised, z_mixed

    def _teacher_target(self, model, model_tea, t, t_, lq_noised, lq_weak, z_t, z_noised, z_weak, z_mixed, v_target):
        """Compute teacher/self-distill target v with CFG."""
        with torch.no_grad():
            inp_cond = torch.cat([lq_noised, z_t], dim=1)
            inp_weak = torch.cat([lq_weak, z_t], dim=1)
            omega = torch.where((t_ >= self.t_start) & (t_ <= self.t_end), self.cfg_scale, 1.0)
            if model_tea is not None:
                v_cond = model_tea(inp_cond, t=t, z=z_noised)
                v_weak_pred = model_tea(inp_weak, t=t, z=z_weak)
                v = v_weak_pred + omega * (v_cond - v_weak_pred)
            else:
                v_cond = v_target
                v_weak_pred = model(inp_weak, t=t, r=t, z=z_weak)
                v = v_weak_pred + omega * (v_cond - v_weak_pred)
        return v

    # ──────────────────── Training losses ──────────────────── #

    def loss_fm(self, model, lq, hq, z=None, weight_dtype=None):
        B, device = hq.size(0), hq.device
        if self.time_dist[0] == 'uniform':
            t = torch.rand(B, device=device)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            rnd_normal = torch.randn(B, device=device)
            t = torch.sigmoid(rnd_normal * sigma + mu)

        t_ = rearrange(t, "b -> b 1 1 1")

        cond_strength_aelq = torch.sigmoid(torch.randn(B, device=device) * self.args.cond_strength_aelq_list[1] + self.args.cond_strength_aelq_list[0])
        cond_strength_aelq = rearrange(cond_strength_aelq, "b -> b 1 1 1")

        eps = torch.randn_like(lq)

        _, _, _, lq_mixed, _, _, z_mixed = self._prepare_cfg_conditions(B, device, lq, z, cond_strength_aelq)

        z_t = (1 - t_) * hq + t_ * eps
        inp = torch.cat([lq_mixed, z_t], dim=1)

        v = eps - hq
        v_current = model(inp, t, z=z_mixed)
        v_error = v_current - v
        loss = (v_error ** 2).mean()
        return loss, loss

    def loss_fm_distill_shortcut_improved(self, model, lq, hq, z=None, model_tea=None):
        B, device = hq.size(0), hq.device
        t, r = self.sample_t_r_v1(B, device)
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        cond_strength_aelq = torch.sigmoid(torch.randn(B, device=device) * self.args.cond_strength_aelq_list[1] + self.args.cond_strength_aelq_list[0])
        cond_strength_aelq = rearrange(cond_strength_aelq, "b -> b 1 1 1")

        eps = torch.randn_like(lq)
        _, lq_weak, lq_noised, lq_mixed, z_weak, z_noised, z_mixed = self._prepare_cfg_conditions_distill(B, device, lq, z, cond_strength_aelq)

        z_t = (1 - t_) * hq + t_ * eps
        v_target = eps - hq
        inp_mixed = torch.cat([lq_mixed, z_t], dim=1)

        v = self._teacher_target(model, model_tea, t, t_, lq_noised, lq_weak, z_t, z_noised, z_weak, z_mixed, v_target)

        v_current = model(inp_mixed, t=t, r=t, z=z_mixed)
        v_loss = ((v_current - v) ** 2).mean()

        with torch.no_grad():
            s = (t + r) / 2
            inp_i = torch.cat([lq_mixed, z_t], dim=1)
            v1 = model(inp_i, t=t, r=s, z=z_mixed)

            z_s = z_t - (t_-r_)/2*v1
            inp_i = torch.cat([lq_mixed, z_s], dim=1)
            v2 = model(inp_i, t=s, r=r, z=z_mixed)

            integral_approx = (v1 + v2) / 2

        u_current = model(inp_mixed, t, r, z_mixed)
        u_loss = ((u_current - stopgrad(integral_approx)) ** 2).mean()

        loss = v_loss + self.u_weight * u_loss
        return loss, loss

    def _rcgm_consistency(self, model, rng_state, inp_mixed, lq_mixed, z_t, z_mixed, t, r, t_, r_, v, DELTA_T, RCGM_N):
        """Compute RCGM consistency loss."""
        torch.cuda.set_rng_state(rng_state)
        u_current = model(inp_mixed, t=t, r=r, z=z_mixed)

        with torch.no_grad():
            delta = DELTA_T
            t_m_ = (t_ - delta).clamp(min=r_)
            actual_delta = t_ - t_m_
            z_t_m = z_t - v * actual_delta

            z_cur = z_t_m
            Ft_tar = torch.zeros_like(z_t)
            for step_i in range(RCGM_N):
                frac_start = step_i / RCGM_N
                frac_end = (step_i + 1) / RCGM_N
                t_step_ = t_m_ * (1 - frac_start) + r_ * frac_start
                t_next_ = t_m_ * (1 - frac_end) + r_ * frac_end
                dt = t_step_ - t_next_

                torch.cuda.set_rng_state(rng_state)
                inp_step = torch.cat([lq_mixed, z_cur], dim=1)
                v_step = model(inp_step, t=t_step_.reshape(-1), r=t_next_.reshape(-1), z=z_mixed)

                z_cur = z_cur - dt * v_step
                Ft_tar = Ft_tar + v_step * dt

            d = (t_ - r_).clamp(min=1e-6)
            near_mask = (t_ < (r_ + delta + 1e-6))
            cof_l = torch.where(near_mask, torch.ones_like(t_), d / delta)
            cof_r = torch.where(near_mask, 1.0 / d, torch.ones_like(t_) / delta)

            correction = (u_current.detach() * cof_l - Ft_tar * cof_r) - v
            correction = correction.clamp(min=-1.0, max=1.0)
            rcgm_target = u_current.detach() - correction

        u_loss = ((u_current - stopgrad(rcgm_target)) ** 2).mean()
        return u_loss

    def loss_fm_distill_rcgm_improved(self, model, lq, hq, z=None, model_tea=None):
        B, device = hq.size(0), hq.device
        DELTA_T = getattr(self.args, 'rcgm_delta_t', 0.01)
        RCGM_N = getattr(self.args, 'rcgm_n_steps', 2)
        t, r = self.sample_t_r_v1(B, device)
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        cond_strength_aelq = torch.sigmoid(torch.randn(B, device=device) * self.args.cond_strength_aelq_list[1] + self.args.cond_strength_aelq_list[0])
        cond_strength_aelq = rearrange(cond_strength_aelq, "b -> b 1 1 1")

        eps = torch.randn_like(lq)
        _, lq_weak, lq_noised, lq_mixed, z_weak, z_noised, z_mixed = self._prepare_cfg_conditions_distill(B, device, lq, z, cond_strength_aelq)

        z_t = (1 - t_) * hq + t_ * eps
        v_target = eps - hq
        inp_mixed = torch.cat([lq_mixed, z_t], dim=1)

        v = self._teacher_target(model, model_tea, t, t_, lq_noised, lq_weak, z_t, z_noised, z_weak, z_mixed, v_target)

        rng_state = torch.cuda.get_rng_state()
        v_current = model(inp_mixed, t=t, r=t, z=z_mixed)
        v_loss = ((v_current - v) ** 2).mean()

        u_loss = self._rcgm_consistency(model, rng_state, inp_mixed, lq_mixed, z_t, z_mixed, t, r, t_, r_, v, DELTA_T, RCGM_N)

        loss = v_loss + self.u_weight * u_loss
        return loss, loss

    

    # ──────────────────── Sampling ──────────────────── #

    @torch.no_grad()
    def sample_onestep(self, model, lq, venc_fea=None, n_steps: int = 8, schedule: str = "linear"):
        B, device = lq.size(0), lq.device
        z = torch.randn_like(lq)

        if schedule == "linear":
            t_seq = torch.linspace(1., 0., n_steps + 1, device=device)
        elif schedule == "cosine":
            t_seq = torch.cos(torch.linspace(0, np.pi / 2, n_steps + 1, device=device))
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        for i in range(n_steps):
            t_cur, t_next = t_seq[i], t_seq[i + 1]
            inp = torch.cat([lq, z], dim=1)
            u = model(inp, t_cur.repeat(B), t_next.repeat(B), venc_fea)
            z = z - (t_cur - t_next) * u

        return z

    

    @torch.no_grad()
    def sample_multistep_fm(self, model, lq, venc_fea=None, n_steps: int = 25, schedule: str = "linear"):
        B, device = lq.size(0), lq.device

        eps = torch.randn_like(lq)

        # CFG condition preparation (cfg_ratio > 0 always)
        weak_cond_strength_aelq = (self.args.weak_cond_strength_aelq_list[0] + self.args.weak_cond_strength_aelq_list[1]) / 2.
        lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_cond_strength_aelq, self.interp_type)

        # weak_cond_strength_venc=0 => venc_fea_weak is zeros; cond_strength_venc=1 => venc_fea unchanged
        if isinstance(venc_fea, (list, tuple)):
            venc_fea_weak = [torch.zeros_like(fea) for fea in venc_fea]
        else:
            venc_fea_weak = torch.zeros_like(venc_fea) if venc_fea is not None else None

        z = eps

        if schedule == "linear":
            t_seq = torch.linspace(1., 0., n_steps + 1, device=device)
        elif schedule == "cosine":
            t_seq = torch.cos(torch.linspace(0, np.pi / 2, n_steps + 1, device=device))
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        for i in range(n_steps):
            t_cur, t_next = t_seq[i], t_seq[i + 1]
            dt = t_cur - t_next

            model_inp = torch.cat([lq, z], dim=1)
            model_t = t_cur.repeat(B)
            model_venc_fea = venc_fea

            if t_cur <= self.t_end and t_cur >= self.t_start:
                inp_weak = torch.cat([lq_weak, z], dim=1)
                model_inp = torch.cat([model_inp, inp_weak], dim=0)
                model_t = model_t.repeat(2)

                if venc_fea is not None:
                    if isinstance(venc_fea, (list, tuple)):
                        model_venc_fea = [torch.cat([v, w], dim=0) for v, w in zip(venc_fea, venc_fea_weak)]
                    else:
                        model_venc_fea = torch.cat([venc_fea, venc_fea_weak], dim=0)

            d_cur = model(model_inp, model_t, z=model_venc_fea)

            if t_cur <= self.t_end and t_cur >= self.t_start:
                d_cur_cond, d_cur_weak = d_cur.chunk(2)
                d_cur = d_cur_weak + self.cfg_scale * (d_cur_cond - d_cur_weak)

            z = z - dt * d_cur

        return z
