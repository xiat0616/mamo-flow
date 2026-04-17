import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jvp
from torchdiffeq import odeint
from dataclasses import dataclass

TensorOrTensors = Tensor | tuple[Tensor, ...]


@dataclass
class SampleConfig:
    cfg_mode: str = "none"
    cfg_scale: float = 1.0
    null_keys: set[str] | None = None


@dataclass
class MeanFlowConfig:
    """Training hyperparameters for MeanFlow (Table 4 / Sec 4.3)."""

    ratio_r_neq_t: float = 0.25          # fraction of samples where r != t (Tab 1a)
    time_sampler: str = "lognorm"         # "uniform" | "lognorm" (Tab 1d)
    lognorm_mu: float = -0.4             # logit-normal mean
    lognorm_sigma: float = 1.0           # logit-normal std
    adaptive_weight_p: float = 1.0       # power p for adaptive loss weight (Tab 1e)
    adaptive_weight_eps: float = 1e-3    # small constant c in denominator (Eq 22)
    cfg_omega: float = 1.0               # guidance weight omega (Eq 19/21), 1.0 = no CFG
    cfg_kappa: float = 0.0               # mixing scale kappa (Eq 21 / Tab 5)
    cfg_min_t: float = 0.0               # minimum t for CFG activation
    cfg_max_t: float = 1.0               # maximum t for CFG activation


@dataclass
class BlockConfig:
    resample_filter: tuple[int, int]
    channels_per_head: int
    dropout: float
    res_balance: float
    attn_balance: float
    clip_act: int | None


@dataclass
class UNetConfig:
    img_height: int
    img_width: int
    img_channels: int
    cond_embed_dim: int
    model_channels: int
    channel_mult: tuple[int, ...]
    channel_mult_time: int | None
    channel_mult_emb: int | None
    num_blocks: int
    attn_resolutions: tuple[tuple[int, int], ...]
    label_balance: float
    concat_balance: float


amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class MeanFlow(nn.Module):
    """MeanFlow: one-step generative modelling via average velocity fields.

    Implements the MeanFlow Identity (Eq 6) for training and single-NFE
    sampling (Alg 1 & 2) from https://arxiv.org/abs/2505.13447.

    The neural network ``forward_nn`` predicts the *average* velocity
    u(z, r, t) instead of the instantaneous velocity v(z, t).
    It must accept: forward_nn(z_t, r, t, cond_emb).
    """

    def __init__(
        self,
        forward_nn: nn.Module,
        cond_embedder: nn.Module | None = None,
        p_uncond: float = 0.1,
        amp_dtype: torch.dtype | None = None,
        mf_config: MeanFlowConfig | None = None,
    ):
        super().__init__()
        self.forward_nn = forward_nn
        self.cond_embedder = cond_embedder
        self.p_uncond = p_uncond
        self.amp_dtype = amp_dtype
        self.cfg = mf_config or MeanFlowConfig()

    # ------------------------------------------------------------------
    # Time sampling  (Sec 4.3, Tab 1d)
    # ------------------------------------------------------------------
    def _sample_times(
        self, batch_size: int, device: torch.device, g: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample (r, t) pairs according to the configured distribution."""
        if self.cfg.time_sampler == "uniform":
            raw = torch.rand(batch_size, 2, device=device, generator=g)
        elif self.cfg.time_sampler == "lognorm":
            normal = torch.randn(batch_size, 2, device=device, generator=g)
            raw = torch.sigmoid(normal * self.cfg.lognorm_sigma + self.cfg.lognorm_mu)
        else:
            raise ValueError(f"Unknown time_sampler: {self.cfg.time_sampler}")

        # enforce t >= r by sorting
        sorted_raw, _ = torch.sort(raw, dim=1)
        r = sorted_raw[:, 0]
        t = sorted_raw[:, 1]

        # With probability (1 - ratio_r_neq_t), collapse r = t (pure FM signal)
        if self.cfg.ratio_r_neq_t < 1.0:
            fraction_equal = 1.0 - self.cfg.ratio_r_neq_t
            eq_mask = torch.rand(batch_size, device=device, generator=g) < fraction_equal
            r = torch.where(eq_mask, t, r)

        return r, t

    # ------------------------------------------------------------------
    # Network call helper (handles autocast)
    # ------------------------------------------------------------------
    def _net_call(self, z: Tensor, r: Tensor, t: Tensor, cond_emb) -> Tensor:
        if self.amp_dtype is not None:
            with torch.autocast(z.device.type, dtype=self.amp_dtype):
                return self.forward_nn(z, r, t, cond_emb).float()
        return self.forward_nn(z, r, t, cond_emb)

    # ------------------------------------------------------------------
    # Conditional embedding helpers
    # ------------------------------------------------------------------
    def _get_cond_emb(
        self,
        pa: dict[str, Tensor] | None,
        null_keys: set[str] | None = None,
    ) -> Tensor | None:
        if self.cond_embedder is None or pa is None:
            return None
        if null_keys is None:
            return self.cond_embedder(pa)
        return self.cond_embedder(pa, null_keys=null_keys)

    # ------------------------------------------------------------------
    # Forward (training)  –  Algorithm 1
    #
    # KEY INSIGHT from the reference implementation:
    # When CFG is active, v_tilde must be computed BEFORE the JVP,
    # because the JVP tangent for ∂_z u must be v_tilde (Eq 18):
    #   du/dt = v_tilde · ∂_z u + ∂_t u
    # The batch is split into CFG-eligible and non-CFG samples because
    # unconditional samples (from label dropout) should NOT get CFG.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        g: torch.Generator | None = None,
    ) -> Tensor:
        """Compute the MeanFlow training loss (Eq 9 / Eq 17).

        Args:
            x:  clean data batch, shape (B, *).
            pa: conditioning dict passed to ``cond_embedder``.
            g:  optional torch Generator for reproducibility.

        Returns:
            Scalar loss.
        """
        B, device = x.shape[0], x.device

        # --- sample noise & time -------------------------------------------------
        eps = torch.randn(x.shape, device=device, dtype=x.dtype, generator=g)
        r, t = self._sample_times(B, device, g)

        # conditional velocity  v_t = eps - x  (standard linear interpolant)
        v_cond = eps - x  # (B, *)

        # interpolant  z_t = (1-t)*x + t*eps
        t_bc = t.reshape(-1, *([1] * (x.dim() - 1)))
        z_t = (1.0 - t_bc) * x + t_bc * eps

        # --- conditioning --------------------------------------------------------
        cond_emb = None
        uncond_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if self.cond_embedder is not None and pa is not None:
            cond_emb = self.cond_embedder(pa)
            if self.training and self.p_uncond > 0:
                uncond_mask = torch.rand(B, device=device) < self.p_uncond
                keep = (~uncond_mask).to(cond_emb.dtype)
                cond_emb = cond_emb * keep[:, None]

        # --- Determine which samples get CFG treatment ---------------------------
        omega = self.cfg.cfg_omega
        kappa = self.cfg.cfg_kappa
        use_cfg = not (omega == 1.0 and kappa == 0.0)

        # CFG only applies within the time range AND to conditional samples
        cfg_time_mask = (t >= self.cfg.cfg_min_t) & (t <= self.cfg.cfg_max_t)
        cfg_eligible = use_cfg & cfg_time_mask & (~uncond_mask) & (cond_emb is not None)

        # --- Build u and u_target per-sample -------------------------------------
        u_output = torch.zeros_like(v_cond)
        u_target = torch.zeros_like(v_cond)
        dt = (t - r).reshape(-1, *([1] * (v_cond.dim() - 1)))

        cfg_indices = torch.where(cfg_eligible)[0] if use_cfg else torch.tensor([], dtype=torch.long, device=device)
        no_cfg_indices = torch.where(~cfg_eligible)[0] if use_cfg else torch.arange(B, device=device)

        # --- CFG samples: compute v_tilde first, then JVP with v_tilde -----------
        if len(cfg_indices) > 0:
            c_z = z_t[cfg_indices]
            c_v = v_cond[cfg_indices]
            c_r = r[cfg_indices]
            c_t = t[cfg_indices]
            c_dt = dt[cfg_indices]
            c_emb = cond_emb[cfg_indices]
            c_null_emb = torch.zeros_like(c_emb)

            # v_tilde (Eq 21) needs u(z_t, t, t) for cond & uncond
            # Batch the two forward passes together for efficiency
            with torch.no_grad():
                z2 = torch.cat([c_z, c_z], dim=0)
                t2 = torch.cat([c_t, c_t], dim=0)
                emb2 = torch.cat([c_emb, c_null_emb], dim=0)
                u_at_t = self._net_call(z2, t2, t2, emb2)
                u_cond_at_t, u_uncond_at_t = u_at_t.chunk(2, dim=0)

            c_v_tilde = (omega * c_v
                         + kappa * u_cond_at_t
                         + (1.0 - omega - kappa) * u_uncond_at_t)

            # JVP: tangent is (v_tilde, 0, 1) — v_tilde, NOT v_cond!
            def _fn_cfg(z_: Tensor, r_: Tensor, t_: Tensor) -> Tensor:
                return self._net_call(z_, r_, t_, c_emb)

            c_u, c_dudt = jvp(
                _fn_cfg,
                (c_z, c_r, c_t),
                (c_v_tilde, torch.zeros_like(c_r), torch.ones_like(c_t)),
            )
            u_output[cfg_indices] = c_u
            u_target[cfg_indices] = c_v_tilde - c_dt * c_dudt

        # --- Non-CFG samples: standard JVP with v_cond --------------------------
        if len(no_cfg_indices) > 0:
            nc_z = z_t[no_cfg_indices]
            nc_v = v_cond[no_cfg_indices]
            nc_r = r[no_cfg_indices]
            nc_t = t[no_cfg_indices]
            nc_dt = dt[no_cfg_indices]
            nc_emb = cond_emb[no_cfg_indices] if cond_emb is not None else None

            def _fn_no_cfg(z_: Tensor, r_: Tensor, t_: Tensor) -> Tensor:
                return self._net_call(z_, r_, t_, nc_emb)

            nc_u, nc_dudt = jvp(
                _fn_no_cfg,
                (nc_z, nc_r, nc_t),
                (nc_v, torch.zeros_like(nc_r), torch.ones_like(nc_t)),
            )
            u_output[no_cfg_indices] = nc_u
            u_target[no_cfg_indices] = nc_v - nc_dt * nc_dudt

        # --- loss  (Eq 9 + adaptive weighting, Appendix B.2) --------------------
        error = u_output - u_target.detach()
        sq_err = (error ** 2).flatten(1).sum(1)

        p = self.cfg.adaptive_weight_p
        if p > 0:
            weight = 1.0 / (sq_err.detach() + self.cfg.adaptive_weight_eps) ** p
        else:
            weight = torch.ones_like(sq_err)

        loss = (weight * sq_err).mean()
        return loss

    # ------------------------------------------------------------------
    # Sampling  –  Algorithm 2
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample_1nfe(
        self,
        noise: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
    ) -> Tensor:
        """One-step generation: z_0 = eps - u(eps, r=0, t=1).  (Alg 2)

        For models trained WITH baked-in CFG (omega != 1), set cfg_scale=1.0
        (no additional guidance needed). For models trained WITHOUT CFG,
        set cfg_mode="cfg" and cfg_scale > 1.0 for inference-time guidance.
        """
        if sample_args is None:
            sample_args = SampleConfig()

        B, device = noise.shape[0], noise.device
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)

        if sample_args.cfg_mode == "cfg" and sample_args.cfg_scale != 1.0:
            cond_emb = self._get_cond_emb(pa)
            null_emb = self._get_cond_emb(pa, null_keys=sample_args.null_keys)
            if null_emb is None:
                null_emb = torch.zeros_like(cond_emb) if cond_emb is not None else None

            if cond_emb is not None and null_emb is not None:
                z_cat = torch.cat([noise, noise], dim=0)
                r_cat = torch.cat([r, r], dim=0)
                t_cat = torch.cat([t, t], dim=0)
                emb_cat = torch.cat([cond_emb, null_emb], dim=0)
                u_cat = self._net_call(z_cat, r_cat, t_cat, emb_cat)
                u_cond, u_uncond = u_cat.chunk(2, dim=0)
                u = u_uncond + sample_args.cfg_scale * (u_cond - u_uncond)
            else:
                cond_emb = self._get_cond_emb(pa)
                u = self._net_call(noise, r, t, cond_emb)
        else:
            cond_emb = self._get_cond_emb(pa, null_keys=sample_args.null_keys)
            u = self._net_call(noise, r, t, cond_emb)

        return noise - u

    # ------------------------------------------------------------------
    # Few-step sampling  (Eq 12)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample_nsteps(
        self,
        noise: Tensor,
        steps: int = 2,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
    ) -> Tensor:
        """Few-step generation using Eq 12: z_r = z_t - (t-r)*u(z_t, r, t)."""
        if sample_args is None:
            sample_args = SampleConfig()

        B, device = noise.shape[0], noise.device
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        do_cfg = sample_args.cfg_mode == "cfg" and sample_args.cfg_scale != 1.0
        cond_emb = self._get_cond_emb(pa)
        null_emb = None
        if do_cfg:
            null_emb = self._get_cond_emb(pa, null_keys=sample_args.null_keys)
            if null_emb is None:
                null_emb = torch.zeros_like(cond_emb) if cond_emb is not None else None

        z = noise
        for i in range(steps):
            t_val = ts[i]
            r_val = ts[i + 1]
            t_batch = torch.full((B,), t_val, device=device)
            r_batch = torch.full((B,), r_val, device=device)

            if do_cfg and cond_emb is not None and null_emb is not None:
                z_cat = torch.cat([z, z], dim=0)
                r_cat = torch.cat([r_batch, r_batch], dim=0)
                t_cat = torch.cat([t_batch, t_batch], dim=0)
                emb_cat = torch.cat([cond_emb, null_emb], dim=0)
                u_cat = self._net_call(z_cat, r_cat, t_cat, emb_cat)
                u_cond, u_uncond = u_cat.chunk(2, dim=0)
                u = u_uncond + sample_args.cfg_scale * (u_cond - u_uncond)
            else:
                u = self._net_call(z, r_batch, t_batch, cond_emb)

            z = z - (t_val - r_val) * u

        return z

    # ------------------------------------------------------------------
    # ODE solve (for evaluation / comparison with multi-step baselines)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def ode_solve(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
        **kwargs,
    ) -> TensorOrTensors | tuple[Tensor, TensorOrTensors]:
        """Solve the instantaneous ODE dz/dt = v(z,t) using the average
        velocity network.  When r = t the network predicts v (instantaneous),
        so we call u(z, t, t) ≈ v(z, t)."""
        if sample_args is None:
            sample_args = SampleConfig()

        cond_emb = self._get_cond_emb(pa, null_keys=sample_args.null_keys)

        def func(t: Tensor, y: Tensor) -> Tensor:
            B = y.shape[0]
            t_batch = t.expand(B) if t.ndim == 0 else t
            r_batch = t_batch
            return self._net_call(y, r_batch, t_batch, cond_emb)

        return odeint(func, x, **kwargs)
