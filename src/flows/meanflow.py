import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jvp
from torchdiffeq import odeint
from dataclasses import dataclass


@dataclass
class SampleConfig:
    cfg_mode: str = "none"
    cfg_scale: float = 1.0
    null_keys: set[str] | None = None


@dataclass
class MeanFlowConfig:
    """Training hyperparameters for MeanFlow / minimal Improved Mean Flow."""

    ratio_r_neq_t: float = 0.25
    time_sampler: str = "lognorm"
    lognorm_mu: float = -0.4
    lognorm_sigma: float = 1.0
    adaptive_weight_p: float = 1.0
    adaptive_weight_eps: float = 1e-3


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


class MeanFlow(nn.Module):
    """MeanFlow: one-step generative modelling via average velocity fields.

    Implements the MeanFlow Identity for training and single-NFE sampling.

    Training uses standard label dropout for classifier-free guidance.
    CFG is applied at inference time via batch doubling.
    The network predicts the average velocity u(z, r, t, cond_emb).
    """

    def __init__(
        self,
        forward_nn: nn.Module,
        cond_embedder: nn.Module | None = None,
        p_uncond: float = 0.2,
        amp_dtype: torch.dtype | None = None,
        mf_config: MeanFlowConfig | None = None,
    ):
        super().__init__()
        self.forward_nn = forward_nn
        self.cond_embedder = cond_embedder
        self.p_uncond = p_uncond
        self.amp_dtype = amp_dtype
        self.mf_config = mf_config or MeanFlowConfig()

    # ------------------------------------------------------------------
    # Time sampling
    # ------------------------------------------------------------------
    def _sample_times(
        self,
        batch_size: int,
        device: torch.device,
        g: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample (r, t) pairs with t >= r."""
        if self.mf_config.time_sampler == "uniform":
            raw = torch.rand(batch_size, 2, device=device, generator=g)
        elif self.mf_config.time_sampler == "lognorm":
            normal = torch.randn(batch_size, 2, device=device, generator=g)
            raw = torch.sigmoid(
                normal * self.mf_config.lognorm_sigma + self.mf_config.lognorm_mu
            )
        else:
            raise ValueError(f"Unknown time_sampler: {self.mf_config.time_sampler}")

        sorted_raw, _ = torch.sort(raw, dim=1)
        r = sorted_raw[:, 0]
        t = sorted_raw[:, 1]

        if self.mf_config.ratio_r_neq_t < 1.0:
            fraction_equal = 1.0 - self.mf_config.ratio_r_neq_t
            eq_mask = torch.rand(batch_size, device=device, generator=g) < fraction_equal
            r = torch.where(eq_mask, t, r)

        return r, t

    # ------------------------------------------------------------------
    # Network call helper
    # ------------------------------------------------------------------
    def _net_call(
        self,
        z: Tensor,
        r: Tensor,
        t: Tensor,
        cond_emb: Tensor | None,
    ) -> Tensor:
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

    def _get_train_cond_emb(
        self,
        pa: dict[str, Tensor] | None,
        batch_size: int,
        device: torch.device,
        g: torch.Generator | None = None,
    ) -> Tensor | None:
        """Build the training-time conditioning embedding with label dropout."""
        if self.cond_embedder is None or pa is None:
            return None

        cond_emb = self.cond_embedder(pa)
        if self.training and self.p_uncond > 0:
            keep = (
                torch.rand(batch_size, device=device, generator=g) > self.p_uncond
            ).to(cond_emb.dtype)
            cond_emb = cond_emb * keep[:, None]
        return cond_emb

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        g: torch.Generator | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute the MeanFlow training loss.

        Returns:
            During training:
                (loss, raw_mse)
            During evaluation:
                loss
        """
        batch_size, device = x.shape[0], x.device

        eps = torch.randn(x.shape, device=device, dtype=x.dtype, generator=g)
        r, t = self._sample_times(batch_size, device, g)

        v_cond = eps - x

        t_bc = t.reshape(-1, *([1] * (x.dim() - 1)))
        z_t = (1.0 - t_bc) * x + t_bc * eps

        cond_emb = self._get_train_cond_emb(pa, batch_size, device, g)

        if self.training and hasattr(self.forward_nn, "normalize_weights"):
            self.forward_nn.normalize_weights()

        u = self._net_call(z_t, r, t, cond_emb)

        def _fn(z_: Tensor, r_: Tensor, t_: Tensor) -> Tensor:
            return self._net_call(z_, r_, t_, cond_emb)

        tangents = (
            v_cond,
            torch.zeros_like(r),
            torch.ones_like(t),
        )

        with torch.no_grad():
            _, dudt = jvp(_fn, (z_t, r, t), tangents)
            dt = (t - r).reshape(-1, *([1] * (u.dim() - 1)))
            u_tgt = v_cond - dt * dudt

        error = u - u_tgt
        sq_err = (error ** 2).flatten(1).mean(1)

        p = self.mf_config.adaptive_weight_p
        if p > 0:
            weight = 1.0 / (sq_err.detach() + self.mf_config.adaptive_weight_eps) ** p
        else:
            weight = torch.ones_like(sq_err)

        loss = (weight * sq_err).mean()
        raw_mse = sq_err.mean().detach()

        if self.training:
            return loss, raw_mse
        return loss

    # ------------------------------------------------------------------
    # Guided vector field
    # ------------------------------------------------------------------
    def _guided_forward(
        self,
        noise: Tensor,
        r: Tensor,
        t: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
    ) -> Tensor:
        """Run the network with optional standard CFG."""
        if sample_args is None:
            sample_args = SampleConfig()

        if sample_args.cfg_mode == "none" or sample_args.cfg_scale == 1.0:
            cond_emb = self._get_cond_emb(pa)
            return self._net_call(noise, r, t, cond_emb)

        if sample_args.cfg_mode == "cfg":
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
                return u_uncond + sample_args.cfg_scale * (u_cond - u_uncond)

            cond_emb = self._get_cond_emb(pa)
            return self._net_call(noise, r, t, cond_emb)

        raise ValueError(f"Unknown cfg_mode: {sample_args.cfg_mode}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample(
        self,
        noise: Tensor,
        steps: int = 1,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
    ) -> Tensor:
        """Generate samples using z_r = z_t - (t-r) * u(z_t, r, t)."""
        batch_size, device = noise.shape[0], noise.device
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        z = noise
        for i in range(steps):
            t_val = ts[i]
            r_val = ts[i + 1]
            t_batch = torch.full((batch_size,), t_val, device=device)
            r_batch = torch.full((batch_size,), r_val, device=device)
            u = self._guided_forward(z, r_batch, t_batch, pa, sample_args)
            z = z - (t_val - r_val) * u

        return z

    # ------------------------------------------------------------------
    # ODE solve
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def ode_solve(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
        **kwargs,
    ) -> Tensor:
        """Solve dz/dt = v(z, t) using the boundary velocity u(z, t, t)."""
        if sample_args is None:
            sample_args = SampleConfig()

        cond_emb = self._get_cond_emb(pa, null_keys=sample_args.null_keys)

        def func(t: Tensor, y: Tensor) -> Tensor:
            batch_size = y.shape[0]
            t_batch = t.expand(batch_size) if t.ndim == 0 else t
            return self._net_call(y, t_batch, t_batch, cond_emb)

        return odeint(func, x, **kwargs)


class ImprovedMeanFlow(MeanFlow):
    """Improved Mean Flow with boundary-condition parameterization.

    The network predicts the average velocity field u(z, r, t, cond_emb).
    For a noisy sample z_t, the boundary value u(z_t, t, t) is interpreted
    as the predicted instantaneous velocity. A JVP is then used to compute
    the total derivative of u(z_t, r, t) along that predicted direction.

    Training forms the compound predictor

        V = u(z_t, r, t) + (t - r) * stopgrad(dudt),

    and regresses it toward the conditional velocity eps - x.

    This implementation includes the core iMF training objective with the
    boundary-condition formulation. Optional extensions such as an auxiliary
    v-head and flexible-CFG training are not included here.
    """

    def forward(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        g: torch.Generator | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute the iMF training objective.

        Returns:
            During training:
                (loss, raw_mse)
            During evaluation:
                loss
        """
        batch_size, device = x.shape[0], x.device

        eps = torch.randn(x.shape, device=device, dtype=x.dtype, generator=g)
        r, t = self._sample_times(batch_size, device, g)

        v_target = eps - x

        t_bc = t.reshape(-1, *([1] * (x.dim() - 1)))
        z_t = (1.0 - t_bc) * x + t_bc * eps

        cond_emb = self._get_train_cond_emb(pa, batch_size, device, g)

        if self.training and hasattr(self.forward_nn, "normalize_weights"):
            self.forward_nn.normalize_weights()

        def _fn(z_: Tensor, r_: Tensor, t_: Tensor) -> Tensor:
            return self._net_call(z_, r_, t_, cond_emb)

        u = _fn(z_t, r, t)

        with torch.no_grad():
            v_hat = _fn(z_t, t, t)
            tangents = (
                v_hat,
                torch.zeros_like(r),
                torch.ones_like(t),
            )
            _, dudt = jvp(_fn, (z_t, r, t), tangents)

        dt = (t - r).reshape(-1, *([1] * (u.dim() - 1)))
        V = u + dt * dudt.detach()

        error = V - v_target
        sq_err = (error ** 2).flatten(1).mean(1)
        loss = sq_err.mean()
        raw_mse = sq_err.mean().detach()

        if self.training:
            return loss, raw_mse
        return loss