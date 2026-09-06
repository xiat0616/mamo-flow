import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor
from torchdiffeq import odeint


TensorOrTensors = Tensor | tuple[Tensor, ...]


# ============================================================
# Sampling config
# ============================================================

@dataclass
class SampleConfig:
    cfg_mode: str = "none"
    cfg_scale: float = 1.0
    null_keys: set[str] | None = None


# ============================================================
# UNet configs
# ============================================================

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

    attn_resolutions: tuple[
        tuple[int, int],
        ...
    ]

    label_balance: float
    concat_balance: float


# ============================================================
# DiT config
# ============================================================

@dataclass
class DiTConfig:
    img_height: int
    img_width: int

    patch_size: int
    in_channels: int

    hidden_size: int
    depth: int
    num_heads: int

    mlp_ratio: float

    cond_embed_dim: int

    grad_checkpointing: bool = False


# ============================================================
# Generic rectified flow
#
# Works with:
#
#   UNet
#   DiT
#   any other backbone implementing:
#
#       forward_nn(x, t, cond_emb)
#
# ============================================================

class Flow(nn.Module):

    def __init__(
        self,
        forward_nn: nn.Module,
        cond_embedder: nn.Module | None = None,
        sigma: float = 0.0,
        alpha: float = 1.0,
        p_uncond: float = 0.2,
        amp_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.forward_nn = forward_nn
        self.cond_embedder = cond_embedder

        self.sigma = sigma
        self.alpha = alpha

        self.p_uncond = p_uncond

        self.amp_dtype = amp_dtype


    # ========================================================
    # Backbone call
    #
    # Both UNet and DiT should satisfy:
    #
    #   y = backbone(x, t, cond_emb)
    #
    # y must have the same shape as x.
    # ========================================================

    def vector_field(
        self,
        x: Tensor,
        t: Tensor,
        cond_emb: Tensor | None,
    ) -> Tensor:

        v = self.forward_nn(
            x,
            t,
            cond_emb,
        )

        if v.shape != x.shape:
            raise RuntimeError(
                "Flow backbone output shape mismatch.\n"
                f"input : {tuple(x.shape)}\n"
                f"output: {tuple(v.shape)}\n"
                f"backbone: {type(self.forward_nn).__name__}"
            )

        return v


    # ========================================================
    # Training
    # ========================================================

    def forward(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        g: torch.Generator | None = None,
    ) -> Tensor:

        # --------------------------------
        # Noise endpoint
        # --------------------------------

        u = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=g,
        )

        # --------------------------------
        # Random flow time
        # --------------------------------

        t = torch.rand(
            x.shape[0],
            device=x.device,
            generator=g,
        )

        t = self.schedule(
            t,
            self.alpha,
        )

        # --------------------------------
        # Interpolated state
        # --------------------------------

        x_t = self.interpolant(
            u,
            x,
            t,
            self.sigma,
        )

        # --------------------------------
        # Conditioning
        # --------------------------------

        cond_emb = None

        if (
            self.cond_embedder is not None
            and pa is not None
        ):

            cond_emb = self.cond_embedder(
                pa
            )

            # Classifier-free conditioning dropout
            if (
                self.training
                and self.p_uncond > 0
            ):

                bs = x.shape[0]

                keep_mask = (
                    torch.rand(
                        bs,
                        device=x.device,
                    )
                    > self.p_uncond
                ).to(
                    cond_emb.dtype
                )

                # Works for [B,D].
                #
                # If later cond_emb becomes
                # [B,N,D], change this to:
                #
                # keep_mask[:, None, None]
                #
                cond_emb = (
                    cond_emb
                    * keep_mask[:, None]
                )

        # --------------------------------
        # Predict velocity
        # --------------------------------

        v_t = self.vector_field(
            x_t,
            t,
            cond_emb,
        )

        # --------------------------------
        # Rectified-flow target:
        #
        # dx_t / dt = x - u
        # --------------------------------

        target = x - u

        return torch.mean(
            (
                target
                - v_t
            ) ** 2
        )


    # ========================================================
    # Flow interpolant
    # ========================================================

    def interpolant(
        self,
        u: Tensor,
        x: Tensor,
        t: Tensor,
        sigma: float,
    ) -> Tensor:

        t = t.reshape(
            -1,
            *(
                [1]
                * (u.dim() - 1)
            ),
        )

        x_t = (
            (1 - t) * u
            + t * x
        )

        if sigma > 0:

            x_t = (
                x_t
                + sigma
                * torch.randn_like(x)
            )

        return x_t


    # ========================================================
    # Time schedule
    # ========================================================

    def schedule(
        self,
        t: Tensor,
        alpha: float,
    ) -> Tensor:

        return (
            t
            / (
                alpha
                - (alpha - 1) * t
            )
        )


    # ========================================================
    # Conditioning helper
    # ========================================================

    def get_cond_emb(
        self,
        pa: dict[str, Tensor] | None = None,
        null_keys: set[str] | None = None,
    ) -> Tensor | None:

        if (
            self.cond_embedder is None
            or pa is None
        ):
            return None

        if null_keys is None:

            return self.cond_embedder(
                pa
            )

        return self.cond_embedder(
            pa,
            null_keys=null_keys,
        )


    # ========================================================
    # CFG / guided vector field
    # ========================================================

    def guided_vector_field(
        self,
        y: Tensor,
        t: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
    ) -> Tensor:

        if sample_args is None:
            sample_args = SampleConfig()

        # ODE solver normally gives scalar t.
        t_batch = (
            t.expand(y.shape[0])
            if t.ndim == 0
            else t
        )

        # ----------------------------------------------------
        # No CFG
        # ----------------------------------------------------

        if sample_args.cfg_mode == "none":

            cond_emb = self.get_cond_emb(
                pa,
                null_keys=(
                    sample_args.null_keys
                ),
            )

            return self.vector_field(
                y,
                t_batch,
                cond_emb,
            )

        # ----------------------------------------------------
        # CFG requires conditioning
        # ----------------------------------------------------

        if (
            self.cond_embedder is None
            or pa is None
        ):

            raise ValueError(
                f"cfg_mode="
                f"{sample_args.cfg_mode!r} "
                "requires both pa and "
                "cond_embedder."
            )

        # ----------------------------------------------------
        # Standard CFG
        # ----------------------------------------------------

        if sample_args.cfg_mode == "cfg":

            all_keys = set(
                self.cond_embedder.parents
            )

            cond_full = (
                self.get_cond_emb(
                    pa
                )
            )

            cond_null = (
                self.get_cond_emb(
                    pa,
                    null_keys=all_keys,
                )
            )

            v_cond = self.vector_field(
                y,
                t_batch,
                cond_full,
            )

            v_uncond = self.vector_field(
                y,
                t_batch,
                cond_null,
            )

            return (
                v_uncond
                + sample_args.cfg_scale
                * (
                    v_cond
                    - v_uncond
                )
            )

        # ----------------------------------------------------
        # Factorized CFG
        # ----------------------------------------------------

        if sample_args.cfg_mode == "fcfg":

            raise NotImplementedError(
                "fcfg sampling is not "
                "implemented yet."
            )

        raise ValueError(
            f"Unknown cfg_mode: "
            f"{sample_args.cfg_mode}"
        )


    # ========================================================
    # ODE solver
    #
    # Works identically for UNet and DiT.
    # ========================================================

    @torch.inference_mode()
    def ode_solve(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
        **kwargs,
    ) -> TensorOrTensors | tuple[
        Tensor,
        TensorOrTensors,
    ]:

        if sample_args is None:
            sample_args = SampleConfig()

        def func(
            t: Tensor,
            y: Tensor,
        ) -> Tensor:

            if self.amp_dtype is None:

                dydt = (
                    self.guided_vector_field(
                        y,
                        t,
                        pa=pa,
                        sample_args=sample_args,
                    )
                )

            else:

                with torch.autocast(
                    y.device.type,
                    dtype=self.amp_dtype,
                ):

                    dydt = (
                        self.guided_vector_field(
                            y,
                            t,
                            pa=pa,
                            sample_args=sample_args,
                        )
                    )

            # torchdiffeq integrates in float32.
            return dydt.float()

        return odeint(
            func,
            x,
            **kwargs,
        )