import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint
from dataclasses import dataclass

TensorOrTensors = Tensor | tuple[Tensor, ...]


@dataclass
class SampleConfig:
    cfg_mode: str = "none"
    cfg_scale: float = 1.0
    null_keys: set[str] | None = None

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

    def forward(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        g: torch.Generator | None = None,
    ) -> Tensor:
        
        u = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
        t = torch.rand(x.shape[0], device=x.device, generator=g)

        t = self.schedule(t, self.alpha)
        x_t = self.interpolant(u, x, t, self.sigma)

        cond_emb = None
        if self.cond_embedder is not None and pa is not None:
            cond_emb = self.cond_embedder(pa)

            if self.training and self.p_uncond > 0:
                bs, device = x.shape[0], x.device
                keep_mask = (torch.rand(bs, device=device) > self.p_uncond).to(cond_emb.dtype)
                cond_emb = cond_emb * keep_mask[:, None]

        v_t = self.forward_nn(x_t, t, cond_emb)
        return torch.mean((x - u - v_t) ** 2)

    def interpolant(self, u: Tensor, x: Tensor, t: Tensor, sigma: float) -> Tensor:
        t = t.reshape(-1, *([1] * (u.dim() - 1)))
        x_t = (1 - t) * u + t * x
        if sigma > 0:
            x_t = x_t + sigma * torch.randn_like(x)
        return x_t

    def schedule(self, t: Tensor, alpha: float) -> Tensor:
        return t / (alpha - (alpha - 1) * t)

    def get_cond_emb(
        self,
        pa: dict[str, Tensor] | None = None,
        null_keys: set[str] | None = None,
    ) -> Tensor | None:
        if self.cond_embedder is None or pa is None:
            return None
        if null_keys is None:
            return self.cond_embedder(pa)
        return self.cond_embedder(pa, null_keys=null_keys)

    def guided_vector_field(
        self,
        y: Tensor,
        t: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
    ) -> Tensor:
        if sample_args is None:
            sample_args = SampleConfig()

        t_batch = t.expand(y.shape[0]) if t.ndim == 0 else t

        if sample_args.cfg_mode == "none":
            cond_emb = self.get_cond_emb(pa, null_keys=sample_args.null_keys)
            return self.forward_nn(y, t_batch, cond_emb)

        if self.cond_embedder is None or pa is None:
            raise ValueError(f"cfg_mode='{sample_args.cfg_mode}' requires both pa and cond_embedder")

        if sample_args.cfg_mode == "cfg":
            all_keys = set(self.cond_embedder.parents)
            cond_full = self.get_cond_emb(pa)
            cond_null = self.get_cond_emb(pa, null_keys=all_keys)

            v_cond = self.forward_nn(y, t_batch, cond_full)
            v_uncond = self.forward_nn(y, t_batch, cond_null)
            return v_uncond + sample_args.cfg_scale * (v_cond - v_uncond)

        if sample_args.cfg_mode == "fcfg":
            raise NotImplementedError(
                "fcfg sampling is not implemented yet. "
                "Revisit group-wise factorized guidance after training."
            )

        raise ValueError(f"Unknown cfg_mode: {sample_args.cfg_mode}")

    @torch.inference_mode()
    def ode_solve(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        sample_args: SampleConfig | None = None,
        **kwargs,
    ) -> TensorOrTensors | tuple[Tensor, TensorOrTensors]:
        if sample_args is None:
            sample_args = SampleConfig()

        def func(t: Tensor, y: Tensor) -> Tensor:
            if self.amp_dtype is None:
                # print("Warning: amp_dtype is not set, running guided_vector_field without autocast.")
                dydt = self.guided_vector_field(
                    y,
                    t,
                    pa=pa,
                    sample_args=sample_args,
                )
            else:
                with torch.autocast(y.device.type, dtype=self.amp_dtype):
                    # print("Running guided_vector_field with autocast.")
                    dydt = self.guided_vector_field(
                        y,
                        t,
                        pa=pa,
                        sample_args=sample_args,
                    )
            return dydt.float()

        return odeint(func, x, **kwargs)