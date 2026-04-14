from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint

TensorOrTensors = Tensor | tuple[Tensor, ...]

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
    img_resolution: int
    img_channels: int
    cond_dim: int
    model_channels: int
    channel_mult: tuple[int, ...]
    channel_mult_time: int | None
    channel_mult_emb: int | None
    num_blocks: int
    attn_resolutions: tuple[int, ...]
    label_balance: float
    concat_balance: float

class Flow(nn.Module):
    def __init__(
        self,
        forward_nn: nn.Module,
        cond_embedder: nn.Module | None = None,
        sigma: float = 0.0,
        alpha: float = 1.0,
        p_uncond: float = 0.2,
    ):
        super().__init__()
        self.forward_nn = forward_nn
        self.cond_embedder = cond_embedder
        self.sigma = sigma
        self.alpha = alpha
        self.p_uncond = p_uncond

    def forward(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        g: torch.Generator | None = None,
    ) -> Tensor:
        u = torch.randn_like(x, generator=g)
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

    @torch.inference_mode()
    def ode_solve(
        self,
        x: Tensor,
        pa: dict[str, Tensor] | None = None,
        null_keys: set[str] | None = None,
        **kwargs,
    ) -> TensorOrTensors | tuple[Tensor, TensorOrTensors]:
        cond_emb = None
        if self.cond_embedder is not None and pa is not None:
            if null_keys is None:
                cond_emb = self.cond_embedder(pa)
            else:
                cond_emb = self.cond_embedder(pa, null_keys=null_keys)

        def func(t: Tensor, y: Tensor) -> Tensor:
            with torch.autocast(y.device.type, dtype=torch.bfloat16):
                dydt = self.forward_nn(y, t.expand(y.shape[0]), cond_emb)
            return dydt.float()

        return odeint(func, x, **kwargs)