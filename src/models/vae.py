import argparse
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from unet import MPConv, mp_silu, mp_sum, normalize, resample


@dataclass
class VAEConfig:
    img_resolution: int
    img_channels: int
    latent_channels: int
    model_channels: int
    channel_mult: tuple[int, ...]
    num_blocks: int
    attn_resolutions: tuple[int, ...]
    beta: float
    var: float
    resample_filter: tuple[int, int]
    channels_per_head: int
    dropout: float
    res_balance: float
    attn_balance: float
    clip_act: int | None


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        flavor: str = "enc",
        resample_mode: str = "keep",
        resample_filter: tuple[int, int] = (1, 1),
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        res_balance: float = 0.3,
        attn_balance: float = 0.3,
        clip_act: int | None = 256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels,
            kernel=[3, 3],
        )
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels
            else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        y = mp_silu(self.conv_res0(mp_silu(x)))
        if self.training and self.dropout != 0:
            y = nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(
                dim=3
            )
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        latent_channels: int = 16,
        model_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16, 8),
        beta: float = 1.0,
        var: float = 1.0,
        resample_filter: tuple[int, int] = (1, 1),
        channels_per_head: int = 64,
        dropout: float = 0.0,
        res_balance: float = 0.3,
        attn_balance: float = 0.3,
        clip_act: int = 256,
    ):
        super().__init__()
        self.beta, self.var = beta, var
        self.register_buffer("const", torch.tensor(0.5 * np.log(2 * np.pi * self.var)))
        cblock = [model_channels * x for x in channel_mult]
        self.out_gain = nn.Parameter(torch.zeros([]))
        block_kwargs = dict(
            resample_filter=resample_filter,
            channels_per_head=channels_per_head,
            dropout=dropout,
            res_balance=res_balance,
            attn_balance=attn_balance,
            clip_act=clip_act,
        )

        self.enc = nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    cout, cout, flavor="enc", resample_mode="down", **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    flavor="enc",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
            if level == len(cblock) - 1:
                self.enc[f"{res}x{res}_out0"] = Block(
                    cout, cout, flavor="enc", attention=True, **block_kwargs
                )
                self.enc[f"{res}x{res}_out1"] = Block(
                    cout, cout, flavor="enc", **block_kwargs
                )
                if self.beta > 0:
                    head = nn.Conv2d(cout, 2 * latent_channels, 1)
                    nn.init.zeros_(head.weight), nn.init.zeros_(head.bias)
                else:
                    head = MPConv(cout, latent_channels, kernel=[1, 1])
                self.enc["enc_out"] = head

        self.dec = nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec["dec_in"] = MPConv(latent_channels, cout, kernel=[1, 1])
                self.dec[f"{res}x{res}_in0"] = Block(
                    cout, cout, flavor="dec", attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = Block(
                    cout, cout, flavor="dec", **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    cout, cout, flavor="dec", resample_mode="up", **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    flavor="dec",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_conv = MPConv(cout, img_channels, kernel=[3, 3])

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        is_stoch = self.beta > 0
        z = self.encode(x, sample=is_stoch)
        if is_stoch and isinstance(z, tuple):
            z, loc, scale = z
        rec = self.decode(z)
        nll = torch.mean((rec - x).pow(2) / (2 * self.var))  # + self.const)
        out = dict(loss=nll, nll=nll, rec=rec)
        if is_stoch:
            kl = 0.5 * (scale**2 + loc**2 - 1 - 2 * scale.log())
            out["kl"] = torch.mean(kl.sum(dim=(1, 2, 3)) / x[0].numel())  # per-pixel
            out["loss"] = out["nll"] + self.beta * out["kl"]
        return out

    def encode(self, x: Tensor, sample: bool = False) -> Tensor | tuple[Tensor, ...]:
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        for _, block in self.enc.items():
            x = block(x)
        if not sample:
            return x
        loc, raw_scale = x.chunk(2, dim=1)
        scale = nn.functional.softplus(raw_scale, beta=np.log(2)) + 1e-6
        z = loc + scale * torch.randn_like(loc)
        return z, loc, scale

    def decode(self, z: Tensor) -> Tensor:
        for _, block in self.dec.items():
            z = block(z)
        return self.out_conv(z, gain=self.out_gain)


def get_pretrained_vae(checkpoint_path: str) -> nn.Module:
    print(f"\nLoading vae: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path)
    args = argparse.Namespace(**ckpt["args"])
    config = VAEConfig(**{f.name: getattr(args, f.name) for f in fields(VAEConfig)})
    model = VAE(**vars(config))
    model.load_state_dict(ckpt["model_state_dict"])
    model.register_buffer("mean", torch.tensor(-0.507020))
    model.register_buffer("std", torch.tensor(3.663423))
    model.requires_grad_(False)
    model.eval()
    return model
