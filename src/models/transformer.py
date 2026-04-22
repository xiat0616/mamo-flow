# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        assert img_height % patch_size == 0 and img_width % patch_size == 0, (
            f"Image size ({img_height}x{img_width}) must be divisible by patch_size ({patch_size})"
        )
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.img_height and W == self.img_width, (
            f"Input size ({H}x{W}) doesn't match expected ({self.img_height}x{self.img_width})"
        )
        x = self.proj(x)                   # (B, D, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)   # (B, T, D)
        return x


class RMSNormFP32(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape: tuple[int, ...] = (dim,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.float() if self.weight is not None else None
        return nn.functional.rms_norm(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qk_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, dim * 3)
        if self.qk_norm:
            self.q_norm = RMSNormFP32(self.head_dim, eps=1e-6)
            self.k_norm = RMSNormFP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, T, DH)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.proj(x))


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def init_gate_weights(self):
        assert self.fc1.bias is not None
        nn.init.ones_(self.fc1.bias[self.hidden_features:])
        nn.init.normal_(self.fc1.weight[self.hidden_features:], std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = nn.functional.silu(gate) * x
        x = self.drop1(x)
        return self.drop2(self.fc2(x))


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 8 / 3,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qk_norm=qk_norm, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio), dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        img_height: int = 32,
        img_width: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 8 / 3,
        cond_embed_dim: int = 0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.grad_checkpointing = grad_checkpointing

        self.x_embedder = PatchEmbed(img_height, img_width, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Project cond_emb to hidden_size if dimensions differ
        self.cond_proj = (
            nn.Linear(cond_embed_dim, hidden_size)
            if cond_embed_dim > 0 and cond_embed_dim != hidden_size
            else None
        )
        self._has_cond = cond_embed_dim > 0

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.x_embedder.grid_h,
            self.x_embedder.grid_w,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.x_embedder.proj.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            block.mlp.init_gate_weights()

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: Tensor) -> Tensor:
        p = self.x_embedder.patch_size
        h = self.x_embedder.grid_h
        w = self.x_embedder.grid_w
        c = self.out_channels
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: Tensor, t: Tensor, cond_emb: Tensor | None = None) -> Tensor:
        x = self.x_embedder(x) + self.pos_embed  # (B, T, D)
        c = self.t_embedder(t)                    # (B, D)

        if cond_emb is not None:
            c = c + (self.cond_proj(cond_emb) if self.cond_proj is not None else cond_emb)

        for block in self.blocks:
            if self.training and self.grad_checkpointing:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        x = self.final_layer(x, c)  # (B, T, patch_size**2 * C)
        return self.unpatchify(x)   # (B, C, H, W)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    grid_col = np.arange(grid_w, dtype=np.float32)
    grid_row = np.arange(grid_h, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_col, grid_row), axis=0)  # (2, grid_h, grid_w)
    grid = grid.reshape(2, 1, grid_h, grid_w)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)                       # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
