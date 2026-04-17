"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models",
adapted for MeanFlow (https://arxiv.org/abs/2505.13447).

Changes vs. original EDM2 UNet:
  - forward() signature: (x, r, t, cond_emb) instead of (x, time_labels, cond_emb)
  - Second Fourier + linear embedding for the interval dt = t - r
  - Two time embeddings are summed before being passed to blocks
"""

import numpy as np
import torch

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)

#----------------------------------------------------------------------------

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------

def resample(x, f=(1, 1), mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(
            x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
        )
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
    )

#----------------------------------------------------------------------------

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------

class Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        emb_channels,
        flavor              = 'enc',
        resample_mode       = 'keep',
        resample_filter     = [1,1],
        attention           = False,
        channels_per_head   = 64,
        dropout             = 0,
        res_balance         = 0.3,
        attn_balance        = 0.3,
        clip_act            = 256,
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
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net adapted for MeanFlow.
#
# Key change: the network is conditioned on TWO time variables.
# Following Table 1c / Appendix A of the MeanFlow paper, we embed
# t and dt = t - r separately, then sum the two embeddings.
#----------------------------------------------------------------------------

class UNet(torch.nn.Module):
    def __init__(
        self,
        img_height,
        img_width,
        img_channels,
        cond_embed_dim,
        model_channels      = 192,
        channel_mult        = (1, 2, 3, 4),
        channel_mult_time   = None,
        channel_mult_emb    = None,
        num_blocks          = 3,
        attn_resolutions    = ((16, 16), (8, 8)),
        label_balance       = 0.5,
        concat_balance      = 0.5,
        **block_kwargs,
    ):
        super().__init__()

        cblock = [model_channels * n for n in channel_mult]
        ctime = model_channels * channel_mult_time if channel_mult_time is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)

        self.cemb = cemb
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        attn_resolutions = {tuple(r) for r in attn_resolutions}

        # ---- Time embeddings (MeanFlow: two separate embeddings) ----
        # Embedding for t
        self.emb_fourier_t = MPFourier(ctime)
        self.emb_time_t = MPConv(ctime, cemb, kernel=[])

        # Embedding for dt = t - r  (the interval)
        self.emb_fourier_dt = MPFourier(ctime)
        self.emb_time_dt = MPConv(ctime, cemb, kernel=[])

        # Conditioning embedding (unchanged)
        self.emb_cond = MPConv(cond_embed_dim, cemb, kernel=[]) if cond_embed_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            h = img_height >> level
            w = img_width >> level
            res_hw = (h, w)

            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{h}x{w}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{h}x{w}_down"] = Block(
                    cout, cout, cemb,
                    flavor="enc",
                    resample_mode="down",
                    **block_kwargs,
                )

            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{h}x{w}_block{idx}"] = Block(
                    cin, cout, cemb,
                    flavor="enc",
                    attention=(res_hw in attn_resolutions),
                    **block_kwargs,
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]

        for level, channels in reversed(list(enumerate(cblock))):
            h = img_height >> level
            w = img_width >> level
            res_hw = (h, w)

            if level == len(cblock) - 1:
                self.dec[f"{h}x{w}_in0"] = Block(
                    cout, cout, cemb,
                    flavor="dec",
                    attention=True,
                    **block_kwargs,
                )
                self.dec[f"{h}x{w}_in1"] = Block(
                    cout, cout, cemb,
                    flavor="dec",
                    **block_kwargs,
                )
            else:
                self.dec[f"{h}x{w}_up"] = Block(
                    cout, cout, cemb,
                    flavor="dec",
                    resample_mode="up",
                    **block_kwargs,
                )

            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{h}x{w}_block{idx}"] = Block(
                    cin, cout, cemb,
                    flavor="dec",
                    attention=(res_hw in attn_resolutions),
                    **block_kwargs,
                )

        self.out_conv = MPConv(cout, img_channels, kernel=[3, 3])

    def forward(self, x, r, t, cond_emb):
        """
        Args:
            x:        input tensor (B, C, H, W)
            r:        start time  (B,)  — scalar per sample
            t:        end time    (B,)  — scalar per sample
            cond_emb: conditioning embedding (B, D) or None
        """
        # ---- Build combined time embedding ----
        # Embed t and dt = t - r separately, then sum  (Appendix A)
        dt = t - r
        emb_t = self.emb_time_t(self.emb_fourier_t(t))
        emb_dt = self.emb_time_dt(self.emb_fourier_dt(dt))
        emb = emb_t + emb_dt

        if self.emb_cond is not None and cond_emb is not None:
            emb = mp_sum(
                emb,
                self.emb_cond(cond_emb * np.sqrt(cond_emb.shape[1])),
                t=self.label_balance,
            )
        emb = mp_silu(emb)

        # ---- Encoder ----
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # ---- Decoder ----
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)

        x = self.out_conv(x, gain=self.out_gain)
        return x