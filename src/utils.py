from __future__ import annotations

import builtins
import copy
import datetime
import inspect
import numbers
import os
import random
import time

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torchvision.utils import make_grid


DEBUG = False
# DEBUG = True


# ============================================================
# Pretty-name maps
# ============================================================

DEFAULT_IDX_TO_NAME: dict[
    str,
    list[str],
] = {

    "view": [
        "MLO",
        "non-MLO",
    ],

    "density": (
        [
            "D",
            "C",
            "B",
            "A",
        ]
        if DEBUG
        else [
            "A",
            "B",
            "C",
            "D",
        ]
    ),

    "scanner": [
        "Selenia Dimensions",
        "Senographe Pristina",
        "Senograph 2000D",
        "Lorad Selenia",
        "Clearview CSm",
        "Senographe Essential",
    ],

    "y": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}


# ============================================================
# EMA
# ============================================================

class ModelEMA:

    def __init__(
        self,
        params: Iterable[
            torch.nn.Parameter
        ],
        rate: float = 0.999,
    ):

        self.rate = float(
            rate
        )

        self.params = list(
            params
        )

        self.ema_params = [
            copy.deepcopy(
                p
            )
            .detach()
            .requires_grad_(
                False
            )
            for p
            in self.params
        ]

        self.stored_params: (
            list[torch.Tensor]
            | None
        ) = None


    @torch.no_grad()
    def update(
        self,
    ) -> None:

        for ema_p, p in zip(
            self.ema_params,
            self.params,
            strict=True,
        ):

            ema_p.mul_(
                self.rate
            ).add_(
                p,
                alpha=(
                    1.0
                    - self.rate
                ),
            )


    @torch.no_grad()
    def apply(
        self,
    ) -> None:

        self.stored_params = [
            p.detach().clone()
            for p
            in self.params
        ]

        for p, ema_p in zip(
            self.params,
            self.ema_params,
            strict=True,
        ):

            p.copy_(
                ema_p
            )


    @torch.no_grad()
    def restore(
        self,
    ) -> None:

        if self.stored_params is None:

            raise RuntimeError(
                "ModelEMA.restore() "
                "called before apply()."
            )

        for p, stored_p in zip(
            self.params,
            self.stored_params,
            strict=True,
        ):

            p.copy_(
                stored_p
            )

        self.stored_params = None


    def state_dict(
        self,
    ) -> list[torch.Tensor]:

        return [
            p.detach().clone()
            for p
            in self.ema_params
        ]


    def load_state_dict(
        self,
        state: list[
            torch.Tensor
        ],
    ) -> None:

        if (
            len(state)
            != len(
                self.ema_params
            )
        ):

            raise ValueError(
                "EMA state length mismatch: "
                f"got {len(state)}, "
                "expected "
                f"{len(self.ema_params)}."
            )

        for ema_p, saved_p in zip(
            self.ema_params,
            state,
            strict=True,
        ):

            ema_p.copy_(
                saved_p
            )


# ============================================================
# Reproducibility
# ============================================================

def seed_all(
    seed: int,
    deterministic: bool = True,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    if deterministic:

        torch.backends.cudnn.benchmark = (
            False
        )

        torch.backends.cudnn.deterministic = (
            True
        )


def seed_worker(
    worker_id: int,
) -> None:

    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


# ============================================================
# Distributed helpers
# ============================================================

def setup_distributed(
    suppress_non_master_print: bool = True,
) -> tuple[
    torch.device,
    int,
    int,
]:
    """
    Returns:
        device
        rank
        world_size

    Supports:
        torchrun
        SLURM
        local single-process execution
    """

    use_torchrun = (
        "RANK" in os.environ
        and "WORLD_SIZE"
        in os.environ
    )

    use_slurm = (
        "SLURM_PROCID"
        in os.environ
        and "SLURM_NTASKS"
        in os.environ
    )

    # --------------------------------------------------------
    # torchrun
    # --------------------------------------------------------

    if use_torchrun:

        rank = int(
            os.environ[
                "RANK"
            ]
        )

        local_rank = int(
            os.environ[
                "LOCAL_RANK"
            ]
        )

        world_size = int(
            os.environ[
                "WORLD_SIZE"
            ]
        )

    # --------------------------------------------------------
    # SLURM
    # --------------------------------------------------------

    elif use_slurm:

        rank = int(
            os.environ[
                "SLURM_PROCID"
            ]
        )

        local_rank = int(
            os.environ.get(
                "SLURM_LOCALID",
                0,
            )
        )

        world_size = int(
            os.environ[
                "SLURM_NTASKS"
            ]
        )

    # --------------------------------------------------------
    # Local
    # --------------------------------------------------------

    else:

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        if device.type == "cuda":

            torch.cuda.set_device(
                device
            )

        return (
            device,
            0,
            1,
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            f"cuda:{local_rank}"
        )

        torch.cuda.set_device(
            device
        )

        backend = "nccl"

    else:

        device = torch.device(
            "cpu"
        )

        backend = "gloo"

    # No need for process group for world size 1.
    if world_size <= 1:

        return (
            device,
            rank,
            world_size,
        )

    # --------------------------------------------------------
    # Initialise process group
    # --------------------------------------------------------

    init_pg_kwargs: dict[
        str,
        Any,
    ] = {
        "backend": backend,
        "world_size": world_size,
        "rank": rank,
        "timeout":
            datetime.timedelta(
                minutes=30
            ),
    }

    if (
        device.type == "cuda"
        and "device_id"
        in inspect.signature(
            dist.init_process_group
        ).parameters
    ):

        init_pg_kwargs[
            "device_id"
        ] = device

    if not dist.is_initialized():

        dist.init_process_group(
            **init_pg_kwargs
        )

    master_addr = (
        os.environ.get(
            "MASTER_ADDR",
            "?",
        )
    )

    master_port = (
        os.environ.get(
            "MASTER_PORT",
            "?",
        )
    )

    print(
        f"WORLD_SIZE: {world_size}, "
        f"RANK: {rank}, "
        f"LOCAL_RANK: {local_rank}, "
        f"MASTER: "
        f"{master_addr}:"
        f"{master_port}"
    )

    # --------------------------------------------------------
    # Suppress non-master printing
    # --------------------------------------------------------

    if (
        rank != 0
        and suppress_non_master_print
    ):

        def _print_pass(
            *args: Any,
            **kwargs: Any,
        ) -> None:

            del args, kwargs

            return None

        builtins.print = (
            _print_pass
        )

    return (
        device,
        rank,
        world_size,
    )


def unwrap(
    model: torch.nn.Module,
) -> torch.nn.Module:

    while True:

        # torch.compile
        if hasattr(
            model,
            "_orig_mod",
        ):

            model = (
                model._orig_mod
            )

            continue

        # DDP
        if isinstance(
            model,
            torch.nn.parallel
            .DistributedDataParallel,
        ):

            model = (
                model.module
            )

            continue

        return model


# ============================================================
# FLUX.2 VAE
# ============================================================

def get_pretrained_flux2vae(
) -> torch.nn.Module:

    from diffusers.models import (
        AutoencoderKLFlux2
    )

    print(
        "\nLoading FLUX.2 vae..."
    )

    model = (
        AutoencoderKLFlux2
        .from_pretrained(
            "black-forest-labs/"
            "FLUX.2-dev",
            subfolder="vae",
            torch_dtype=(
                torch.bfloat16
            ),
        )
    )

    # ========================================================
    # Grayscale encoder
    #
    # [B,1,H,W]
    #      ↓ repeat
    # [B,3,H,W]
    #      ↓
    # FLUX encoder
    #      ↓
    # posterior mode
    # ========================================================

    orig_encode = (
        model.encode
    )

    model.encode = (
        lambda x:
            orig_encode(
                x.repeat(
                    1,
                    3,
                    1,
                    1,
                )
            )
            .latent_dist
            .sample()
    )

    # ========================================================
    # Grayscale decoder
    #
    # latent
    #   ↓
    # FLUX RGB decoder
    #   ↓
    # [B,3,H,W]
    #   ↓ RGB mean
    # [B,1,H,W]
    # ========================================================

    orig_decode = (
        model.decode
    )

    model.decode = (
        lambda z:
            orig_decode(
                z
            )
            .sample
            .mean(
                dim=1,
                keepdim=True,
            )
    )

    # --------------------------------------------------------
    # Reference FLUX latent normalization constants.
    #
    # IMPORTANT:
    #
    # These are stored because your latent-cache writer
    # records them in metadata.
    #
    # Your CURRENT embed.py training normalization instead
    # uses train-set per-channel mean/std.
    #
    # Therefore save_plots() does NOT automatically use these.
    # It uses dataset.cache_spec.mean/std instead.
    # --------------------------------------------------------

    model.register_buffer(
        "mean",
        torch.tensor(
            -0.061467
        ),
    )

    model.register_buffer(
        "std",
        torch.tensor(
            1.633637
        ),
    )

    model.requires_grad_(
        False
    )

    model.eval()

    return model


# ============================================================
# Small utilities
# ============================================================

@torch.inference_mode()
def get_mc_stats(
    samples: (
        torch.Tensor
        | list[
            torch.Tensor
        ]
    ),
    prefix: str | None = None,
) -> dict[
    str,
    torch.Tensor,
]:

    x = (
        samples
        if torch.is_tensor(
            samples
        )
        else torch.stack(
            samples
        )
    )

    x = (
        x.detach()
        .float()
    )

    if x.ndim == 0:

        x = x.unsqueeze(
            0
        )

    mc_samples = int(
        x.shape[0]
    )

    if mc_samples < 1:

        raise ValueError(
            "Need at least one sample "
            "to compute MC stats."
        )

    mean = x.mean(
        dim=0
    )

    std = (
        x.std(
            dim=0
        )
        if mc_samples > 1
        else torch.zeros_like(
            mean
        )
    )

    out = {

        "mean":
            mean,

        "std":
            std,

        "se":
            (
                std
                / (
                    mc_samples
                    ** 0.5
                )
            ),
    }

    if prefix:

        return {
            f"{prefix}_{k}":
                v
            for k, v
            in out.items()
        }

    return out


# ============================================================
# Pretty metadata values
# ============================================================

def value_to_name(
    pa: dict[
        str,
        Tensor,
    ],
    idx_to_name: (
        dict[
            str,
            list[str],
        ]
        | None
    ) = None,
) -> dict[
    str,
    list[str],
]:

    idx_to_name = (
        DEFAULT_IDX_TO_NAME
        if idx_to_name is None
        else idx_to_name
    )

    out = {
        k: []
        for k
        in pa.keys()
    }

    def _scalar(
        x: Tensor,
    ) -> float:

        if x.numel() != 1:

            raise ValueError(
                "Expected scalar tensor, "
                f"got shape "
                f"{tuple(x.shape)}"
            )

        return float(
            x.item()
        )

    def _is_nan(
        x: float,
    ) -> bool:

        return bool(
            np.isnan(x)
        )

    for k, v in pa.items():

        for i in range(
            v.shape[0]
        ):

            vi = v[i]

            is_onehot = (
                vi.numel() > 1
            )

            # ------------------------------------------------
            # One-hot categorical
            # ------------------------------------------------

            if is_onehot:

                idx = int(
                    vi.argmax()
                    .item()
                )

                if k in idx_to_name:

                    names = (
                        idx_to_name[k]
                    )

                    out[k].append(
                        names[idx]
                        if (
                            0
                            <= idx
                            < len(names)
                        )
                        else str(idx)
                    )

                else:

                    out[k].append(
                        str(idx)
                    )

                continue

            # ------------------------------------------------
            # Scalar
            # ------------------------------------------------

            val = _scalar(
                vi
            )

            if k == "age":

                out[k].append(
                    "nan"
                    if _is_nan(val)
                    else str(
                        int(
                            round(
                                val
                                * 100
                            )
                        )
                    )
                )

                continue

            if k in idx_to_name:

                if _is_nan(
                    val
                ):

                    out[k].append(
                        "nan"
                    )

                else:

                    idx = int(
                        round(val)
                    )

                    names = (
                        idx_to_name[k]
                    )

                    out[k].append(
                        names[idx]
                        if (
                            0
                            <= idx
                            < len(names)
                        )
                        else str(idx)
                    )

                continue

            if _is_nan(
                val
            ):

                out[k].append(
                    "nan"
                )

            elif (
                abs(
                    val
                    - round(val)
                )
                < 1e-6
            ):

                out[k].append(
                    str(
                        int(
                            round(val)
                        )
                    )
                )

            else:

                out[k].append(
                    f"{val:.3f}"
                )

    return out


# ============================================================
# Internal helpers
# ============================================================

def _get_module_device(
    module: torch.nn.Module,
) -> torch.device:

    try:

        return next(
            module.parameters()
        ).device

    except StopIteration:

        try:

            return next(
                module.buffers()
            ).device

        except StopIteration:

            return torch.device(
                "cpu"
            )


def _autocast_bf16(
    device: torch.device,
):

    if device.type in {
        "cuda",
        "cpu",
    }:

        try:

            return torch.autocast(
                device_type=(
                    device.type
                ),
                dtype=(
                    torch.bfloat16
                ),
            )

        except Exception:

            return nullcontext()

    return nullcontext()


def _maybe_get_sample(
    obj: Any,
) -> Tensor:

    return (
        obj.sample
        if hasattr(
            obj,
            "sample",
        )
        else obj
    )


# ============================================================
# Dataset mode helpers
# ============================================================

def _get_cache_spec(
    dataset: torch.utils.data.Dataset,
) -> Any | None:
    """
    Return dataset.cache_spec if present.

    Also supports torch.utils.data.Subset.
    """

    current = dataset

    while isinstance(
        current,
        torch.utils.data.Subset,
    ):

        current = (
            current.dataset
        )

    return getattr(
        current,
        "cache_spec",
        None,
    )


def _is_latent_dataset(
    dataset: torch.utils.data.Dataset,
) -> bool:

    return (
        _get_cache_spec(
            dataset
        )
        is not None
    )


def _get_latent_normalization(
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> tuple[
    Tensor | None,
    Tensor | None,
]:
    """
    Get the EXACT latent normalization used by embed.py.

    For latent EMBED datasets:

        z_norm =
            (z_raw - mean)
            / std

    Returns:

        mean:
            [1,C,1,1]

        std:
            [1,C,1,1]

    Raw image dataset:
        returns (None, None)
    """

    cache_spec = (
        _get_cache_spec(
            dataset
        )
    )

    if cache_spec is None:

        return (
            None,
            None,
        )

    mean = torch.as_tensor(
        cache_spec.mean,
        dtype=torch.float32,
        device=device,
    ).view(
        1,
        -1,
        1,
        1,
    )

    std = torch.as_tensor(
        cache_spec.std,
        dtype=torch.float32,
        device=device,
    ).view(
        1,
        -1,
        1,
        1,
    )

    if not torch.isfinite(
        mean
    ).all():

        raise ValueError(
            "Latent normalization mean "
            "contains NaN/Inf."
        )

    if not torch.isfinite(
        std
    ).all():

        raise ValueError(
            "Latent normalization std "
            "contains NaN/Inf."
        )

    if torch.any(
        std <= 0
    ):

        raise ValueError(
            "Latent normalization std "
            "contains non-positive values."
        )

    return (
        mean,
        std,
    )


# ============================================================
# FLUX VAE decoding helpers
# ============================================================

def _decode_with_vae(
    x: Tensor,
    vae: (
        torch.nn.Module
        | None
    ),
    latent_mean: Tensor | None = None,
    latent_std: Tensor | None = None,
) -> Tensor:
    """
    Decode normalized FLOW latent into image.

    Flow latent:
        z_norm

    Training normalization:
        z_norm =
            (z_raw - mean) / std

    Therefore:
        z_raw =
            z_norm * std + mean

    Then:
        FLUX VAE decode(z_raw)
    """

    # Raw image flow:
    # no VAE decoding.
    if vae is None:

        return x

    if (
        latent_mean is None
        or latent_std is None
    ):

        raise ValueError(
            "VAE decoding for latent flow "
            "requires latent_mean and latent_std."
        )

    vae_device = (
        _get_module_device(
            vae
        )
    )

    x = x.to(
        device=vae_device,
        dtype=torch.float32,
    )

    latent_mean = (
        latent_mean.to(
            device=vae_device,
            dtype=torch.float32,
        )
    )

    latent_std = (
        latent_std.to(
            device=vae_device,
            dtype=torch.float32,
        )
    )

    # --------------------------------------------------------
    # Validate channels
    # --------------------------------------------------------

    if (
        x.ndim != 4
    ):

        raise ValueError(
            "Expected latent BCHW tensor, "
            f"got {tuple(x.shape)}."
        )

    if (
        latent_mean.shape[1]
        != x.shape[1]
    ):

        raise ValueError(
            "Latent mean channel mismatch: "
            f"x has {x.shape[1]} channels, "
            "mean has "
            f"{latent_mean.shape[1]}."
        )

    if (
        latent_std.shape[1]
        != x.shape[1]
    ):

        raise ValueError(
            "Latent std channel mismatch: "
            f"x has {x.shape[1]} channels, "
            "std has "
            f"{latent_std.shape[1]}."
        )

    # ========================================================
    # Denormalize:
    #
    # z_norm -> raw FLUX latent
    # ========================================================

    z = (
        x
        * latent_std
        + latent_mean
    )

    # --------------------------------------------------------
    # Match FLUX VAE dtype.
    # --------------------------------------------------------

    vae_dtype = next(
        vae.parameters()
    ).dtype

    z = z.to(
        device=vae_device,
        dtype=vae_dtype,
    )

    # --------------------------------------------------------
    # Decode
    #
    # get_pretrained_flux2vae() already averages RGB.
    # --------------------------------------------------------

    with _autocast_bf16(
        z.device
    ):

        decoded = (
            vae.decode(
                z
            )
        )

    decoded = (
        _maybe_get_sample(
            decoded
        )
    )

    return (
        decoded.float()
    )


def _decode_pair_with_vae(
    x1: Tensor,
    x2: Tensor,
    vae: (
        torch.nn.Module
        | None
    ),
    latent_mean: Tensor | None = None,
    latent_std: Tensor | None = None,
) -> tuple[
    Tensor,
    Tensor,
]:
    """
    Decode original + counterfactual normalized latents together.
    """

    if vae is None:

        return (
            x1,
            x2,
        )

    if (
        latent_mean is None
        or latent_std is None
    ):

        raise ValueError(
            "VAE decoding for latent flow "
            "requires latent_mean and latent_std."
        )

    vae_device = (
        _get_module_device(
            vae
        )
    )

    x1 = x1.to(
        device=vae_device,
        dtype=torch.float32,
    )

    x2 = x2.to(
        device=vae_device,
        dtype=torch.float32,
    )

    latent_mean = (
        latent_mean.to(
            device=vae_device,
            dtype=torch.float32,
        )
    )

    latent_std = (
        latent_std.to(
            device=vae_device,
            dtype=torch.float32,
        )
    )

    if (
        x1.shape[1]
        != latent_mean.shape[1]
    ):

        raise ValueError(
            "Latent channel mismatch "
            "between x and normalization."
        )

    # --------------------------------------------------------
    # Combine batch
    # --------------------------------------------------------

    latents = torch.cat(
        [
            x1,
            x2,
        ],
        dim=0,
    )

    # --------------------------------------------------------
    # normalized latent -> raw FLUX latent
    # --------------------------------------------------------

    z = (
        latents
        * latent_std
        + latent_mean
    )

    vae_dtype = next(
        vae.parameters()
    ).dtype

    z = z.to(
        device=vae_device,
        dtype=vae_dtype,
    )

    with _autocast_bf16(
        z.device
    ):

        decoded = (
            vae.decode(
                z
            )
        )

    decoded = (
        _maybe_get_sample(
            decoded
        )
        .float()
    )

    out1, out2 = (
        decoded.chunk(
            2,
            dim=0,
        )
    )

    return (
        out1,
        out2,
    )


# ============================================================
# Display utilities
# ============================================================

def _to_display_range(
    x: Tensor,
) -> Tensor:

    return (
        (
            x.float()
            .clamp(
                min=-1.0,
                max=1.0,
            )
            + 1.0
        )
        * 0.5
    )


def _save_image_grid(
    x: Tensor,
    file_path: str | None = None,
    nrow: int = 6,
) -> None:

    x = (
        x.detach()
        .cpu()
    )

    if x.ndim != 4:

        raise ValueError(
            "Expected BCHW tensor for image grid, "
            f"got {tuple(x.shape)}."
        )

    if x.shape[1] not in {
        1,
        3,
        4,
    }:

        raise ValueError(
            "Cannot display tensor with "
            f"{x.shape[1]} channels. "
            "For latent flow, pass the FLUX VAE "
            "so the latent is decoded first."
        )

    ncol = min(
        nrow,
        max(
            1,
            x.shape[0],
        ),
    )

    nrows = int(
        np.ceil(
            x.shape[0]
            / nrow
        )
    )

    plt.figure(
        figsize=(
            3.2 * ncol,
            3.2 * nrows,
        )
    )

    plt.axis(
        "off"
    )

    grid = (
        make_grid(
            x,
            padding=0,
            nrow=nrow,
        )
        .permute(
            1,
            2,
            0,
        )
    )

    if grid.shape[-1] == 1:

        plt.imshow(
            grid[..., 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

    else:

        plt.imshow(
            grid
        )

    plt.tight_layout()

    if file_path is not None:

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()


# ============================================================
# Metadata schema
# ============================================================

def _get_class_schema(
) -> dict[
    str,
    Any,
]:

    schema: dict[
        str,
        Any,
    ] = {}

    try:

        from src.data_handle.embed import (
            CLASS_SCHEMA
            as EMBED_CLASS_SCHEMA
        )

        schema.update(
            EMBED_CLASS_SCHEMA
        )

    except Exception:

        pass

    return schema


# ============================================================
# Random interventions
# ============================================================

def _sample_random_interventions(
    pa: dict[
        str,
        Tensor,
    ],
    class_schema: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> tuple[
    dict[
        str,
        Tensor,
    ],
    list[str],
]:

    class_schema = (
        {}
        if class_schema is None
        else class_schema
    )

    if len(pa) == 0:

        raise ValueError(
            "No parent keys found "
            "for counterfactual plotting."
        )

    do_pa = {
        k: v.clone()
        for k, v
        in pa.items()
    }

    do_keys: list[str] = []

    pa_keys = list(
        pa.keys()
    )

    batch_size = (
        next(
            iter(
                pa.values()
            )
        ).shape[0]
    )

    for i in range(
        batch_size
    ):

        k = pa_keys[
            i
            % len(
                pa_keys
            )
        ]

        do_keys.append(
            k
        )

        y = (
            do_pa[k][i]
            .clone()
        )

        schema_val = (
            class_schema.get(
                k,
                None,
            )
        )

        is_onehot = (
            y.ndim > 0
            and y.numel() > 1
        )

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        if isinstance(
            schema_val,
            numbers.Integral,
        ):

            num_classes = int(
                schema_val
            )

            is_categorical = True

        elif is_onehot:

            num_classes = int(
                y.numel()
            )

            is_categorical = True

        else:

            num_classes = None

            is_categorical = False

        if is_categorical:

            assert (
                num_classes
                is not None
            )

            if num_classes <= 1:

                do_pa[k][i] = (
                    y.clone()
                )

                continue

            if is_onehot:

                orig_idx = int(
                    y.argmax()
                    .item()
                )

            else:

                orig_idx = int(
                    y.round()
                    .item()
                )

            new_idx = random.randrange(
                num_classes
            )

            tries = 0

            while (
                new_idx == orig_idx
                and tries < 32
            ):

                new_idx = (
                    random.randrange(
                        num_classes
                    )
                )

                tries += 1

            if new_idx == orig_idx:

                new_idx = (
                    orig_idx + 1
                ) % num_classes

            if is_onehot:

                do_y = (
                    torch.zeros_like(
                        y
                    )
                )

                do_y[
                    new_idx
                ] = 1.0

            else:

                do_y = torch.tensor(
                    new_idx,
                    dtype=y.dtype,
                    device=y.device,
                )

            do_pa[k][i] = (
                do_y
            )

        # ----------------------------------------------------
        # Continuous
        # ----------------------------------------------------

        else:

            do_pa[k][i] = (
                torch.rand_like(
                    do_pa[k][i]
                )
            )

    return (
        do_pa,
        do_keys,
    )


# ============================================================
# Image display
# ============================================================

def _show_image(
    ax: plt.Axes,
    img: Tensor,
) -> None:

    if img.ndim == 2:

        ax.imshow(
            img,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        return

    if img.ndim != 3:

        raise ValueError(
            "Expected 2D or 3D image tensor, "
            f"got shape {tuple(img.shape)}"
        )

    if img.shape[0] == 1:

        ax.imshow(
            img[0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

    elif img.shape[0] in {
        3,
        4,
    }:

        ax.imshow(
            img.permute(
                1,
                2,
                0,
            ).clamp(
                0,
                1,
            )
        )

    else:

        raise ValueError(
            "Unsupported channel count: "
            f"{img.shape[0]}"
        )


# ============================================================
# Effect map
# ============================================================

def _make_effect_map(
    x: Tensor,
    cf_x: Tensor,
) -> Tensor:

    diff = (
        cf_x
        - x
    ).detach().cpu()

    if diff.ndim != 4:

        raise ValueError(
            "Expected BCHW tensor, "
            f"got shape "
            f"{tuple(diff.shape)}"
        )

    if diff.shape[1] == 1:

        return (
            diff[:, 0]
            * 255.0
        )

    return (
        diff.mean(
            dim=1
        )
        * 255.0
    )


# ============================================================
# Counterfactual plot
# ============================================================

def _plot_counterfactual_triplets(
    x: Tensor,
    cf_x: Tensor,
    pa: dict[
        str,
        Tensor,
    ],
    do_pa: dict[
        str,
        Tensor,
    ],
    do_keys: list[str],
    file_path: str | None = None,
) -> None:

    x = (
        x.detach()
        .cpu()
    )

    cf_x = (
        cf_x.detach()
        .cpu()
    )

    effect = (
        _make_effect_map(
            x,
            cf_x,
        )
    )

    pa_named = (
        value_to_name(
            pa
        )
    )

    do_pa_named = (
        value_to_name(
            do_pa
        )
    )

    bs = x.shape[0]

    cols = 6

    fs = 20

    nrows = (
        3
        * int(
            np.ceil(
                bs
                / cols
            )
        )
    )

    plt.rcParams[
        "font.size"
    ] = fs

    fig, axes = (
        plt.subplots(
            nrows,
            cols,
            figsize=(
                8 * cols,
                8 * nrows + 8,
            ),
        )
    )

    if nrows == 1:

        axes = np.array(
            [
                axes
            ]
        )

    axes = (
        np.atleast_2d(
            axes
        )
    )

    count = [
        0,
        0,
        0,
    ]

    for idx, ax in enumerate(
        axes.flatten()
    ):

        row = (
            idx
            // cols
        )

        group = (
            row
            % 3
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        if (
            group == 0
            and count[group] < bs
        ):

            sample_idx = (
                count[group]
            )

            k = (
                do_keys[
                    sample_idx
                ]
            )

            _show_image(
                ax,
                x[
                    sample_idx
                ],
            )

            ax.set_xlabel(
                f"{k}: "
                f"{pa_named[k][sample_idx]}",
                labelpad=8,
            )

            count[group] += 1

        # ----------------------------------------------------
        # Counterfactual
        # ----------------------------------------------------

        elif (
            group == 1
            and count[group] < bs
        ):

            sample_idx = (
                count[group]
            )

            k = (
                do_keys[
                    sample_idx
                ]
            )

            _show_image(
                ax,
                cf_x[
                    sample_idx
                ],
            )

            ax.set_xlabel(
                f"do("
                f"{k}="
                f"{do_pa_named[k][sample_idx]}"
                f")",
                labelpad=8,
            )

            count[group] += 1

        # ----------------------------------------------------
        # Effect
        # ----------------------------------------------------

        elif (
            group == 2
            and count[group] < bs
        ):

            sample_idx = (
                count[group]
            )

            img = (
                effect[
                    sample_idx
                ]
            )

            amax = float(
                img.abs()
                .max()
            )

            if amax == 0:

                amax = 1.0

            eff = ax.imshow(
                img,
                cmap="RdBu_r",
                vmin=-amax,
                vmax=amax,
            )

            divider = (
                make_axes_locatable(
                    ax
                )
            )

            cax = (
                divider
                .append_axes(
                    "bottom",
                    size="3%",
                    pad=0.08,
                )
            )

            cbar = (
                fig.colorbar(
                    eff,
                    cax=cax,
                    orientation=(
                        "horizontal"
                    ),
                )
            )

            cbar.outline.set_visible(
                False
            )

            cbar.ax.tick_params(
                labelsize=(
                    fs // 2
                )
            )

            ticks = np.linspace(
                np.ceil(
                    -amax
                ),
                np.floor(
                    amax
                ),
                5,
            )

            cbar.set_ticks(
                np.round(
                    ticks
                )
            )

            count[group] += 1

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )

        for spine in (
            ax.spines.values()
        ):

            spine.set_visible(
                False
            )

    fig.subplots_adjust(
        wspace=0.055,
        hspace=0.01,
    )

    if file_path is not None:

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()


# ============================================================
# Preview batch
# ============================================================

def _sample_preview_batch(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    max_samples: int,
) -> tuple[
    Tensor,
    dict[
        str,
        Tensor,
    ],
]:
    """
    Return data in FLOW space.

    Raw-image mode:

        dataset:
            [0,1]

        flow:
            [-1,1]

    Latent mode:

        dataset:
            already normalized latent

        flow:
            same normalized latent

    IMPORTANT:
        Never apply x * 2 - 1 to latent data.
    """

    requested = max(
        1,
        6
        * round(
            batch_size
            / 6
        ),
    )

    num_samples = min(
        requested,
        max_samples,
        len(dataset),
    )

    if num_samples <= 0:

        raise ValueError(
            "Dataset contains no samples."
        )

    dataloader = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=num_samples,
            shuffle=True,
        )
    )

    batch = next(
        iter(
            dataloader
        )
    )

    x = (
        batch["x"]
        .float()
        .to(
            device
        )
    )

    latent_mode = (
        _is_latent_dataset(
            dataset
        )
    )

    # --------------------------------------------------------
    # Raw image:
    #
    # [0,1] -> [-1,1]
    # --------------------------------------------------------

    if not latent_mode:

        x = (
            x * 2.0
            - 1.0
        )

    # --------------------------------------------------------
    # Latent:
    #
    # embed.py already returns normalized latent.
    # Do nothing.
    # --------------------------------------------------------

    pa = {
        k: v.to(
            device
        )
        for k, v
        in batch[
            "pa"
        ].items()
    }

    return (
        x,
        pa,
    )


# ============================================================
# Standard Flow plotting
# ============================================================

@torch.inference_mode()
def plot_samples(
    u: Tensor,
    pa: dict[
        str,
        Tensor,
    ],
    model: torch.nn.Module,
    vae: (
        torch.nn.Module
        | None
    ) = None,
    latent_mean: Tensor | None = None,
    latent_std: Tensor | None = None,
    steps: int | None = None,
    file_path: str | None = None,
    **kwargs,
) -> None:

    ode_kwargs = dict(
        kwargs
    )

    if steps is not None:

        ode_kwargs = {

            "method":
                "euler",

            "t":
                model.schedule(
                    torch.linspace(
                        0.0,
                        1.0,
                        steps,
                        device=u.device,
                    ),
                    model.alpha,
                ),
        }

    # ========================================================
    # Noise -> data in FLOW space
    #
    # Raw:
    #   [-1,1]
    #
    # Latent:
    #   normalized FLUX latent
    # ========================================================

    x = (
        model.ode_solve(
            x=u,
            pa=pa,
            **ode_kwargs,
        )[-1]
    )

    # ========================================================
    # Latent -> image
    #
    # If vae=None:
    #   raw image passes through unchanged.
    #
    # If VAE supplied:
    #   normalized latent
    #       ↓ denormalize
    #   raw FLUX latent
    #       ↓ VAE
    #   grayscale image
    # ========================================================

    x = (
        _decode_with_vae(
            x,
            vae,
            latent_mean=(
                latent_mean
            ),
            latent_std=(
                latent_std
            ),
        )
    )

    x = (
        _to_display_range(
            x
        )
    )

    _save_image_grid(
        x,
        file_path=file_path,
    )


# ============================================================
# Counterfactual plotting
# ============================================================

@torch.inference_mode()
def plot_counterfactuals(
    x: Tensor,
    pa: dict[
        str,
        Tensor,
    ],
    model: torch.nn.Module,
    vae: (
        torch.nn.Module
        | None
    ) = None,
    latent_mean: Tensor | None = None,
    latent_std: Tensor | None = None,
    steps: int | None = None,
    file_path: str | None = None,
    **kwargs,
) -> None:

    class_schema = (
        _get_class_schema()
    )

    do_pa, do_keys = (
        _sample_random_interventions(
            pa,
            class_schema=(
                class_schema
            ),
        )
    )

    inv_kwargs = dict(
        kwargs
    )

    gen_kwargs = dict(
        kwargs
    )

    # ========================================================
    # Integration grids
    # ========================================================

    if steps is not None:

        # ----------------------------------------------------
        # Abduction:
        #
        # data -> noise
        # ----------------------------------------------------

        inv_kwargs = {

            "method":
                "euler",

            "t":
                model.schedule(
                    torch.linspace(
                        1.0,
                        0.0,
                        steps,
                        device=x.device,
                    ),
                    model.alpha,
                ),
        }

        # ----------------------------------------------------
        # Prediction:
        #
        # noise -> data
        # ----------------------------------------------------

        gen_kwargs = {

            "method":
                "euler",

            "t":
                model.schedule(
                    torch.linspace(
                        0.0,
                        1.0,
                        steps,
                        device=x.device,
                    ),
                    model.alpha,
                ),
        }

    else:

        inv_kwargs.setdefault(
            "t",
            torch.tensor(
                [
                    1.0,
                    0.0,
                ],
                device=x.device,
            ),
        )

        gen_kwargs.setdefault(
            "t",
            torch.tensor(
                [
                    0.0,
                    1.0,
                ],
                device=x.device,
            ),
        )

    # ========================================================
    # ABDUCTION
    #
    # Raw:
    #   x is image in [-1,1]
    #
    # Latent:
    #   x is normalized latent
    # ========================================================

    u = (
        model.ode_solve(
            x,
            pa=pa,
            **inv_kwargs,
        )[-1]
    )

    # ========================================================
    # ACTION + PREDICTION
    # ========================================================

    cf_x = (
        model.ode_solve(
            u,
            pa=do_pa,
            **gen_kwargs,
        )[-1]
    )

    # ========================================================
    # Latent decoding
    #
    # IMPORTANT:
    #
    # x and cf_x stay normalized while inside the flow.
    #
    # Only AFTER prediction:
    #
    # z_raw =
    #     z_norm * train_std + train_mean
    #
    # then FLUX decode.
    # ========================================================

    if vae is not None:

        t0 = time.time()

        x, cf_x = (
            _decode_pair_with_vae(
                x,
                cf_x,
                vae,
                latent_mean=(
                    latent_mean
                ),
                latent_std=(
                    latent_std
                ),
            )
        )

        print(
            "VAE inference time elapsed: "
            f"{time.time() - t0:.2f}s"
        )

    # ========================================================
    # [-1,1] -> [0,1]
    # ========================================================

    x = (
        _to_display_range(
            x
        )
    )

    cf_x = (
        _to_display_range(
            cf_x
        )
    )

    _plot_counterfactual_triplets(
        x=x,
        cf_x=cf_x,
        pa=pa,
        do_pa=do_pa,
        do_keys=do_keys,
        file_path=file_path,
    )


# ============================================================
# save_plots
# ============================================================

@torch.inference_mode()
def save_plots(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    steps: int,
    save_path: str,
    vae: (
        torch.nn.Module
        | None
    ) = None,
) -> None:
    """
    Supports BOTH:

    1. raw-image Flow:
        dataset [0,1]
            ↓
        flow [-1,1]
            ↓
        image

    2. FLUX latent Flow:
        dataset normalized latent
            ↓
        flow normalized latent
            ↓
        denormalize using cache TRAIN stats
            ↓
        FLUX VAE
            ↓
        grayscale image
    """

    base = unwrap(
        model
    )

    device = (
        _get_module_device(
            base
        )
    )

    latent_mode = (
        _is_latent_dataset(
            dataset
        )
    )

    # --------------------------------------------------------
    # Latent mode requires VAE for visualization.
    # --------------------------------------------------------

    if (
        latent_mode
        and vae is None
    ):

        raise ValueError(
            "save_plots(): latent dataset detected "
            "but vae=None. "
            "Pass the FLUX.2 VAE to visualize "
            "latent-flow outputs."
        )

    # ========================================================
    # Get preview data in FLOW space.
    # ========================================================

    x, pa = (
        _sample_preview_batch(
            batch_size=(
                batch_size
            ),
            dataset=(
                dataset
            ),
            device=(
                device
            ),
            max_samples=12,
        )
    )

    # ========================================================
    # Retrieve exact latent normalization.
    #
    # Raw:
    #   None, None
    #
    # Latent:
    #   [1,C,1,1] train mean/std
    # ========================================================

    latent_mean, latent_std = (
        _get_latent_normalization(
            dataset,
            device=device,
        )
    )

    if latent_mode:

        print(
            "save_plots: latent flow"
        )

        print(
            "  flow input shape:",
            tuple(
                x.shape
            ),
        )

        print(
            "  normalization channels:",
            (
                latent_mean.shape[1]
                if latent_mean
                is not None
                else None
            ),
        )

    else:

        print(
            "save_plots: raw-image flow"
        )

        print(
            "  flow input shape:",
            tuple(
                x.shape
            ),
        )

    # ========================================================
    # Random generation
    # ========================================================

    u = torch.randn_like(
        x
    )

    plot_samples(
        u,
        pa=pa,
        model=base,
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
        steps=steps,
        file_path=(
            save_path
            + ".pdf"
        ),
    )

    # ========================================================
    # Counterfactual generation
    # ========================================================

    if len(pa) > 0:

        plot_counterfactuals(
            x,
            pa=pa,
            model=base,
            vae=vae,
            latent_mean=latent_mean,
            latent_std=latent_std,
            steps=steps,
            file_path=(
                save_path
                + "_cf.pdf"
            ),
        )


# ============================================================
# MeanFlow plotting
# ============================================================

@torch.inference_mode()
def plot_samples_mf(
    noise: Tensor,
    pa: dict[
        str,
        Tensor,
    ],
    model: torch.nn.Module,
    vae: (
        torch.nn.Module
        | None
    ) = None,
    latent_mean: Tensor | None = None,
    latent_std: Tensor | None = None,
    steps: int = 1,
    file_path: str | None = None,
) -> None:

    x = (
        model.sample(
            noise,
            steps=steps,
            pa=pa,
        )
    )

    x = (
        _decode_with_vae(
            x,
            vae,
            latent_mean=(
                latent_mean
            ),
            latent_std=(
                latent_std
            ),
        )
    )

    x = (
        _to_display_range(
            x
        )
    )

    _save_image_grid(
        x,
        file_path=file_path,
    )


@torch.inference_mode()
def plot_counterfactuals_mf(
    x: Tensor,
    pa: dict[
        str,
        Tensor,
    ],
    model: torch.nn.Module,
    vae: (
        torch.nn.Module
        | None
    ) = None,
    latent_mean: Tensor | None = None,
    latent_std: Tensor | None = None,
    steps: int = 1,
    file_path: str | None = None,
    **kwargs,
) -> None:

    class_schema = (
        _get_class_schema()
    )

    do_pa, do_keys = (
        _sample_random_interventions(
            pa,
            class_schema=(
                class_schema
            ),
        )
    )

    inv_sample_args = (
        kwargs.pop(
            "inv_sample_args",
            None,
        )
    )

    gen_sample_args = (
        kwargs.pop(
            "gen_sample_args",
            None,
        )
    )

    inv_steps = (
        kwargs.pop(
            "inv_steps",
            150,
        )
    )

    ode_kwargs = dict(
        kwargs
    )

    if inv_steps is not None:

        ode_kwargs.setdefault(
            "method",
            "euler",
        )

        ode_kwargs.setdefault(
            "t",
            torch.linspace(
                0.0,
                1.0,
                inv_steps,
                device=x.device,
            ),
        )

    else:

        ode_kwargs.setdefault(
            "t",
            torch.tensor(
                [
                    0.0,
                    1.0,
                ],
                device=x.device,
            ),
        )

    noise = (
        model.ode_solve(
            x,
            pa=pa,
            sample_args=(
                inv_sample_args
            ),
            **ode_kwargs,
        )[-1]
    )

    cf_x = (
        model.sample(
            noise,
            steps=steps,
            pa=do_pa,
            sample_args=(
                gen_sample_args
            ),
        )
    )

    if vae is not None:

        t0 = time.time()

        x, cf_x = (
            _decode_pair_with_vae(
                x,
                cf_x,
                vae,
                latent_mean=(
                    latent_mean
                ),
                latent_std=(
                    latent_std
                ),
            )
        )

        print(
            "VAE inference time elapsed: "
            f"{time.time() - t0:.2f}s"
        )

    x = (
        _to_display_range(
            x
        )
    )

    cf_x = (
        _to_display_range(
            cf_x
        )
    )

    _plot_counterfactual_triplets(
        x=x,
        cf_x=cf_x,
        pa=pa,
        do_pa=do_pa,
        do_keys=do_keys,
        file_path=file_path,
    )


# ============================================================
# save_plots_mf
# ============================================================

@torch.inference_mode()
def save_plots_mf(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    steps: int,
    save_path: str,
    vae: (
        torch.nn.Module
        | None
    ) = None,
) -> None:

    base = unwrap(
        model
    )

    device = (
        _get_module_device(
            base
        )
    )

    latent_mode = (
        _is_latent_dataset(
            dataset
        )
    )

    if (
        latent_mode
        and vae is None
    ):

        raise ValueError(
            "save_plots_mf(): latent dataset "
            "detected but vae=None."
        )

    x, pa = (
        _sample_preview_batch(
            batch_size=(
                batch_size
            ),
            dataset=(
                dataset
            ),
            device=(
                device
            ),
            max_samples=12,
        )
    )

    latent_mean, latent_std = (
        _get_latent_normalization(
            dataset,
            device=device,
        )
    )

    plot_samples_mf(
        torch.randn_like(
            x
        ),
        pa=pa,
        model=base,
        vae=vae,
        latent_mean=(
            latent_mean
        ),
        latent_std=(
            latent_std
        ),
        steps=steps,
        file_path=(
            save_path
            + ".pdf"
        ),
    )

    if len(pa) > 0:

        plot_counterfactuals_mf(
            x,
            pa=pa,
            model=base,
            vae=vae,
            latent_mean=(
                latent_mean
            ),
            latent_std=(
                latent_std
            ),
            steps=steps,
            file_path=(
                save_path
                + "_cf.pdf"
            ),
        )


# ============================================================
# Reconstruction plots
# ============================================================

@torch.inference_mode()
def save_reconstructions(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    file_path: str | None = None,
) -> None:

    base = unwrap(
        model
    )

    device = (
        _get_module_device(
            base
        )
    )

    if (
        not hasattr(
            base,
            "encode",
        )
        or not hasattr(
            base,
            "decode",
        )
    ):

        raise AttributeError(
            "save_reconstructions() "
            "expects model to have "
            "encode() and decode()."
        )

    num_samples = min(
        max(
            1,
            6
            * round(
                batch_size
                / 6
            ),
        ),
        18,
        len(dataset),
    )

    dataloader = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=num_samples,
            shuffle=True,
        )
    )

    batch = next(
        iter(
            dataloader
        )
    )

    x = (
        batch["x"]
        .float()
        .to(
            device
        )
    )

    # save_reconstructions is currently intended
    # for pixel-space encoder/decoder models.
    if _is_latent_dataset(
        dataset
    ):

        raise ValueError(
            "save_reconstructions() is currently "
            "for pixel-space encoder/decoder models, "
            "not cached latent-flow datasets."
        )

    x = (
        x * 2.0
        - 1.0
    )

    with _autocast_bf16(
        x.device
    ):

        rec = base.decode(
            base.encode(
                x
            )
        )

        rec = (
            _maybe_get_sample(
                rec
            )
        )

    interleaved = (
        torch.stack(
            (
                x,
                rec,
            ),
            dim=1,
        )
        .reshape(
            -1,
            *x.shape[1:],
        )
    )

    interleaved = (
        _to_display_range(
            interleaved
        )
    )

    _save_image_grid(
        interleaved,
        file_path=file_path,
    )