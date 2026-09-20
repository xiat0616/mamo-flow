#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.utils import save_image

PROJECT_ROOT = Path("/vol/biomedic3/tx1215/mamo-flow")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_handle.embed import get_embed, DatasetConfig
from src.sampling.sample_flow import (
    apply_single_intervention,
    build_sampling_root,
    generate_from_inverted_noise,
    get_class_schema,
    invert_to_noise,
    maybe_apply_ema,
)
from src.utils import get_pretrained_flux2vae


def to_namespace(d: dict) -> argparse.Namespace:
    return argparse.Namespace(**d)


def select_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


# ============================================================
# Dataset / latent helpers
# ============================================================

def unwrap_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset


def get_cache_spec(dataset):
    return getattr(
        unwrap_dataset(dataset),
        "cache_spec",
        None,
    )


def is_latent_dataset(dataset) -> bool:
    return get_cache_spec(dataset) is not None


def get_latent_normalization(dataset, device):
    spec = get_cache_spec(dataset)

    if spec is None:
        return None, None

    mean = torch.as_tensor(
        spec.mean,
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1, 1)

    std = torch.as_tensor(
        spec.std,
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1, 1)

    if not torch.isfinite(mean).all():
        raise ValueError("Latent mean contains NaN/Inf.")

    if not torch.isfinite(std).all():
        raise ValueError("Latent std contains NaN/Inf.")

    if torch.any(std <= 0):
        raise ValueError(
            "Latent std contains non-positive values."
        )

    return mean, std


def move_sample_pa_to_device(sample, device):
    return {
        k: v.unsqueeze(0).to(device)
        for k, v in sample["pa"].items()
    }


def prepare_sample_x(sample, device, latent_mode):
    """
    Return x in exactly the space used by the Flow.

    Image flow:
        dataset [0,1] -> flow [-1,1]

    Latent flow:
        dataset already returns normalized latent.
    """
    x = (
        sample["x"]
        .unsqueeze(0)
        .float()
        .to(device)
    )

    if not latent_mode:
        x = x * 2.0 - 1.0

    return x


# ============================================================
# cache_idx -> dataset index
# ============================================================

def build_cache_idx_lookup(dataset):
    """
    Build:

        cache_idx -> dataset item index

    This avoids assuming:

        dataset[i] == cache[i]

    even though that is usually true in the current pipeline.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base = unwrap_dataset(dataset)

        if not hasattr(base, "df"):
            raise AttributeError(
                "Underlying dataset has no df attribute."
            )

        subset_indices = np.asarray(
            dataset.indices,
            dtype=np.int64,
        )

        df = (
            base.df
            .iloc[subset_indices]
            .reset_index(drop=True)
        )

    else:
        base = dataset

        if not hasattr(base, "df"):
            raise AttributeError(
                "Dataset has no df attribute; cannot build "
                "cache_idx lookup."
            )

        df = base.df.reset_index(drop=True)

    if "cache_idx" not in df.columns:
        raise ValueError(
            "Dataset dataframe has no cache_idx column."
        )

    lookup = {}

    for dataset_idx, cache_idx in enumerate(
        df["cache_idx"].astype(int)
    ):
        cache_idx = int(cache_idx)

        if cache_idx in lookup:
            raise ValueError(
                f"Duplicate cache_idx={cache_idx}."
            )

        lookup[cache_idx] = dataset_idx

    return lookup


def get_sample_by_cache_idx(
    dataset,
    lookup,
    cache_idx,
):
    cache_idx = int(cache_idx)

    if cache_idx not in lookup:
        raise KeyError(
            f"cache_idx={cache_idx} not found in dataset."
        )

    return dataset[
        lookup[cache_idx]
    ]


# ============================================================
# Model builder
# ============================================================

def build_flow_model_from_train_args(
    train_args: argparse.Namespace,
    datasets,
    device: torch.device,
) -> nn.Module:

    from src.flows.flow import Flow
    from src.models.embedder import (
        CondEmbedderConfig,
        GlobalCondEmbedder,
        PerAttrCondEmbedder,
        infer_parent_dims_from_batch,
    )

    amp_dtype = select_amp_dtype(device)

    sample = datasets["train"][0]

    pa_batch = {
        k: v.unsqueeze(0)
        for k, v in sample["pa"].items()
    }

    parent_dims = infer_parent_dims_from_batch(
        pa_batch,
        train_args.parents,
    )

    cond_embedder = None

    if (
        train_args.cond_embedder != "none"
        and len(train_args.parents) > 0
    ):
        embedder_cfg = CondEmbedderConfig(
            parents=train_args.parents,
            parent_dims=parent_dims,
            cond_embed_dim=train_args.cond_embed_dim,
        )

        if train_args.cond_embedder == "per_attr":
            cond_embedder = PerAttrCondEmbedder(
                embedder_cfg
            )

        elif train_args.cond_embedder == "global":
            cond_embedder = GlobalCondEmbedder(
                embedder_cfg
            )

        else:
            raise ValueError(
                f"Unknown cond_embedder: "
                f"{train_args.cond_embedder}"
            )

    model_type = getattr(
        train_args,
        "model",
        None,
    )

    if model_type is None:
        model_type = (
            "dit"
            if (
                hasattr(train_args, "hidden_size")
                and hasattr(train_args, "patch_size")
            )
            else "unet"
        )

    if model_type == "dit":

        from src.models.DiT import DiT

        forward_nn = DiT(
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            patch_size=train_args.patch_size,
            in_channels=train_args.img_channels,
            hidden_size=train_args.hidden_size,
            depth=train_args.depth,
            num_heads=train_args.num_heads,
            mlp_ratio=train_args.mlp_ratio,
            cond_embed_dim=(
                train_args.cond_embed_dim
                if train_args.cond_embedder != "none"
                else 0
            ),
            grad_checkpointing=False,
        )

    elif model_type == "unet":

        from src.flows.flow import (
            BlockConfig,
            UNetConfig,
        )
        from src.models.unet import UNet

        unet_cfg = UNetConfig(
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            img_channels=train_args.img_channels,
            cond_embed_dim=train_args.cond_embed_dim,
            model_channels=train_args.model_channels,
            channel_mult=tuple(
                train_args.channel_mult
            ),
            channel_mult_time=getattr(
                train_args,
                "channel_mult_time",
                None,
            ),
            channel_mult_emb=getattr(
                train_args,
                "channel_mult_emb",
                None,
            ),
            num_blocks=train_args.num_blocks,
            attn_resolutions=tuple(
                train_args.attn_resolutions
            ),
            label_balance=train_args.label_balance,
            concat_balance=train_args.concat_balance,
        )

        block_cfg = BlockConfig(
            resample_filter=tuple(
                train_args.resample_filter
            ),
            channels_per_head=train_args.channels_per_head,
            dropout=train_args.dropout,
            res_balance=train_args.res_balance,
            attn_balance=train_args.attn_balance,
            clip_act=train_args.clip_act,
        )

        forward_nn = UNet(
            **vars(unet_cfg),
            **vars(block_cfg),
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    model = Flow(
        forward_nn=forward_nn,
        cond_embedder=cond_embedder,
        sigma=train_args.sigma,
        alpha=train_args.alpha,
        p_uncond=train_args.p_uncond,
        amp_dtype=amp_dtype,
    )

    return model.to(device)


# ============================================================
# Flow space -> image space
# ============================================================

@torch.inference_mode()
def decode_flow_tensor(
    x: torch.Tensor,
    latent_mode: bool,
    vae: nn.Module | None,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
) -> torch.Tensor:

    # Image flow:
    # already [-1,1]
    if not latent_mode:
        return x

    if vae is None:
        raise ValueError(
            "Latent flow requires VAE decoding."
        )

    if (
        latent_mean is None
        or latent_std is None
    ):
        raise ValueError(
            "Latent flow requires latent mean/std."
        )

    if x.ndim != 4:
        raise ValueError(
            f"Expected BCHW latent, got {tuple(x.shape)}"
        )

    if x.shape[1] != latent_mean.shape[1]:
        raise ValueError(
            f"Latent channel mismatch: "
            f"x={x.shape[1]}, "
            f"stats={latent_mean.shape[1]}"
        )

    # --------------------------------------------------------
    # Training:
    #
    # z_norm = (z_raw - mean) / std
    #
    # Decode:
    #
    # z_raw = z_norm * std + mean
    # --------------------------------------------------------

    z_raw = (
        x.float()
        * latent_std
        + latent_mean
    )

    vae_device = next(
        vae.parameters()
    ).device

    vae_dtype = next(
        vae.parameters()
    ).dtype

    z_raw = z_raw.to(
        device=vae_device,
        dtype=vae_dtype,
    )

    decoded = vae.decode(
        z_raw
    )

    if hasattr(
        decoded,
        "sample",
    ):
        decoded = decoded.sample

    return decoded.float()


def to_vis(x: torch.Tensor) -> torch.Tensor:

    if x.ndim != 4:
        raise ValueError(
            f"Expected BCHW image tensor, "
            f"got {tuple(x.shape)}"
        )

    if x.shape[1] not in {
        1,
        3,
        4,
    }:
        raise ValueError(
            f"Cannot visualize "
            f"{x.shape[1]} channels. "
            "Decode latent first."
        )

    return (
        (
            x.detach()
            .float()
            .clamp(-1, 1)
            + 1.0
        )
        / 2.0
    ).cpu()


# ============================================================
# Visualization
# ============================================================

def save_visual(
    x_src,
    x_recon,
    x_cf,
    x_gt,
    source_view,
    target_view,
    save_path,
):
    src = to_vis(
        x_src
    )[0, 0].numpy()

    recon = to_vis(
        x_recon
    )[0, 0].numpy()

    cf = to_vis(
        x_cf
    )[0, 0].numpy()

    gt = to_vis(
        x_gt
    )[0, 0].numpy()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 5),
    )

    axes[0].imshow(
        src,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title(
        f"Source {source_view}"
    )

    axes[1].imshow(
        recon,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        "Reconstruction"
    )

    axes[2].imshow(
        cf,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Generated {target_view} CF"
    )

    axes[3].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(
        f"Real {target_view}"
    )

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate paired CC<->MLO counterfactuals "
            "for image-space or latent-space Flow."
        )
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pair_csv",
        type=str,
        default=None,
        help=(
            "Optional pair CSV override. "
            "Image flow defaults to split_dir/"
            "EMBED_same_breast_CC_MLO_pairs.csv. "
            "Latent flow defaults to cache_dir/"
            "pair_csv_latent.csv."
        ),
    )

    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--source_view",
        type=str,
        default="CC",
        choices=[
            "CC",
            "MLO",
        ],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "train",
            "valid",
            "test",
            "all",
        ],
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--use_ema",
        type=int,
        default=1,
        choices=[0, 1],
    )

    parser.add_argument(
        "--ode_method",
        type=str,
        default="dopri5",
    )

    parser.add_argument(
        "--ode_atol",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--ode_rtol",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error(
            "--num_samples must be >= 1"
        )

    torch.manual_seed(
        args.seed
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    ckpt = torch.load(
        args.ckpt,
        map_location="cpu",
    )

    train_args = to_namespace(
        ckpt["args"]
    )

    if args.split_dir is not None:
        train_args.split_dir = (
            args.split_dir
        )

    # ========================================================
    # Dataset
    # ========================================================

    datasets = get_embed(
        DatasetConfig(
            data_dir=train_args.data_dir,
            split_dir=train_args.split_dir,
            cache_dir=getattr(
                train_args,
                "cache_dir",
                None,
            ),
            parents=train_args.parents,
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            img_channels=train_args.img_channels,
            vae_ckpt=getattr(
                train_args,
                "vae_ckpt",
                None,
            ),
        )
    )

    latent_mode = is_latent_dataset(
        datasets["train"]
    )

    print(
        "Sampling mode:",
        (
            "LATENT FLOW"
            if latent_mode
            else "IMAGE FLOW"
        ),
    )

    # ========================================================
    # Build cache_idx lookup
    # ========================================================

    cache_idx_lookup = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:
        cache_idx_lookup[split] = (
            build_cache_idx_lookup(
                datasets[split]
            )
        )

    # ========================================================
    # Latent normalization + VAE
    # ========================================================

    vae = None
    latent_mean = None
    latent_std = None

    if latent_mode:

        latent_mean, latent_std = (
            get_latent_normalization(
                datasets["train"],
                device,
            )
        )

        vae_ckpt = getattr(
            train_args,
            "vae_ckpt",
            None,
        )

        if vae_ckpt != "flux2":
            raise ValueError(
                f"Latent checkpoint uses "
                f"vae_ckpt={vae_ckpt!r}; "
                "only FLUX.2 is supported here."
            )

        vae = (
            get_pretrained_flux2vae()
            .to(device)
        )

        vae.requires_grad_(
            False
        )

        vae.eval()

        print(
            "Latent shape:",
            (
                train_args.img_channels,
                train_args.img_height,
                train_args.img_width,
            ),
        )

        print(
            "Latent normalization channels:",
            latent_mean.shape[1],
        )

    # ========================================================
    # Pair CSV
    #
    # IMAGE:
    #
    # split_dir/EMBED_same_breast_CC_MLO_pairs.csv
    #
    # LATENT:
    #
    # cache_dir/pair_csv_latent.csv
    # ========================================================

    if args.pair_csv is not None:

        pair_csv = Path(
            args.pair_csv
        )

    elif latent_mode:

        cache_dir = getattr(
            train_args,
            "cache_dir",
            None,
        )

        if cache_dir is None:
            raise ValueError(
                "Latent mode detected but checkpoint "
                "has no cache_dir."
            )

        pair_csv = (
            Path(cache_dir)
            / "pair_csv_latent.csv"
        )

    else:

        pair_csv = (
            Path(
                train_args.split_dir
            )
            / "EMBED_same_breast_CC_MLO_pairs.csv"
        )

    if not pair_csv.exists():
        raise FileNotFoundError(
            f"Pair CSV not found: {pair_csv}"
        )

    pairs_df = pd.read_csv(
        pair_csv,
        low_memory=False,
    )

    required_pair_cols = {
        "split",
        "cc_cache_idx",
        "mlo_cache_idx",
    }

    missing = (
        required_pair_cols
        - set(pairs_df.columns)
    )

    if missing:
        raise ValueError(
            f"Pair CSV missing columns: "
            f"{sorted(missing)}"
        )

    if args.split != "all":
        pairs_df = pairs_df.loc[
            pairs_df["split"]
            == args.split
        ].copy()

    pairs_df = (
        pairs_df
        .reset_index()
        .rename(
            columns={
                "index":
                    "pair_csv_index"
            }
        )
    )

    if len(pairs_df) == 0:
        raise ValueError(
            f"No paired cases found "
            f"for split={args.split}"
        )

    n = min(
        args.num_samples,
        len(pairs_df),
    )

    if args.shuffle:

        pairs_df = (
            pairs_df
            .sample(
                n=n,
                random_state=args.seed,
                replace=False,
            )
            .reset_index(
                drop=True
            )
        )

    else:

        pairs_df = (
            pairs_df
            .iloc[:n]
            .reset_index(
                drop=True
            )
        )

    print(
        f"Using {len(pairs_df)} paired cases"
    )

    print(
        "Pair CSV:",
        pair_csv,
    )

    # ========================================================
    # Model
    # ========================================================

    model = (
        build_flow_model_from_train_args(
            train_args=train_args,
            datasets=datasets,
            device=device,
        )
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ],
        strict=True,
    )

    maybe_apply_ema(
        model=model,
        ckpt=ckpt,
        ema_rate=getattr(
            train_args,
            "ema_rate",
            0.9999,
        ),
        use_ema=bool(
            args.use_ema
        ),
    )

    model.eval()

    print(
        "Model loaded:",
        getattr(
            train_args,
            "model",
            "legacy/unet",
        ),
    )

    class_schema = (
        get_class_schema(
            train_args
        )
    )

    source_view = (
        args.source_view
    )

    target_view = (
        "MLO"
        if source_view == "CC"
        else "CC"
    )

    # ========================================================
    # Output
    # ========================================================

    sampling_root = (
        build_sampling_root(
            save_root=args.save_dir,
            ckpt_path=args.ckpt,
            train_args=train_args,
            ode_method=args.ode_method,
            ode_atol=args.ode_atol,
            ode_rtol=args.ode_rtol,
            ode_steps=args.ode_steps,
        )
    )

    out_root = (
        sampling_root
        / "paired_view_cfs"
        / f"{source_view}_to_{target_view}"
        / args.split
    )

    dirs = {
        "inputs":
            out_root
            / "inputs",

        "reconstructions":
            out_root
            / "reconstructions",

        "cfs":
            out_root
            / "cfs",

        "ground_truth":
            out_root
            / "ground_truth",

        "visuals":
            out_root
            / "visuals",
    }

    for d in dirs.values():
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    meta = {
        "ckpt":
            args.ckpt,

        "pair_csv":
            str(pair_csv),

        "num_samples_requested":
            args.num_samples,

        "num_samples_used":
            len(pairs_df),

        "source_view":
            source_view,

        "target_view":
            target_view,

        "split":
            args.split,

        "shuffle":
            args.shuffle,

        "seed":
            args.seed,

        "use_ema":
            bool(
                args.use_ema
            ),

        "ode_method":
            args.ode_method,

        "ode_atol":
            args.ode_atol,

        "ode_rtol":
            args.ode_rtol,

        "ode_steps":
            args.ode_steps,

        "latent_mode":
            latent_mode,

        "cache_dir":
            getattr(
                train_args,
                "cache_dir",
                None,
            ),

        "vae_ckpt":
            getattr(
                train_args,
                "vae_ckpt",
                None,
            ),
    }

    with open(
        out_root
        / "sampling_args.json",
        "w",
    ) as f:

        json.dump(
            meta,
            f,
            indent=2,
        )

    # ========================================================
    # Counterfactual generation
    # ========================================================

    records = []

    for output_idx, row in pairs_df.iterrows():

        split = str(
            row["split"]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # pair CSV stores CACHE indices.
        #
        # We explicitly map:
        #
        # cache_idx -> dataset index
        #
        # before retrieving each sample.
        # ----------------------------------------------------

        cc_sample = (
            get_sample_by_cache_idx(
                datasets[split],
                cache_idx_lookup[split],
                row["cc_cache_idx"],
            )
        )

        mlo_sample = (
            get_sample_by_cache_idx(
                datasets[split],
                cache_idx_lookup[split],
                row["mlo_cache_idx"],
            )
        )

        if source_view == "CC":
            source_sample = cc_sample
            gt_sample = mlo_sample

        else:
            source_sample = mlo_sample
            gt_sample = cc_sample

        # ====================================================
        # FLOW SPACE
        #
        # image:
        #     [-1,1]
        #
        # latent:
        #     normalized FLUX latent
        # ====================================================

        x_src = prepare_sample_x(
            source_sample,
            device,
            latent_mode,
        )

        x_gt = prepare_sample_x(
            gt_sample,
            device,
            latent_mode,
        )

        pa_src = (
            move_sample_pa_to_device(
                source_sample,
                device,
            )
        )

        # ----------------------------------------------------
        # do(view)
        # ----------------------------------------------------

        pa_cf = (
            apply_single_intervention(
                pa=pa_src,
                pa_rand=None,
                do_key="view",
                do_mode="flip",
                class_schema=class_schema,
            )
        )

        src_view_idx = int(
            pa_src["view"]
            .argmax(dim=-1)
            .item()
        )

        expected_src_idx = (
            1
            if source_view == "CC"
            else 0
        )

        if (
            src_view_idx
            != expected_src_idx
        ):
            raise ValueError(
                f"Pair {output_idx}: "
                f"source view mismatch. "
                f"Requested {source_view}, "
                f"but pa gives "
                f"class {src_view_idx}."
            )

        # ====================================================
        # COUNTERFACTUAL IN FLOW SPACE
        #
        # factual
        #   ↓ inverse Flow
        # noise
        #   ↓ factual condition
        # reconstruction
        #
        # same noise
        #   ↓ do(view)
        # counterfactual
        #
        # For latent models this whole section operates on
        # normalized FLUX latents.
        # ====================================================

        with torch.inference_mode():

            noise = invert_to_noise(
                model=model,
                x=x_src,
                pa_src=pa_src,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )

            x_recon = (
                generate_from_inverted_noise(
                    model=model,
                    noise=noise,
                    pa_cf=pa_src,
                    ode_method=args.ode_method,
                    ode_atol=args.ode_atol,
                    ode_rtol=args.ode_rtol,
                    ode_steps=args.ode_steps,
                )
            )

            x_cf = (
                generate_from_inverted_noise(
                    model=model,
                    noise=noise,
                    pa_cf=pa_cf,
                    ode_method=args.ode_method,
                    ode_atol=args.ode_atol,
                    ode_rtol=args.ode_rtol,
                    ode_steps=args.ode_steps,
                )
            )

        # ====================================================
        # FLOW SPACE -> IMAGE SPACE
        #
        # latent:
        #
        # z_norm
        #   ↓
        # z_raw = z_norm * std + mean
        #   ↓
        # FLUX VAE decode
        #   ↓
        # mammogram
        # ====================================================

        x_src_img = decode_flow_tensor(
            x_src,
            latent_mode,
            vae,
            latent_mean,
            latent_std,
        )

        x_recon_img = decode_flow_tensor(
            x_recon,
            latent_mode,
            vae,
            latent_mean,
            latent_std,
        )

        x_cf_img = decode_flow_tensor(
            x_cf,
            latent_mode,
            vae,
            latent_mean,
            latent_std,
        )

        x_gt_img = decode_flow_tensor(
            x_gt,
            latent_mode,
            vae,
            latent_mean,
            latent_std,
        )

        # ====================================================
        # Save
        # ====================================================

        name = (
            f"{output_idx:06d}"
        )

        save_image(
            to_vis(
                x_src_img
            )[0],
            dirs["inputs"]
            / (
                f"{name}_source_"
                f"{source_view}.png"
            ),
        )

        save_image(
            to_vis(
                x_recon_img
            )[0],
            dirs["reconstructions"]
            / (
                f"{name}_recon_"
                f"{source_view}.png"
            ),
        )

        save_image(
            to_vis(
                x_cf_img
            )[0],
            dirs["cfs"]
            / (
                f"{name}_cf_"
                f"{target_view}.png"
            ),
        )

        save_image(
            to_vis(
                x_gt_img
            )[0],
            dirs["ground_truth"]
            / (
                f"{name}_real_"
                f"{target_view}.png"
            ),
        )

        save_visual(
            x_src=x_src_img,
            x_recon=x_recon_img,
            x_cf=x_cf_img,
            x_gt=x_gt_img,
            source_view=source_view,
            target_view=target_view,
            save_path=(
                dirs["visuals"]
                / f"{name}_comparison.png"
            ),
        )

        record = {
            "output_idx":
                output_idx,

            "pair_csv_index":
                int(
                    row[
                        "pair_csv_index"
                    ]
                ),

            "split":
                split,

            "source_view":
                source_view,

            "target_view":
                target_view,

            "cc_cache_idx":
                int(
                    row[
                        "cc_cache_idx"
                    ]
                ),

            "mlo_cache_idx":
                int(
                    row[
                        "mlo_cache_idx"
                    ]
                ),
        }

        # Optional metadata.
        for key in [
            "empi_anon",
            "acc_anon",
            "laterality",
        ]:
            if key in row:
                record[key] = row[key]

        if "shortpath" in cc_sample:
            record[
                "cc_shortpath"
            ] = cc_sample[
                "shortpath"
            ]

        if "shortpath" in mlo_sample:
            record[
                "mlo_shortpath"
            ] = mlo_sample[
                "shortpath"
            ]

        records.append(
            record
        )

        print(
            f"[{output_idx + 1}/"
            f"{len(pairs_df)}] "
            f"{source_view}"
            f"->{target_view} | "
            f"split={split} | "
            f"cc_idx="
            f"{int(row['cc_cache_idx'])} | "
            f"mlo_idx="
            f"{int(row['mlo_cache_idx'])}"
        )

    pd.DataFrame(
        records
    ).to_csv(
        out_root
        / "samples.csv",
        index=False,
    )

    print()
    print(
        "Done."
    )
    print(
        "Saved to:",
        out_root,
    )


if __name__ == "__main__":
    main()