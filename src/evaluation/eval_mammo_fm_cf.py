#!/usr/bin/env python3

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


HF_REPO = "batmanLab/Mammo-FM"
HF_CKPT = "Mammo-FM_ASU_Trained_CLIP.tar"

# Defaults can be overridden from the .sh with:
#
#   --height 1024 --width 768
#
DEFAULT_HEIGHT = 1520
DEFAULT_WIDTH = 912
DEFAULT_MEAN = 0.3089279
DEFAULT_STD = 0.25053555408335154

# IMPORTANT:
# The released Mammo-FM checkpoint uses efficientnet_pytorch
# parameter names:
#
#   _conv_stem
#   _bn0
#   _blocks
#   _conv_head
#   ...
#
# not timm parameter names.
DEFAULT_MODEL = "efficientnet-b5"


# ============================================================
# Checkpoint / model
# ============================================================

class MammoFMImageEncoder(nn.Module):
    """
    Mammo-FM EfficientNet-B5 image encoder returning the
    pooled image representation before the classifier.

    Output:
        [B, 2048]
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.flatten(1)


def get_tensor_state_dict(ckpt):
    """
    Find the actual tensor state_dict inside the Mammo-FM
    checkpoint.
    """
    for key in [
        "model",
        "state_dict",
        "model_state_dict",
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]

            if any(
                torch.is_tensor(v)
                for v in state.values()
            ):
                return state

    if isinstance(ckpt, dict) and any(
        torch.is_tensor(v)
        for v in ckpt.values()
    ):
        return ckpt

    raise ValueError(
        "Could not find model state_dict "
        "in Mammo-FM checkpoint."
    )


def extract_image_encoder_state(state, backbone):
    """
    Extract the image encoder from the full Mammo-FM VLM.

    The downloaded checkpoint contains keys such as:

        image_encoder._conv_stem.weight
        image_encoder._bn0.weight
        image_encoder._blocks.0._depthwise_conv.weight
        ...

    These match efficientnet_pytorch EfficientNet-B5 after
    removing the 'image_encoder.' prefix.
    """
    backbone_state = backbone.state_dict()

    prefixes = [
        "image_encoder.",
        "module.image_encoder.",
        "model.image_encoder.",
    ]

    best_prefix = None
    best_state = {}

    for prefix in prefixes:
        candidate = {}

        for key, value in state.items():
            if not key.startswith(prefix):
                continue

            new_key = key[len(prefix):]

            if (
                new_key in backbone_state
                and backbone_state[new_key].shape == value.shape
            ):
                candidate[new_key] = value

        if len(candidate) > len(best_state):
            best_prefix = prefix
            best_state = candidate

    if len(best_state) == 0:
        print("\nFirst checkpoint keys:")

        for key in list(state.keys())[:50]:
            print(" ", key)

        raise ValueError(
            "Could not identify Mammo-FM EfficientNet "
            "image encoder weights."
        )

    # _fc is irrelevant because feature extraction stops before it.
    required_backbone_keys = {
        key
        for key in backbone_state
        if not key.startswith("_fc.")
    }

    matched_required_keys = (
        set(best_state)
        & required_backbone_keys
    )

    coverage = (
        len(matched_required_keys)
        / len(required_backbone_keys)
    )

    print(
        "Selected checkpoint prefix:",
        best_prefix,
    )

    print(
        f"Matched EfficientNet feature tensors: "
        f"{len(matched_required_keys)}/"
        f"{len(required_backbone_keys)} "
        f"({coverage * 100:.1f}%)"
    )

    if coverage < 0.99:
        missing = sorted(
            required_backbone_keys
            - matched_required_keys
        )

        print(
            "\nFirst missing EfficientNet keys:"
        )

        for key in missing[:30]:
            print(" ", key)

        raise ValueError(
            f"Only {coverage * 100:.1f}% of the "
            "EfficientNet feature extractor matched."
        )

    return best_state


def load_mammo_fm(
    device,
    model_name=DEFAULT_MODEL,
    hf_cache_dir=None,
):
    """
    Download the Mayo/ASU Mammo-FM checkpoint from Hugging Face
    and load only its EfficientNet-B5 image encoder.
    """
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_CKPT,
        cache_dir=hf_cache_dir,
    )

    print(
        "Mammo-FM checkpoint:",
        ckpt_path,
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state = get_tensor_state_dict(
        ckpt
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use efficientnet_pytorch, not timm.
    # --------------------------------------------------------

    backbone = EfficientNet.from_name(
        model_name
    )

    image_state = extract_image_encoder_state(
        state,
        backbone,
    )

    missing, unexpected = backbone.load_state_dict(
        image_state,
        strict=False,
    )

    # We never use the EfficientNet classification layer.
    meaningful_missing = [
        key
        for key in missing
        if not key.startswith("_fc.")
    ]

    if meaningful_missing:
        raise ValueError(
            "Missing Mammo-FM EfficientNet weights:\n"
            + "\n".join(
                meaningful_missing[:30]
            )
        )

    meaningful_unexpected = [
        key
        for key in unexpected
        if not key.startswith("_fc.")
    ]

    if meaningful_unexpected:
        raise ValueError(
            "Unexpected Mammo-FM EfficientNet weights:\n"
            + "\n".join(
                meaningful_unexpected[:30]
            )
        )

    model = MammoFMImageEncoder(
        backbone
    )

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Determine feature dimension without running a huge image.
    with torch.inference_mode():
        dummy = torch.zeros(
            1,
            3,
            224,
            224,
            device=device,
        )

        feature_dim = int(
            model(dummy).shape[-1]
        )

    print(
        "Mammo-FM image encoder loaded."
    )

    print(
        "Backbone:",
        model_name,
    )

    print(
        "Feature dimension:",
        feature_dim,
    )

    return model, ckpt_path, feature_dim


# ============================================================
# Preprocessing
# ============================================================

class MammoDataset(Dataset):

    def __init__(
        self,
        paths,
        height,
        width,
        mean,
        std,
    ):
        self.paths = [
            Path(p)
            for p in paths
        ]

        self.transform = transforms.Compose([
            transforms.Resize(
                (height, width),
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    mean,
                    mean,
                    mean,
                ],
                std=[
                    std,
                    std,
                    std,
                ],
            ),
        ])

    def __len__(self):
        return len(
            self.paths
        )

    def __getitem__(self, idx):
        path = self.paths[idx]

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        image = (
            Image.open(path)
            .convert("RGB")
        )

        return self.transform(
            image
        )


# ============================================================
# Feature extraction
# ============================================================

@torch.inference_mode()
def extract_features(
    model,
    paths,
    device,
    height,
    width,
    mean,
    std,
    batch_size,
    num_workers,
    amp,
    desc,
):
    dataset = MammoDataset(
        paths=paths,
        height=height,
        width=width,
        mean=mean,
        std=std,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
        drop_last=False,
    )

    features = []

    for x in tqdm(
        loader,
        desc=desc,
    ):
        x = x.to(
            device,
            non_blocking=True,
        )

        ctx = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
            if (
                amp
                and device.type == "cuda"
            )
            else nullcontext()
        )

        with ctx:
            feat = model(
                x
            )

        if feat.ndim != 2:
            raise ValueError(
                f"Expected [B,D] Mammo-FM features, "
                f"got {tuple(feat.shape)}"
            )

        # Normalize embeddings so dot product == cosine.
        feat = F.normalize(
            feat.float(),
            p=2,
            dim=-1,
        )

        features.append(
            feat.cpu()
        )

    return torch.cat(
        features,
        dim=0,
    )


# ============================================================
# Paths
# ============================================================

def build_paths(
    run_dir,
    df,
):
    src_paths = []
    recon_paths = []
    cf_paths = []
    gt_paths = []

    for _, row in df.iterrows():
        idx = int(
            row["output_idx"]
        )

        source_view = str(
            row["source_view"]
        )

        target_view = str(
            row["target_view"]
        )

        name = (
            f"{idx:06d}"
        )

        src_paths.append(
            run_dir
            / "inputs"
            / f"{name}_source_{source_view}.png"
        )

        recon_paths.append(
            run_dir
            / "reconstructions"
            / f"{name}_recon_{source_view}.png"
        )

        cf_paths.append(
            run_dir
            / "cfs"
            / f"{name}_cf_{target_view}.png"
        )

        gt_paths.append(
            run_dir
            / "ground_truth"
            / f"{name}_real_{target_view}.png"
        )

    return (
        src_paths,
        recon_paths,
        cf_paths,
        gt_paths,
    )


def check_paths(
    paths,
    name,
):
    missing = [
        p
        for p in paths
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing {name} files. "
            f"First missing: {missing[0]}"
        )


# ============================================================
# Similarity metrics
# ============================================================

def cosine(
    a,
    b,
):
    """
    Inputs are already L2 normalized.
    """
    return (
        a * b
    ).sum(
        dim=-1
    ).numpy()


def make_random_indices(
    df,
    seed,
):
    """
    Random target from another patient.

    Prefer:
        same laterality
        different patient

    This makes the random baseline harder and more meaningful.
    """
    rng = np.random.default_rng(
        seed
    )

    n = len(df)

    if n < 2:
        raise ValueError(
            "Need at least two samples "
            "for random baseline."
        )

    indices = np.arange(
        n
    )

    patient_col = None

    for name in [
        "empi_anon",
        "patient_id",
        "PatientID",
    ]:
        if name in df.columns:
            patient_col = name
            break

    laterality_col = None

    for name in [
        "laterality",
        "Laterality",
        "Laterality_norm",
    ]:
        if name in df.columns:
            laterality_col = name
            break

    patient = (
        df[
            patient_col
        ]
        .astype(str)
        .to_numpy()
        if patient_col is not None
        else None
    )

    laterality = (
        df[
            laterality_col
        ]
        .astype(str)
        .to_numpy()
        if laterality_col is not None
        else None
    )

    random_idx = np.empty(
        n,
        dtype=np.int64,
    )

    for i in range(n):
        mask = (
            indices != i
        )

        if patient is not None:
            mask &= (
                patient
                != patient[i]
            )

        if laterality is not None:
            matched = (
                mask
                & (
                    laterality
                    == laterality[i]
                )
            )

            if matched.any():
                mask = matched

        candidates = (
            indices[
                mask
            ]
        )

        if len(candidates) == 0:
            raise ValueError(
                f"No valid random target "
                f"available for sample {i}."
            )

        random_idx[i] = (
            rng.choice(
                candidates
            )
        )

    return random_idx


def paired_retrieval(
    f_query,
    f_gt,
):
    """
    For every query, rank all real target-view images.

    The correct paired GT is at the same row index.

    rank=1:
        correct paired target is nearest neighbour.
    """
    sim = (
        f_query
        @ f_gt.T
    )

    order = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    n = len(
        f_query
    )

    target = torch.arange(
        n,
        device=order.device,
    ).view(
        -1,
        1,
    )

    matches = (
        order
        == target
    )

    ranks = (
        matches
        .nonzero(
            as_tuple=False
        )[:, 1]
        + 1
    )

    return (
        ranks
        .cpu()
        .numpy()
    )


# ============================================================
# Statistics
# ============================================================

def bootstrap_stats(
    values,
    n_bootstrap=5000,
    seed=0,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot summarize empty array."
        )

    rng = np.random.default_rng(
        seed
    )

    boot_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        n_bootstrap
    ):
        boot_means[i] = (
            rng.choice(
                values,
                size=len(values),
                replace=True,
            )
            .mean()
        )

    return {
        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        "median":
            float(
                np.median(
                    values
                )
            ),

        "ci95_low":
            float(
                np.percentile(
                    boot_means,
                    2.5,
                )
            ),

        "ci95_high":
            float(
                np.percentile(
                    boot_means,
                    97.5,
                )
            ),
    }


# ============================================================
# Human-readable summary
# ============================================================

def save_text_summary(
    summary,
    metric_names,
    path,
):
    with open(
        path,
        "w",
    ) as f:

        f.write(
            "=" * 92
            + "\n"
        )

        f.write(
            "Mammo-FM Counterfactual Evaluation\n"
        )

        f.write(
            "=" * 92
            + "\n\n"
        )

        f.write(
            f"Encoder      : "
            f"{summary['encoder']}\n"
        )

        f.write(
            f"HF repo      : "
            f"{summary['hf_repo']}\n"
        )

        f.write(
            f"Checkpoint   : "
            f"{summary['hf_checkpoint']}\n"
        )

        f.write(
            f"Model        : "
            f"{summary['model_name']}\n"
        )

        f.write(
            f"Num samples  : "
            f"{summary['num_samples']}\n"
        )

        f.write(
            f"Feature dim  : "
            f"{summary['feature_dim']}\n"
        )

        prep = (
            summary[
                "preprocessing"
            ]
        )

        f.write(
            f"Input size   : "
            f"{prep['height']} x "
            f"{prep['width']}\n"
        )

        f.write(
            f"Mean / std   : "
            f"{prep['mean']} / "
            f"{prep['std']}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "-" * 92
            + "\n"
        )

        f.write(
            "Average metrics\n"
        )

        f.write(
            "-" * 92
            + "\n"
        )

        for metric in metric_names:
            stats = (
                summary[
                    "metrics"
                ][metric]
            )

            f.write(
                f"{metric:24s} "
                f"{stats['mean']:.4f} "
                f"± {stats['std']:.4f} "
                f"[95% CI "
                f"{stats['ci95_low']:.4f}, "
                f"{stats['ci95_high']:.4f}] "
                f"(median="
                f"{stats['median']:.4f})\n"
            )

        retrieval = (
            summary[
                "retrieval"
            ]
        )

        f.write(
            "\n"
        )

        f.write(
            "-" * 92
            + "\n"
        )

        f.write(
            "Paired retrieval\n"
        )

        f.write(
            "-" * 92
            + "\n"
        )

        f.write(
            f"CF -> paired GT Top-1       : "
            f"{retrieval['cf_to_gt_top1']:.4f}\n"
        )

        f.write(
            f"CF -> paired GT Top-5       : "
            f"{retrieval['cf_to_gt_top5']:.4f}\n"
        )

        f.write(
            f"CF -> paired GT median rank : "
            f"{retrieval['cf_to_gt_median_rank']:.1f}\n"
        )

        f.write(
            f"Src -> paired GT Top-1      : "
            f"{retrieval['src_to_gt_top1']:.4f}\n"
        )

        f.write(
            f"Src -> paired GT Top-5      : "
            f"{retrieval['src_to_gt_top5']:.4f}\n"
        )

        f.write(
            f"Src -> paired GT median rank: "
            f"{retrieval['src_to_gt_median_rank']:.1f}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "-" * 92
            + "\n"
        )

        f.write(
            "Metric interpretation\n"
        )

        f.write(
            "-" * 92
            + "\n"
        )

        f.write(
            "src_recon_cosine   : "
            "source vs same-condition reconstruction\n"
        )

        f.write(
            "src_cf_cosine      : "
            "source vs generated counterfactual\n"
        )

        f.write(
            "src_gt_cosine      : "
            "source vs real paired target view\n"
        )

        f.write(
            "cf_gt_cosine       : "
            "generated counterfactual vs "
            "real paired target\n"
        )

        f.write(
            "cf_random_cosine   : "
            "generated counterfactual vs "
            "random target\n"
        )

        f.write(
            "src_random_cosine  : "
            "source vs random target\n"
        )

        f.write(
            "cf_gt_advantage    : "
            "cf_gt_cosine - cf_random_cosine\n"
        )

        f.write(
            "src_gt_advantage   : "
            "src_gt_cosine - src_random_cosine\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "=" * 92
            + "\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired CC<->MLO counterfactuals "
            "using Mammo-FM ASU/Mayo image features."
        )
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
    )

    parser.add_argument(
        "--mean",
        type=float,
        default=DEFAULT_MEAN,
    )

    parser.add_argument(
        "--std",
        type=float,
        default=DEFAULT_STD,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--amp",
        type=int,
        default=1,
        choices=[
            0,
            1,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--save_features",
        type=int,
        default=0,
        choices=[
            0,
            1,
        ],
    )

    args = (
        parser.parse_args()
    )

    run_dir = Path(
        args.run_dir
    )

    samples_csv = (
        run_dir
        / "samples.csv"
    )

    if not samples_csv.exists():
        raise FileNotFoundError(
            samples_csv
        )

    df = pd.read_csv(
        samples_csv
    )

    required = {
        "output_idx",
        "source_view",
        "target_view",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"samples.csv missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) < 2:
        raise ValueError(
            "At least two samples are "
            "required for evaluation."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 80
    )

    print(
        "Mammo-FM counterfactual evaluation"
    )

    print(
        "=" * 80
    )

    print(
        "Run directory :",
        run_dir,
    )

    print(
        "Samples       :",
        len(df),
    )

    print(
        "Device        :",
        device,
    )

    print(
        "Input size    :",
        f"{args.height} x {args.width}",
    )

    print(
        "Mean / std    :",
        args.mean,
        args.std,
    )

    print(
        "Checkpoint    :",
        HF_CKPT,
    )

    print(
        "=" * 80
    )

    (
        src_paths,
        recon_paths,
        cf_paths,
        gt_paths,
    ) = build_paths(
        run_dir,
        df,
    )

    check_paths(
        src_paths,
        "source",
    )

    check_paths(
        recon_paths,
        "reconstruction",
    )

    check_paths(
        cf_paths,
        "counterfactual",
    )

    check_paths(
        gt_paths,
        "ground-truth",
    )

    # ========================================================
    # Model
    # ========================================================

    (
        model,
        ckpt_path,
        feature_dim,
    ) = load_mammo_fm(
        device=device,
        model_name=args.model_name,
        hf_cache_dir=args.hf_cache_dir,
    )

    extract_kwargs = dict(
        model=model,
        device=device,
        height=args.height,
        width=args.width,
        mean=args.mean,
        std=args.std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=bool(
            args.amp
        ),
    )

    # ========================================================
    # Features
    # ========================================================

    f_src = extract_features(
        paths=src_paths,
        desc="Source",
        **extract_kwargs,
    )

    f_recon = extract_features(
        paths=recon_paths,
        desc="Reconstruction",
        **extract_kwargs,
    )

    f_cf = extract_features(
        paths=cf_paths,
        desc="Counterfactual",
        **extract_kwargs,
    )

    f_gt = extract_features(
        paths=gt_paths,
        desc="Paired GT",
        **extract_kwargs,
    )

    print(
        "Feature shape:",
        tuple(
            f_src.shape
        ),
    )

    # ========================================================
    # Similarities
    # ========================================================

    src_recon = cosine(
        f_src,
        f_recon,
    )

    src_cf = cosine(
        f_src,
        f_cf,
    )

    src_gt = cosine(
        f_src,
        f_gt,
    )

    cf_gt = cosine(
        f_cf,
        f_gt,
    )

    random_idx = (
        make_random_indices(
            df,
            args.seed,
        )
    )

    f_random_gt = (
        f_gt[
            random_idx
        ]
    )

    cf_random = cosine(
        f_cf,
        f_random_gt,
    )

    src_random = cosine(
        f_src,
        f_random_gt,
    )

    cf_gt_advantage = (
        cf_gt
        - cf_random
    )

    src_gt_advantage = (
        src_gt
        - src_random
    )

    # ========================================================
    # Paired retrieval
    # ========================================================

    cf_gt_rank = (
        paired_retrieval(
            f_cf,
            f_gt,
        )
    )

    src_gt_rank = (
        paired_retrieval(
            f_src,
            f_gt,
        )
    )

    # ========================================================
    # Per-sample results
    # ========================================================

    results = (
        df.copy()
    )

    results[
        "src_recon_cosine"
    ] = src_recon

    results[
        "src_cf_cosine"
    ] = src_cf

    results[
        "src_gt_cosine"
    ] = src_gt

    results[
        "cf_gt_cosine"
    ] = cf_gt

    results[
        "cf_random_cosine"
    ] = cf_random

    results[
        "src_random_cosine"
    ] = src_random

    results[
        "cf_gt_advantage"
    ] = cf_gt_advantage

    results[
        "src_gt_advantage"
    ] = src_gt_advantage

    results[
        "cf_gt_rank"
    ] = cf_gt_rank

    results[
        "src_gt_rank"
    ] = src_gt_rank

    results[
        "random_target_output_idx"
    ] = (
        df.iloc[
            random_idx
        ][
            "output_idx"
        ]
        .astype(int)
        .to_numpy()
    )

    metrics_csv = (
        run_dir
        / "mammo_fm_metrics.csv"
    )

    results.to_csv(
        metrics_csv,
        index=False,
    )

    # ========================================================
    # Optional feature cache
    # ========================================================

    if args.save_features:
        torch.save(
            {
                "source":
                    f_src,

                "reconstruction":
                    f_recon,

                "counterfactual":
                    f_cf,

                "ground_truth":
                    f_gt,

                "output_idx":
                    torch.as_tensor(
                        df[
                            "output_idx"
                        ].to_numpy(),
                        dtype=torch.long,
                    ),
            },
            run_dir
            / "mammo_fm_features.pt",
        )

    # ========================================================
    # Summary
    # ========================================================

    metric_names = [
        "src_recon_cosine",
        "src_cf_cosine",
        "src_gt_cosine",
        "cf_gt_cosine",
        "cf_random_cosine",
        "src_random_cosine",
        "cf_gt_advantage",
        "src_gt_advantage",
    ]

    summary = {
        "encoder":
            "Mammo-FM ASU/Mayo EfficientNet-B5",

        "hf_repo":
            HF_REPO,

        "hf_checkpoint":
            HF_CKPT,

        "checkpoint_path":
            str(
                ckpt_path
            ),

        "model_name":
            args.model_name,

        "num_samples":
            int(
                len(df)
            ),

        "feature_dim":
            int(
                feature_dim
            ),

        "preprocessing": {
            "height":
                args.height,

            "width":
                args.width,

            "mean":
                args.mean,

            "std":
                args.std,

            "rgb_from_grayscale":
                True,
        },

        "metrics": {},

        "retrieval": {
            "cf_to_gt_top1":
                float(
                    np.mean(
                        cf_gt_rank
                        <= 1
                    )
                ),

            "cf_to_gt_top5":
                float(
                    np.mean(
                        cf_gt_rank
                        <= min(
                            5,
                            len(df),
                        )
                    )
                ),

            "cf_to_gt_median_rank":
                float(
                    np.median(
                        cf_gt_rank
                    )
                ),

            "src_to_gt_top1":
                float(
                    np.mean(
                        src_gt_rank
                        <= 1
                    )
                ),

            "src_to_gt_top5":
                float(
                    np.mean(
                        src_gt_rank
                        <= min(
                            5,
                            len(df),
                        )
                    )
                ),

            "src_to_gt_median_rank":
                float(
                    np.median(
                        src_gt_rank
                    )
                ),
        },
    }

    for i, metric in enumerate(
        metric_names
    ):
        summary[
            "metrics"
        ][metric] = bootstrap_stats(
            results[
                metric
            ].to_numpy(),
            n_bootstrap=(
                args.bootstrap_samples
            ),
            seed=(
                args.seed
                + i
            ),
        )

    # ========================================================
    # Save JSON summary
    # ========================================================

    summary_path = (
        run_dir
        / "mammo_fm_summary.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Save TXT summary
    # ========================================================

    summary_txt_path = (
        run_dir
        / "mammo_fm_summary.txt"
    )

    save_text_summary(
        summary=summary,
        metric_names=metric_names,
        path=summary_txt_path,
    )

    # ========================================================
    # Print summary
    # ========================================================

    print()

    print(
        "=" * 92
    )

    print(
        "Mammo-FM feature evaluation"
    )

    print(
        "=" * 92
    )

    for metric in metric_names:
        stats = (
            summary[
                "metrics"
            ][metric]
        )

        print(
            f"{metric:24s} "
            f"{stats['mean']:.4f} "
            f"± {stats['std']:.4f} "
            f"[95% CI "
            f"{stats['ci95_low']:.4f}, "
            f"{stats['ci95_high']:.4f}] "
            f"(median="
            f"{stats['median']:.4f})"
        )

    print(
        "-" * 92
    )

    print(
        f"CF -> paired GT Top-1       : "
        f"{summary['retrieval']['cf_to_gt_top1']:.4f}"
    )

    print(
        f"CF -> paired GT Top-5       : "
        f"{summary['retrieval']['cf_to_gt_top5']:.4f}"
    )

    print(
        f"CF -> paired GT median rank : "
        f"{summary['retrieval']['cf_to_gt_median_rank']:.1f}"
    )

    print(
        f"Src -> paired GT Top-1      : "
        f"{summary['retrieval']['src_to_gt_top1']:.4f}"
    )

    print(
        f"Src -> paired GT Top-5      : "
        f"{summary['retrieval']['src_to_gt_top5']:.4f}"
    )

    print(
        f"Src -> paired GT median rank: "
        f"{summary['retrieval']['src_to_gt_median_rank']:.1f}"
    )

    print(
        "=" * 92
    )

    print(
        "Metrics :",
        metrics_csv,
    )

    print(
        "JSON    :",
        summary_path,
    )

    print(
        "TXT     :",
        summary_txt_path,
    )

    if args.save_features:
        print(
            "Features:",
            run_dir
            / "mammo_fm_features.pt",
        )


if __name__ == "__main__":
    main()