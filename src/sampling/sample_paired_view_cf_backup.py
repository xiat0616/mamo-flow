#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import save_image
sys.path.append("/vol/biomedic3/tx1215/mamo-flow")

PROJECT_ROOT = Path("/vol/biomedic3/tx1215/mamo-flow")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_handle.embed import get_embed, DatasetConfig

from src.sampling.sample_flow import (
    apply_single_intervention,
    build_flow_model_from_ckpt_args,
    build_sampling_root,
    generate_from_inverted_noise,
    get_class_schema,
    invert_to_noise,
    maybe_apply_ema,
    preprocess_x_for_sampling,
)


def to_namespace(d: dict) -> argparse.Namespace:
    return argparse.Namespace(**d)


def move_sample_pa_to_device(sample, device):
    return {
        k: v.unsqueeze(0).to(device)
        for k, v in sample["pa"].items()
    }


def prepare_sample_x(sample, device):
    x = sample["x"].unsqueeze(0)
    return preprocess_x_for_sampling(x, device)


def to_vis(x: torch.Tensor) -> torch.Tensor:
    """
    Convert model image tensor [-1, 1] -> display tensor [0, 1].
    """
    if x.shape[1] > 3:
        raise ValueError(
            f"Cannot directly save tensor with {x.shape[1]} channels as a mammogram. "
            "This looks like latent-space sampling; decode with the VAE first."
        )
    return ((x.detach().float().clamp(-1, 1) + 1.0) / 2.0).cpu()


def save_visual(
    x_src,
    x_recon,
    x_cf,
    x_gt,
    source_view,
    target_view,
    save_path,
):
    src = to_vis(x_src)[0, 0].numpy()
    recon = to_vis(x_recon)[0, 0].numpy()
    cf = to_vis(x_cf)[0, 0].numpy()
    gt = to_vis(x_gt)[0, 0].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(src, cmap="gray")
    axes[0].set_title(f"Source {source_view}")

    axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstruction")

    axes[2].imshow(cf, cmap="gray")
    axes[2].set_title(f"Generated {target_view} CF")

    axes[3].imshow(gt, cmap="gray")
    axes[3].set_title(f"Real {target_view}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired CC<->MLO counterfactuals and save source/recon/CF/GT images."
    )

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument(
        "--pair_csv",
        type=str,
        default=None,
        help=(
            "Paired CC/MLO CSV. Default: "
            "<train_args.split_dir>/EMBED_same_breast_CC_MLO_pairs.csv"
        ),
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Optional override for split_dir stored in checkpoint.",
    )

    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--source_view",
        type=str,
        default="CC",
        choices=["CC", "MLO"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly choose paired cases instead of taking the first N.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_ema",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = use EMA weights if present; 0 = raw model weights.",
    )

    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_steps", type=int, default=None)

    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------------------------------------------------------
    # Load checkpoint / training args
    # ---------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = to_namespace(ckpt["args"])

    if args.split_dir is not None:
        train_args.split_dir = args.split_dir

    # ---------------------------------------------------------
    # Build exact EMBED datasets used by training
    # ---------------------------------------------------------
    datasets = get_embed(
        DatasetConfig(
            data_dir=train_args.data_dir,
            split_dir=train_args.split_dir,
            cache_dir=getattr(train_args, "cache_dir", None),
            parents=train_args.parents,
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            img_channels=train_args.img_channels,
            vae_ckpt=getattr(train_args, "vae_ckpt", None),
        )
    )

    # ---------------------------------------------------------
    # Load paired CSV
    # ---------------------------------------------------------
    pair_csv = (
        Path(args.pair_csv)
        if args.pair_csv is not None
        else Path(train_args.split_dir) / "EMBED_same_breast_CC_MLO_pairs.csv"
    )

    pairs_df = pd.read_csv(pair_csv)

    if args.split != "all":
        pairs_df = pairs_df.loc[pairs_df["split"] == args.split].copy()

    pairs_df = pairs_df.reset_index().rename(columns={"index": "pair_csv_index"})

    if len(pairs_df) == 0:
        raise ValueError(f"No paired cases found for split={args.split}")

    n = min(args.num_samples, len(pairs_df))

    if args.shuffle:
        pairs_df = pairs_df.sample(
            n=n,
            random_state=args.seed,
            replace=False,
        ).reset_index(drop=True)
    else:
        pairs_df = pairs_df.iloc[:n].reset_index(drop=True)

    print(f"Using {len(pairs_df)} paired cases")
    print(f"Pair CSV: {pair_csv}")

    # ---------------------------------------------------------
    # Build trained flow
    # ---------------------------------------------------------
    model = build_flow_model_from_ckpt_args(
        train_args=train_args,
        device=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    maybe_apply_ema(
        model=model,
        ckpt=ckpt,
        ema_rate=getattr(train_args, "ema_rate", 0.9999),
        use_ema=bool(args.use_ema),
    )

    model.eval()
    print("Model loaded.")

    class_schema = get_class_schema(train_args)

    source_view = args.source_view
    target_view = "MLO" if source_view == "CC" else "CC"

    # ---------------------------------------------------------
    # Output folders
    # ---------------------------------------------------------
    sampling_root = build_sampling_root(
        save_root=args.save_dir,
        ckpt_path=args.ckpt,
        train_args=train_args,
        ode_method=args.ode_method,
        ode_atol=args.ode_atol,
        ode_rtol=args.ode_rtol,
        ode_steps=args.ode_steps,
    )

    split_tag = args.split
    out_root = (
        sampling_root
        / "paired_view_cfs"
        / f"{source_view}_to_{target_view}"
        / split_tag
    )

    dirs = {
        "inputs": out_root / "inputs",
        "reconstructions": out_root / "reconstructions",
        "cfs": out_root / "cfs",
        "ground_truth": out_root / "ground_truth",
        "visuals": out_root / "visuals",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    meta = {
        "ckpt": args.ckpt,
        "pair_csv": str(pair_csv),
        "num_samples_requested": args.num_samples,
        "num_samples_used": len(pairs_df),
        "source_view": source_view,
        "target_view": target_view,
        "split": args.split,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "use_ema": bool(args.use_ema),
        "ode_method": args.ode_method,
        "ode_atol": args.ode_atol,
        "ode_rtol": args.ode_rtol,
        "ode_steps": args.ode_steps,
    }

    with open(out_root / "sampling_args.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------------------------------------------------
    # Generate paired counterfactuals
    # ---------------------------------------------------------
    records = []

    for output_idx, row in pairs_df.iterrows():
        split = row["split"]

        cc_sample = datasets[split][int(row["cc_cache_idx"])]
        mlo_sample = datasets[split][int(row["mlo_cache_idx"])]

        if source_view == "CC":
            source_sample = cc_sample
            gt_sample = mlo_sample
        else:
            source_sample = mlo_sample
            gt_sample = cc_sample

        x_src = prepare_sample_x(source_sample, device)
        x_gt = prepare_sample_x(gt_sample, device)

        pa_src = move_sample_pa_to_device(source_sample, device)

        # Flip VIEW only, using the exact intervention code from sample_flow.py.
        pa_cf = apply_single_intervention(
            pa=pa_src,
            pa_rand=None,
            do_key="view",
            do_mode="flip",
            class_schema=class_schema,
        )

        # Sanity check: source view must match requested direction.
        src_view_idx = int(pa_src["view"].argmax(dim=-1).item())
        expected_src_idx = 1 if source_view == "CC" else 0

        if src_view_idx != expected_src_idx:
            raise ValueError(
                f"Pair {output_idx}: source view mismatch. "
                f"Requested {source_view}, but pa_src gives class {src_view_idx}."
            )

        with torch.no_grad():
            # Source -> noise
            z = invert_to_noise(
                model=model,
                x=x_src,
                pa_src=pa_src,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )

            # Noise -> same-view reconstruction
            x_recon = generate_from_inverted_noise(
                model=model,
                noise=z,
                pa_cf=pa_src,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )

            # Noise -> opposite-view counterfactual
            x_cf = generate_from_inverted_noise(
                model=model,
                noise=z,
                pa_cf=pa_cf,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )

        name = f"{output_idx:06d}"

        save_image(
            to_vis(x_src)[0],
            dirs["inputs"] / f"{name}_source_{source_view}.png",
        )
        save_image(
            to_vis(x_recon)[0],
            dirs["reconstructions"] / f"{name}_recon_{source_view}.png",
        )
        save_image(
            to_vis(x_cf)[0],
            dirs["cfs"] / f"{name}_cf_{target_view}.png",
        )
        save_image(
            to_vis(x_gt)[0],
            dirs["ground_truth"] / f"{name}_real_{target_view}.png",
        )

        save_visual(
            x_src=x_src,
            x_recon=x_recon,
            x_cf=x_cf,
            x_gt=x_gt,
            source_view=source_view,
            target_view=target_view,
            save_path=dirs["visuals"] / f"{name}_comparison.png",
        )

        records.append(
            {
                "output_idx": output_idx,
                "pair_csv_index": int(row["pair_csv_index"]),
                "split": split,
                "empi_anon": row["empi_anon"],
                "acc_anon": row["acc_anon"],
                "laterality": row["laterality"],
                "source_view": source_view,
                "target_view": target_view,
                "cc_cache_idx": int(row["cc_cache_idx"]),
                "mlo_cache_idx": int(row["mlo_cache_idx"]),
                "cc_shortpath": cc_sample["shortpath"],
                "mlo_shortpath": mlo_sample["shortpath"],
            }
        )

        print(
            f"[{output_idx + 1}/{len(pairs_df)}] "
            f"{source_view}->{target_view} | "
            f"split={split} | "
            f"laterality={row['laterality']}"
        )

    pd.DataFrame(records).to_csv(
        out_root / "samples.csv",
        index=False,
    )

    print()
    print("Done.")
    print("Saved to:", out_root)


if __name__ == "__main__":
    main()