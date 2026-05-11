import sys
sys.path.append("/vol/biomedic3/tx1215/mamo-flow")

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# ============================================================
# image preprocessing: match EMBED image pipeline
# ============================================================

def preprocess_breast(image_path: str | os.PathLike) -> np.ndarray:
    image = cv2.imread(str(image_path))

    if image is None:
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        thresh,
        connectivity=4,
    )

    if nb_components <= 1:
        mask = np.ones_like(gray, dtype=bool)
    else:
        max_label, _ = max(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
            key=lambda x: x[1],
        )
        mask = output == max_label

    image = gray.copy()
    image[~mask] = 0

    return image.astype(np.uint8)


# ============================================================
# split csv dataset
# ============================================================

class SplitImageDataset(Dataset):
    def __init__(
        self,
        split_csv: str | os.PathLike,
        data_dir: str | os.PathLike | None,
        img_height: int,
        img_width: int,
    ):
        self.df = pd.read_csv(split_csv, low_memory=False).reset_index(drop=True)
        self.data_dir = data_dir

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),  # [0, 1], shape [1,H,W]
            ]
        )

        self._validate_df()

    def _validate_df(self) -> None:
        required_cols = {"cache_idx", "image_path"}
        missing = required_cols - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in split csv: {sorted(missing)}"
            )

        cache_idx = self.df["cache_idx"].to_numpy()

        if not np.issubdtype(cache_idx.dtype, np.integer):
            raise ValueError("cache_idx must be integer typed.")

        if len(np.unique(cache_idx)) != len(cache_idx):
            raise ValueError("cache_idx must be unique within each split csv.")

        expected = np.arange(len(self.df), dtype=np.int64)
        actual = np.sort(cache_idx.astype(np.int64))

        if not np.array_equal(expected, actual):
            raise ValueError(
                "cache_idx must cover exactly 0..N-1 within each split csv."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = row["image_path"]

        if self.data_dir is not None and not os.path.isabs(str(image_path)):
            image_path = os.path.join(self.data_dir, image_path)

        image = preprocess_breast(image_path)
        image = self.transform(image)

        shortpath = (
            str(row["shortpath"])
            if "shortpath" in row.index
            else str(row["image_path"])
        )

        return {
            "x": image,
            "cache_idx": int(row["cache_idx"]),
            "shortpath": shortpath,
        }


# ============================================================
# VAE loading / encoding
# ============================================================

def load_vae(args: argparse.Namespace) -> torch.nn.Module:
    if args.vae_ckpt == "flux2":
        from utils import get_pretrained_flux2vae

        vae = get_pretrained_flux2vae()

        if not bool(args.sample_posterior):
            print(
                "\nWARNING: utils.get_pretrained_flux2vae() currently monkey-patches "
                "encode() to use latent_dist.sample(). Therefore --sample_posterior 0 "
                "is ignored for flux2 unless you change that utility function."
            )

    elif args.vae_ckpt is not None and os.path.isfile(args.vae_ckpt):
        from src.models.vae import get_pretrained_vae

        vae = get_pretrained_vae(args.vae_ckpt)

    else:
        raise ValueError(
            "--vae_ckpt must be either 'flux2' or a valid local VAE checkpoint path."
        )

    vae.to(args.device)
    vae.eval()
    vae.requires_grad_(False)

    return vae


@torch.inference_mode()
def encode_with_vae(
    vae: torch.nn.Module,
    x: torch.Tensor,
    sample_posterior: bool,
) -> torch.Tensor:
    """
    x is expected to be in [-1, 1].

    Supports:
        1. utils.get_pretrained_flux2vae()
           - vae.encode(x) returns Tensor directly due monkey patch.

        2. vae.VAE from get_pretrained_vae()
           - if beta > 0, encode(x, sample=True) returns (z, loc, scale).
           - if beta == 0, encode(x, sample=False) returns deterministic z.
    """
    beta = float(getattr(vae, "beta", 0.0))

    if beta > 0:
        out = vae.encode(x, sample=True)

        if isinstance(out, tuple):
            z, loc = out[0], out[1]
            return z.float() if sample_posterior else loc.float()

        if torch.is_tensor(out):
            return out.float()

        raise ValueError("Unexpected stochastic VAE encode output.")

    try:
        out = vae.encode(x, sample=False)
    except TypeError:
        out = vae.encode(x)

    if isinstance(out, tuple):
        return out[0].float()

    if torch.is_tensor(out):
        return out.float()

    if hasattr(out, "latent_dist"):
        z = out.latent_dist.sample() if sample_posterior else out.latent_dist.mode()
        return z.float()

    if hasattr(out, "latents"):
        return out.latents.float()

    raise ValueError("Unexpected VAE encode output.")


# ============================================================
# stats helper
# ============================================================

class RunningChannelStats:
    def __init__(self, channels: int):
        self.channels = channels
        self.count = 0
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.sumsq = torch.zeros(channels, dtype=torch.float64)

    def update(self, z: torch.Tensor) -> None:
        # z: [B,C,H,W]
        z = z.detach().to(torch.float64).cpu()

        self.sum += z.sum(dim=(0, 2, 3))
        self.sumsq += (z * z).sum(dim=(0, 2, 3))
        self.count += int(z.shape[0] * z.shape[2] * z.shape[3])

    def finalize(self) -> dict[str, Any]:
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-12))

        return {
            "count": int(self.count),
            "per_channel_mean": mean.tolist(),
            "per_channel_std": std.tolist(),
            "global_mean": float(mean.mean().item()),
            "global_std": float(std.mean().item()),
        }


# ============================================================
# writing logic
# ============================================================

def load_split_csvs(split_dir: str | os.PathLike) -> dict[str, Path]:
    split_dir = Path(split_dir)
    out = {}

    for split in ["train", "valid", "test"]:
        path = split_dir / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")

        out[split] = path

    return out


def default_file_prefix(args: argparse.Namespace) -> str:
    if args.vae_ckpt == "flux2":
        return "flux2encoding_float32"
    return "encoding_float32"


def write_split_latents(
    split: str,
    split_csv: str | os.PathLike,
    args: argparse.Namespace,
    vae: torch.nn.Module,
) -> dict[str, Any]:
    dataset = SplitImageDataset(
        split_csv=split_csv,
        data_dir=args.data_dir,
        img_height=args.img_height,
        img_width=args.img_width,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = args.file_prefix or default_file_prefix(args)
    out_path = out_dir / f"{file_prefix}_{split}.dat"

    if out_path.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"{out_path} already exists. Pass --overwrite 1 to replace it."
        )

    manifest_copy_path = out_dir / f"{split}_manifest.csv"
    dataset.df.to_csv(manifest_copy_path, index=False)

    memmap = None
    latent_shape = None
    stats = None

    for batch in tqdm(loader, desc=f"Encoding {split}", leave=True):
        x = batch["x"].to(args.device, non_blocking=True)
        cache_idx = batch["cache_idx"].cpu().numpy().astype(np.int64)

        # VAE input expected in [-1, 1].
        x = x * 2.0 - 1.0

        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            z = encode_with_vae(
                vae=vae,
                x=x,
                sample_posterior=bool(args.sample_posterior),
            )

        if z.ndim != 4:
            raise ValueError(
                f"Expected latents with shape [B,C,H,W], got {tuple(z.shape)}"
            )

        z_cpu = z.detach().cpu().numpy().astype(np.float32, copy=False)
        _, c, h, w = z_cpu.shape

        if memmap is None:
            latent_shape = (c, h, w)

            memmap = np.memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), c, h, w),
            )

            stats = RunningChannelStats(c)

        memmap[cache_idx] = z_cpu
        stats.update(z)

    if memmap is None or stats is None or latent_shape is None:
        raise RuntimeError(f"No samples were written for split={split}")

    memmap.flush()
    split_stats = stats.finalize()

    return {
        "split": split,
        "num_samples": int(len(dataset)),
        "latent_shape": list(latent_shape),
        "file": str(out_path),
        "manifest_copy": str(manifest_copy_path),
        **split_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # Follow your training code convention:
    #   --vae_ckpt flux2
    #   --vae_ckpt /path/to/custom_vae_checkpoint.pt
    parser.add_argument("--vae_ckpt", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # These are image H/W before VAE encoding.
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=192)

    # For local VAE:
    #   1 -> sampled z
    #   0 -> deterministic loc if beta > 0
    #
    # For flux2:
    #   your current utils.get_pretrained_flux2vae() always samples.
    parser.add_argument("--sample_posterior", type=int, default=1)

    parser.add_argument("--file_prefix", type=str, default=None)
    parser.add_argument("--overwrite", type=int, default=0)

    args = parser.parse_args()

    split_csvs = load_split_csvs(args.split_dir)
    vae = load_vae(args)

    file_prefix = args.file_prefix or default_file_prefix(args)

    all_meta = {
        "vae_ckpt": args.vae_ckpt,
        "device": args.device,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "sample_posterior": int(args.sample_posterior),
        "file_prefix": file_prefix,
        "splits": {},
    }

    if hasattr(vae, "mean"):
        all_meta["vae_mean"] = float(vae.mean.detach().cpu().item())

    if hasattr(vae, "std"):
        all_meta["vae_std"] = float(vae.std.detach().cpu().item())

    for split, split_csv in split_csvs.items():
        meta = write_split_latents(
            split=split,
            split_csv=split_csv,
            args=args,
            vae=vae,
        )

        all_meta["splits"][split] = meta

        print(f"[{split}] wrote {meta['num_samples']} samples to {meta['file']}")
        print(
            f"[{split}] latent_shape={tuple(meta['latent_shape'])}, "
            f"global_mean={meta['global_mean']:.6f}, "
            f"global_std={meta['global_std']:.6f}"
        )

    meta_path = Path(args.out_dir) / "latent_cache_meta.json"

    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nSaved metadata to {meta_path}")


if __name__ == "__main__":
    main()