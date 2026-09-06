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

from src.utils import get_pretrained_flux2vae


# ============================================================
# EMBED image preprocessing
# ============================================================

def preprocess_breast(image_path: str | os.PathLike) -> np.ndarray:
    """
    Load mammogram, convert to grayscale, and keep the largest
    connected foreground component.

    Returns:
        image: uint8 array with shape [H, W]
    """

    image = cv2.imread(str(image_path))

    if image is None:
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))

    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    thresh = cv2.threshold(
        gray,
        5,
        255,
        cv2.THRESH_BINARY,
    )[1]

    nb_components, output, stats, _ = (
        cv2.connectedComponentsWithStats(
            thresh,
            connectivity=4,
        )
    )

    if nb_components <= 1:
        mask = np.ones_like(
            gray,
            dtype=bool,
        )

    else:
        max_label, _ = max(
            [
                (
                    i,
                    stats[i, cv2.CC_STAT_AREA],
                )
                for i in range(
                    1,
                    nb_components,
                )
            ],
            key=lambda x: x[1],
        )

        mask = output == max_label

    image = gray.copy()
    image[~mask] = 0

    return image.astype(np.uint8)


# ============================================================
# Dataset
# ============================================================

class EmbedImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | os.PathLike,
        data_dir: str | os.PathLike,
        img_height: int,
        img_width: int,
    ):
        self.csv_path = Path(csv_path)
        self.data_dir = Path(data_dir)

        self.df = pd.read_csv(
            self.csv_path,
            low_memory=False,
        ).reset_index(drop=True)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),

                transforms.Resize(
                    (
                        img_height,
                        img_width,
                    ),
                    antialias=True,
                ),

                transforms.ToTensor(),
            ]
        )

        self._validate_dataframe()

    def _validate_dataframe(self):

        required_cols = {
            "cache_idx",
            "image_path",
        }

        missing = (
            required_cols
            - set(self.df.columns)
        )

        if missing:
            raise ValueError(
                f"{self.csv_path} is missing "
                f"required columns: {sorted(missing)}"
            )

        if self.df["cache_idx"].isna().any():
            raise ValueError(
                f"{self.csv_path}: "
                "cache_idx contains NaN."
            )

        cache_idx = (
            self.df["cache_idx"]
            .to_numpy()
            .astype(np.int64)
        )

        if len(np.unique(cache_idx)) != len(cache_idx):
            raise ValueError(
                f"{self.csv_path}: "
                "cache_idx contains duplicates."
            )

        expected = np.arange(
            len(self.df),
            dtype=np.int64,
        )

        actual = np.sort(
            cache_idx
        )

        if not np.array_equal(
            expected,
            actual,
        ):
            raise ValueError(
                f"{self.csv_path}: "
                "cache_idx must cover exactly 0..N-1."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["image_path"]

        if not os.path.isabs(
            str(image_path)
        ):
            image_path = (
                self.data_dir
                / str(image_path)
            )

        image = preprocess_breast(
            image_path
        )

        x = self.transform(
            image
        )

        return {
            "x": x,
            "cache_idx": int(
                row["cache_idx"]
            ),
        }


# ============================================================
# Running latent statistics
# ============================================================

class RunningLatentStats:
    """
    Calculate global and per-channel latent statistics.

    z is expected to have shape:
        [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
    ):
        self.channels = channels

        self.count_per_channel = 0

        self.sum = torch.zeros(
            channels,
            dtype=torch.float64,
        )

        self.sumsq = torch.zeros(
            channels,
            dtype=torch.float64,
        )

    def update(
        self,
        z: torch.Tensor,
    ):

        z = (
            z.detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        self.sum += z.sum(
            dim=(0, 2, 3)
        )

        self.sumsq += (
            z * z
        ).sum(
            dim=(0, 2, 3)
        )

        self.count_per_channel += int(
            z.shape[0]
            * z.shape[2]
            * z.shape[3]
        )

    def finalize(self):

        channel_mean = (
            self.sum
            / self.count_per_channel
        )

        channel_var = (
            self.sumsq
            / self.count_per_channel
            - channel_mean ** 2
        )

        channel_std = torch.sqrt(
            torch.clamp(
                channel_var,
                min=1e-12,
            )
        )

        # Global statistics over:
        # N × C × H × W
        total_count = (
            self.count_per_channel
            * self.channels
        )

        global_mean = (
            self.sum.sum()
            / total_count
        )

        global_var = (
            self.sumsq.sum()
            / total_count
            - global_mean ** 2
        )

        global_std = torch.sqrt(
            torch.clamp(
                global_var,
                min=1e-12,
            )
        )

        return {
            "global_mean":
                float(global_mean.item()),

            "global_std":
                float(global_std.item()),

            "per_channel_mean":
                channel_mean.tolist(),

            "per_channel_std":
                channel_std.tolist(),
        }


# ============================================================
# Split helper
# ============================================================

def get_split_csvs(
    split_dir: str | os.PathLike,
):

    split_dir = Path(
        split_dir
    )

    split_csvs = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:

        path = (
            split_dir
            / f"{split}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing CSV: {path}"
            )

        split_csvs[split] = path

    return split_csvs


# ============================================================
# Save manifest
# ============================================================

def save_manifest(
    dataset: EmbedImageDataset,
    out_dir: Path,
    split: str,
):

    # Sort by cache_idx so that:
    #
    # manifest.iloc[i]
    #
    # corresponds to:
    #
    # memmap[i]

    manifest = (
        dataset.df
        .sort_values("cache_idx")
        .reset_index(drop=True)
    )

    manifest_path = (
        out_dir
        / f"{split}_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    return manifest_path


# ============================================================
# Write one split
# ============================================================

@torch.inference_mode()
def write_split_latents(
    split: str,
    csv_path: str | os.PathLike,
    args: argparse.Namespace,
    vae: torch.nn.Module,
):

    print(
        f"\n======================================"
    )
    print(
        f"Encoding split: {split}"
    )
    print(
        f"======================================"
    )

    dataset = EmbedImageDataset(
        csv_path=csv_path,
        data_dir=args.data_dir,
        img_height=args.img_height,
        img_width=args.img_width,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Split {split} contains no samples."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith(
            "cuda"
        ),
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / f"flux2encoding_float32_{split}.dat"
    )

    if (
        out_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{out_path} already exists.\n"
            "Use --overwrite if you want "
            "to replace it."
        )

    manifest_path = save_manifest(
        dataset=dataset,
        out_dir=out_dir,
        split=split,
    )

    # FLUX.2 is currently loaded as bfloat16
    vae_dtype = next(
        vae.parameters()
    ).dtype

    memmap = None
    latent_shape = None
    stats = None

    for batch in tqdm(
        loader,
        desc=f"Encoding {split}",
    ):

        x = batch["x"]

        cache_idx = (
            batch["cache_idx"]
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        # --------------------------------
        # Image:
        #
        # [0,1] -> [-1,1]
        # --------------------------------

        x = (
            x * 2.0
            - 1.0
        )

        # Match the VAE dtype.
        #
        # e.g.
        # float32 -> bfloat16

        x = x.to(
            device=args.device,
            dtype=vae_dtype,
            non_blocking=True,
        )

        # --------------------------------
        # FLUX encoding
        # --------------------------------
        #
        # Your get_pretrained_flux2vae()
        # already:
        #
        # 1. repeats grayscale -> RGB
        # 2. calls FLUX.2 encoder
        # 3. returns latent_dist.mode()
        #
        # Therefore x remains:
        #
        # [B,1,H,W]
        #
        # here.
        # --------------------------------

        z = vae.encode(
            x
        )

        if z.ndim != 4:
            raise ValueError(
                "Expected z with shape "
                f"[B,C,H,W], got {z.shape}"
            )

        # --------------------------------
        # Save latents as float32
        # --------------------------------

        z = z.float()

        # --------------------------------
        # Initialise memmap from first batch
        # --------------------------------

        if memmap is None:

            _, C, H, W = z.shape

            latent_shape = (
                int(C),
                int(H),
                int(W),
            )

            print(
                f"\n{split} latent shape:"
                f" {latent_shape}"
            )

            memmap = np.memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    len(dataset),
                    C,
                    H,
                    W,
                ),
            )

            stats = RunningLatentStats(
                C
            )

        # --------------------------------
        # GPU Tensor -> NumPy
        # --------------------------------

        z_np = (
            z.cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # --------------------------------
        # Write based on cache_idx
        # --------------------------------
        #
        # e.g.
        #
        # cache_idx = [100,101,102,...]
        #
        # writes those samples to:
        #
        # memmap[100]
        # memmap[101]
        # memmap[102]
        #
        # --------------------------------

        memmap[
            cache_idx
        ] = z_np

        # Statistics calculated on raw latents
        stats.update(
            z
        )

    if memmap is None:
        raise RuntimeError(
            f"No samples written for {split}."
        )

    # Ensure everything is physically
    # written to disk.
    memmap.flush()

    del memmap

    split_stats = (
        stats.finalize()
    )

    result = {
        "split":
            split,

        "num_samples":
            len(dataset),

        "latent_shape":
            list(latent_shape),

        "dtype":
            "float32",

        "file":
            str(out_path),

        "manifest":
            str(manifest_path),

        **split_stats,
    }

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help=(
            "Directory containing "
            "train.csv, valid.csv, test.csv"
        ),
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Root directory containing "
            "the EMBED PNG images"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Directory to save latent caches"
        ),
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    # --------------------------------------------------------
    # Image size before FLUX VAE
    # --------------------------------------------------------

    parser.add_argument(
        "--img_height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--img_width",
        type=int,
        default=192,
    )

    # --------------------------------------------------------
    # Existing files
    # --------------------------------------------------------

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # Load CSV paths
    # ========================================================

    split_csvs = get_split_csvs(
        args.split_dir
    )

    # ========================================================
    # Load FLUX.2 VAE
    # ========================================================

    print(
        "\nLoading FLUX.2 VAE..."
    )

    vae = (
        get_pretrained_flux2vae()
    )

    vae = vae.to(
        args.device
    )

    vae.eval()
    vae.requires_grad_(False)

    vae_dtype = next(
        vae.parameters()
    ).dtype

    print(
        "VAE device:",
        next(
            vae.parameters()
        ).device,
    )

    print(
        "VAE dtype:",
        vae_dtype,
    )

    print(
        "Input image size:",
        (
            args.img_height,
            args.img_width,
        ),
    )

    # ========================================================
    # Metadata
    # ========================================================

    all_meta = {
        "encoder":
            "black-forest-labs/FLUX.2-dev",

        "vae":
            "AutoencoderKLFlux2",

        "posterior":
            "mode",

        "input_channels":
            1,

        "input_height":
            args.img_height,

        "input_width":
            args.img_width,

        "cache_dtype":
            "float32",

        "splits":
            {},
    }

    # Save the registered constants from
    # get_pretrained_flux2vae().
    #
    # We do NOT apply these during encoding.

    if hasattr(
        vae,
        "mean",
    ):
        all_meta[
            "vae_registered_mean"
        ] = float(
            vae.mean
            .detach()
            .float()
            .cpu()
            .item()
        )

    if hasattr(
        vae,
        "std",
    ):
        all_meta[
            "vae_registered_std"
        ] = float(
            vae.std
            .detach()
            .float()
            .cpu()
            .item()
        )

    # ========================================================
    # Encode train / valid / test
    # ========================================================

    for split in [
        "train",
        "valid",
        "test",
    ]:

        meta = write_split_latents(
            split=split,
            csv_path=split_csvs[
                split
            ],
            args=args,
            vae=vae,
        )

        all_meta[
            "splits"
        ][split] = meta

        print(
            f"\n[{split}]"
        )

        print(
            "samples:",
            meta[
                "num_samples"
            ],
        )

        print(
            "latent shape:",
            meta[
                "latent_shape"
            ],
        )

        print(
            "global mean:",
            meta[
                "global_mean"
            ],
        )

        print(
            "global std:",
            meta[
                "global_std"
            ],
        )

        print(
            "saved:",
            meta[
                "file"
            ],
        )

    # ========================================================
    # Save recommended TRAIN statistics
    # ========================================================

    train_meta = (
        all_meta[
            "splits"
        ]["train"]
    )

    all_meta[
        "train_latent_statistics"
    ] = {
        "global_mean":
            train_meta[
                "global_mean"
            ],

        "global_std":
            train_meta[
                "global_std"
            ],

        "per_channel_mean":
            train_meta[
                "per_channel_mean"
            ],

        "per_channel_std":
            train_meta[
                "per_channel_std"
            ],
    }

    # ========================================================
    # Save JSON metadata
    # ========================================================

    meta_path = (
        Path(
            args.out_dir
        )
        / "latent_cache_meta.json"
    )

    with open(
        meta_path,
        "w",
    ) as f:

        json.dump(
            all_meta,
            f,
            indent=2,
        )

    print(
        "\n======================================"
    )

    print(
        "Finished latent caching."
    )

    print(
        "Metadata saved to:",
        meta_path,
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()