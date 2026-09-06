import argparse
import json
import os
import random
import sys

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, get_type_hints

import cv2
import numpy as np
import pandas as pd
import torch

from skimage import io
from torch import Tensor
from torchvision import transforms

sys.path.append("..")

from src.utils import seed_worker


DEBUG = False
# DEBUG = True


# ============================================================
# Config
# ============================================================

@dataclass
class DataLoaderConfig:
    bs: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 0
    resume_step: int = 0


@dataclass
class DatasetConfig:
    data_dir: str | None = None
    split_dir: str | None = None

    # If provided:
    # load precomputed latent memmaps instead of PNG images.
    cache_dir: str | None = None

    parents: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Image mode:
    #
    # img_channels = 1
    # img_height   = original flow image H
    # img_width    = original flow image W
    #
    # Example:
    #   1 x 512 x 384
    #
    # --------------------------------------------------------
    # Latent mode:
    #
    # these describe the FLOW INPUT latent shape.
    #
    # For FLUX.2 on 512x384:
    #
    #   32 x 64 x 48
    #
    # --------------------------------------------------------

    img_height: int = 512
    img_width: int = 384
    img_channels: int = 1

    # Kept for compatibility with training scripts.
    vae_ckpt: str | None = None

    normalize_age: bool = True

    # If True:
    # normalize cached latent using TRAIN latent statistics.
    normalize_latents: bool = True


DEFAULT_EMBED_ROOT = Path(
    "/vol/biodata/data/Mammo/EMBED/"
)

DEFAULT_IMAGE_ROOT = (
    DEFAULT_EMBED_ROOT
    / "pngs/1024x768"
)


# ============================================================
# Metadata
# ============================================================

class Metadata(TypedDict):
    age: float
    view: int
    density: int
    scanner: int
    cview: int


CLASS_SCHEMA: dict[str, int | None] = {
    "age": None,
    "view": 2,
    "density": 4,
    "scanner": 5,
    "cview": 2,
}


# ============================================================
# Latent cache specification
# ============================================================

@dataclass
class LatentCacheSpec:
    path: Path

    # latent shape of ONE sample:
    # (C, H, W)
    shape: tuple[int, int, int]

    dtype: np.dtype

    # TRAIN-set normalization statistics.
    mean: list[float]
    std: list[float]

    num_samples: int

    # Optional information from JSON
    encoder: str | None = None
    posterior: str | None = None


def is_latent_mode(
    cfg: DatasetConfig,
) -> bool:

    return cfg.cache_dir is not None


# ============================================================
# EMBED preprocessing
# ============================================================

def preprocess_breast(
    image_path: str | os.PathLike,
) -> np.ndarray:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        from skimage.util import img_as_ubyte

        image = io.imread(
            image_path
        )

        gray = img_as_ubyte(
            image.astype(
                np.uint16
            )
        )

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

    (
        nb_components,
        output,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        thresh,
        connectivity=4,
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
                    stats[
                        i,
                        cv2.CC_STAT_AREA,
                    ],
                )
                for i in range(
                    1,
                    nb_components,
                )
            ],
            key=lambda x: x[1],
        )

        mask = (
            output == max_label
        )

    image = gray.copy()

    image[
        ~mask
    ] = 0

    return image.astype(
        np.uint8
    )


# ============================================================
# Parent validation
# ============================================================

def validate_parents(
    parents: list[str]
    | tuple[str, ...]
    | None,
) -> list[str]:

    if (
        parents is None
        or len(parents) == 0
    ):
        raise ValueError(
            "parents must be provided explicitly, e.g. "
            "--parents age view density scanner cview"
        )

    allowed = list(
        get_type_hints(
            Metadata
        ).keys()
    )

    invalid = [
        p
        for p in parents
        if p not in allowed
    ]

    if invalid:
        raise ValueError(
            f"Invalid parent(s): {invalid}. "
            f"Allowed: {allowed}"
        )

    seen = set()
    out = []

    for p in parents:

        if p not in seen:

            seen.add(p)
            out.append(p)

    return out


# ============================================================
# Split CSV loading
# ============================================================

def load_split_csvs(
    split_dir: str | os.PathLike,
) -> dict[str, pd.DataFrame]:

    split_dir = Path(
        split_dir
    )

    split_dfs = {}

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
                f"Missing split CSV: {path}"
            )

        split_dfs[
            split
        ] = pd.read_csv(
            path,
            low_memory=False,
        ).reset_index(
            drop=True
        )

    return split_dfs


# ============================================================
# cache_idx validation
# ============================================================

def validate_cache_idx(
    df: pd.DataFrame,
    split: str,
) -> None:

    if "cache_idx" not in df.columns:

        raise ValueError(
            f"Latent mode requires 'cache_idx' "
            f"in {split}.csv."
        )

    cache_idx = (
        df["cache_idx"]
        .to_numpy()
    )

    # Allow CSV parsing to produce e.g. int64.
    try:
        cache_idx_int = (
            cache_idx.astype(
                np.int64
            )
        )
    except Exception as e:
        raise ValueError(
            f"{split}.csv cache_idx "
            "must contain integer values."
        ) from e

    # Ensure e.g. 1.5 didn't silently become 1.
    if not np.array_equal(
        cache_idx,
        cache_idx_int,
    ):
        raise ValueError(
            f"{split}.csv cache_idx "
            "must contain integer values."
        )

    if (
        len(np.unique(cache_idx_int))
        != len(cache_idx_int)
    ):
        raise ValueError(
            f"{split}.csv cache_idx "
            "must be unique."
        )

    expected = np.arange(
        len(df),
        dtype=np.int64,
    )

    actual = np.sort(
        cache_idx_int
    )

    if not np.array_equal(
        expected,
        actual,
    ):
        raise ValueError(
            f"{split}.csv cache_idx must cover "
            f"exactly 0..N-1. "
            f"Got min="
            f"{actual.min() if len(actual) else None}, "
            f"max="
            f"{actual.max() if len(actual) else None}, "
            f"N={len(df)}."
        )


# ============================================================
# Latent cache metadata
# ============================================================

def load_latent_cache_meta(
    cache_dir: str | os.PathLike,
) -> dict:

    cache_dir = Path(
        cache_dir
    )

    meta_path = (
        cache_dir
        / "latent_cache_meta.json"
    )

    if not meta_path.exists():

        raise FileNotFoundError(
            f"Missing latent cache metadata: "
            f"{meta_path}. "
            f"Run write_latent_cache.py first."
        )

    with open(
        meta_path,
        "r",
    ) as f:

        meta = json.load(
            f
        )

    return meta


# ============================================================
# Latent cache spec
# ============================================================

def load_latent_cache_spec(
    cache_dir: str | os.PathLike,
    split: str,
    stats_split: str = "train",
) -> LatentCacheSpec:
    """
    Read the metadata produced by write_latent_cache.py.

    Expected structure:

    {
        "encoder": "...",
        "posterior": "mode",
        "cache_dtype": "float32",

        "splits": {
            "train": {
                "num_samples": ...,
                "latent_shape": [32,64,48],
                "dtype": "float32",
                "file": "...train.dat",
                "per_channel_mean": [...],
                "per_channel_std": [...]
            },

            ...
        }
    }

    Important:
        normalization statistics always come from
        stats_split="train".
    """

    cache_dir = Path(
        cache_dir
    )

    meta = load_latent_cache_meta(
        cache_dir
    )

    # --------------------------------------------------------
    # Validate splits
    # --------------------------------------------------------

    if "splits" not in meta:

        raise ValueError(
            "latent_cache_meta.json "
            "does not contain 'splits'."
        )

    splits = meta[
        "splits"
    ]

    if split not in splits:

        raise ValueError(
            f"Split {split!r} not found "
            f"in latent cache metadata. "
            f"Available: {list(splits.keys())}"
        )

    if stats_split not in splits:

        raise ValueError(
            f"Stats split {stats_split!r} "
            f"not found in latent cache metadata."
        )

    split_meta = splits[
        split
    ]

    stats_meta = splits[
        stats_split
    ]

    # --------------------------------------------------------
    # Cache file
    # --------------------------------------------------------

    if "file" not in split_meta:

        raise ValueError(
            f"Missing 'file' in metadata "
            f"for split={split}."
        )

    saved_path = Path(
        split_meta["file"]
    )

    # First try the exact saved path.
    if saved_path.exists():

        path = saved_path

    else:

        # If cache directory was moved,
        # try same filename inside current cache_dir.
        moved_path = (
            cache_dir
            / saved_path.name
        )

        if moved_path.exists():

            path = moved_path

        else:

            raise FileNotFoundError(
                f"Missing latent cache for "
                f"split={split}.\n"
                f"Tried:\n"
                f"  {saved_path}\n"
                f"  {moved_path}"
            )

    # --------------------------------------------------------
    # Latent shape
    # --------------------------------------------------------

    shape_raw = split_meta.get(
        "latent_shape"
    )

    if shape_raw is None:

        raise ValueError(
            f"Missing latent_shape "
            f"for split={split}."
        )

    shape = tuple(
        int(x)
        for x in shape_raw
    )

    if len(shape) != 3:

        raise ValueError(
            f"Expected latent_shape=[C,H,W], "
            f"got {shape}."
        )

    c, h, w = shape

    # --------------------------------------------------------
    # dtype
    # --------------------------------------------------------

    dtype_string = split_meta.get(
        "dtype",
        meta.get(
            "cache_dtype",
            "float32",
        ),
    )

    dtype = np.dtype(
        dtype_string
    )

    # --------------------------------------------------------
    # Normalization statistics
    #
    # IMPORTANT:
    # always from training split
    # --------------------------------------------------------

    mean = stats_meta.get(
        "per_channel_mean"
    )

    std = stats_meta.get(
        "per_channel_std"
    )

    if mean is None or std is None:

        raise ValueError(
            f"Missing per_channel_mean / "
            f"per_channel_std in "
            f"stats split={stats_split}."
        )

    if len(mean) != c:

        raise ValueError(
            f"Mean/channel mismatch: "
            f"C={c}, len(mean)={len(mean)}"
        )

    if len(std) != c:

        raise ValueError(
            f"Std/channel mismatch: "
            f"C={c}, len(std)={len(std)}"
        )

    # Avoid divide-by-zero.
    if any(
        float(s) <= 0
        for s in std
    ):

        raise ValueError(
            "Latent std contains "
            "non-positive values."
        )

    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    num_samples = int(
        split_meta.get(
            "num_samples",
            -1,
        )
    )

    if num_samples <= 0:

        raise ValueError(
            f"Invalid num_samples "
            f"for split={split}: "
            f"{num_samples}"
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return LatentCacheSpec(
        path=path,
        shape=(
            c,
            h,
            w,
        ),
        dtype=dtype,
        mean=[
            float(x)
            for x in mean
        ],
        std=[
            float(x)
            for x in std
        ],
        num_samples=num_samples,
        encoder=meta.get(
            "encoder"
        ),
        posterior=meta.get(
            "posterior"
        ),
    )


# ============================================================
# Metadata
# ============================================================

def get_metadata(
    row: pd.Series,
    normalize_age: bool = True,
) -> Metadata:

    metadata: Metadata = {
        k: row[k]
        for k in get_type_hints(
            Metadata
        )
    }

    if normalize_age:

        metadata[
            "age"
        ] = (
            float(
                metadata["age"]
            )
            / 100.0
        )

    else:

        metadata[
            "age"
        ] = float(
            metadata["age"]
        )

    if DEBUG:

        metadata[
            "density"
        ] = (
            3.0
            - float(
                metadata["density"]
            )
        )

    return metadata


# ============================================================
# Image loading
# ============================================================

def load_image(
    root: str | os.PathLike | None,
    row: pd.Series,
) -> np.ndarray:

    image_path = row[
        "image_path"
    ]

    if (
        root is not None
        and not os.path.isabs(
            str(image_path)
        )
    ):

        image_path = os.path.join(
            root,
            image_path,
        )

    return preprocess_breast(
        image_path
    )


def get_sample(
    root: str | os.PathLike | None,
    row: pd.Series,
    return_image: bool = True,
    normalize_age: bool = True,
) -> (
    tuple[np.ndarray, Metadata]
    | Metadata
):

    metadata = get_metadata(
        row,
        normalize_age=normalize_age,
    )

    if not return_image:

        return metadata

    image = load_image(
        root,
        row,
    )

    return (
        image,
        metadata,
    )


# ============================================================
# Parent encoding
# ============================================================

def encode_parent_metadata(
    metadata: Metadata,
    parents: list[str],
) -> dict[str, Tensor]:

    pa = {}

    for k in parents:

        spec = CLASS_SCHEMA.get(
            k
        )

        if (
            isinstance(spec, int)
            and spec > 0
        ):

            value = int(
                metadata[k]
            )

            if not (
                0 <= value < spec
            ):

                raise ValueError(
                    f"Invalid categorical value "
                    f"for {k}: {value}. "
                    f"Expected [0, {spec - 1}]."
                )

            one_hot = torch.zeros(
                spec,
                dtype=torch.float32,
            )

            one_hot[
                value
            ] = 1.0

            pa[
                k
            ] = one_hot

        else:

            pa[
                k
            ] = torch.as_tensor(
                metadata[k],
                dtype=torch.float32,
            ).unsqueeze(
                0
            )

    return pa


# ============================================================
# EMBED Dataset
# ============================================================

class EMBED(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        root: str
        | os.PathLike
        | None,
        df: pd.DataFrame,
        split: str,
        transform: Callable
        | None = None,
        parents: list[str]
        | None = None,
        cache_dir: str
        | os.PathLike
        | None = None,
        normalize_age: bool = True,
    ):

        super().__init__()

        self.root = root

        self.df = df.reset_index(
            drop=True
        )

        self.split = split
        self.transform = transform

        self.parents = validate_parents(
            parents
        )

        self.normalize_age = (
            normalize_age
        )

        # Lazy memmap
        self.cache = None

        self.cache_spec: (
            LatentCacheSpec
            | None
        ) = None

        # ----------------------------------------------------
        # Latent mode
        # ----------------------------------------------------

        if cache_dir is not None:

            validate_cache_idx(
                self.df,
                split,
            )

            self.cache_spec = (
                load_latent_cache_spec(
                    cache_dir,
                    split,
                    stats_split="train",
                )
            )

            if (
                self.cache_spec.num_samples
                != len(self.df)
            ):

                raise ValueError(
                    f"Latent cache sample count "
                    f"mismatch for split={split}: "
                    f"cache="
                    f"{self.cache_spec.num_samples}, "
                    f"CSV={len(self.df)}"
                )

            c, h, w = (
                self.cache_spec.shape
            )

            print(
                f"Using {split} latent memmap:\n"
                f"  file  = "
                f"{self.cache_spec.path}\n"
                f"  shape = "
                f"({len(self.df)}, "
                f"{c}, {h}, {w})\n"
                f"  dtype = "
                f"{self.cache_spec.dtype}"
            )


    def _maybe_get_cache(
        self,
    ) -> None:

        if (
            self.cache is None
            and self.cache_spec
            is not None
        ):

            self.cache = np.memmap(
                self.cache_spec.path,
                mode="r",
                dtype=(
                    self.cache_spec.dtype
                ),
                shape=(
                    len(self),
                    *self.cache_spec.shape,
                ),
            )


    def __len__(
        self,
    ) -> int:

        return len(
            self.df
        )


    # ========================================================
    # Image mode
    # ========================================================

    def _load_image_item(
        self,
        idx: int,
    ) -> tuple[
        np.ndarray,
        Metadata,
        pd.Series,
    ]:

        while True:

            try:

                row = self.df.iloc[
                    idx
                ]

                image, metadata = (
                    get_sample(
                        self.root,
                        row,
                        return_image=True,
                        normalize_age=(
                            self.normalize_age
                        ),
                    )
                )

                return (
                    image,
                    metadata,
                    row,
                )

            except (
                OSError,
                RuntimeError,
                FileNotFoundError,
            ):

                idx = random.randrange(
                    len(self)
                )


    # ========================================================
    # Latent mode
    # ========================================================

    def _load_latent_item(
        self,
        idx: int,
    ) -> tuple[
        np.ndarray,
        Metadata,
        pd.Series,
    ]:

        self._maybe_get_cache()

        assert (
            self.cache is not None
        )

        row = self.df.iloc[
            idx
        ]

        cache_idx = int(
            row[
                "cache_idx"
            ]
        )

        latent = np.array(
            self.cache[
                cache_idx
            ],
            copy=True,
        )

        metadata = get_sample(
            self.root,
            row,
            return_image=False,
            normalize_age=(
                self.normalize_age
            ),
        )

        return (
            latent,
            metadata,
            row,
        )


    # ========================================================
    # __getitem__
    # ========================================================

    def __getitem__(
        self,
        idx: int,
    ) -> dict:

        # ----------------------------------------------------
        # Latent
        # ----------------------------------------------------

        if self.cache_spec is not None:

            x_np, metadata, row = (
                self._load_latent_item(
                    idx
                )
            )

            if x_np.ndim < 3:

                x_np = x_np[
                    None,
                    ...
                ]

            x = torch.from_numpy(
                x_np
            ).float()

            # e.g.
            #
            # z_norm =
            # (z - train_mean)
            # / train_std

            if self.transform is not None:

                x = self.transform(
                    x
                )

            # Latents are NOT bounded to [0,1].
            # Only require finite values.

            if not torch.isfinite(
                x
            ).all():

                raise ValueError(
                    f"Latent contains NaN/Inf "
                    f"for idx={idx}"
                )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        else:

            x_np, metadata, row = (
                self._load_image_item(
                    idx
                )
            )

            if self.transform is not None:

                x = self.transform(
                    x_np
                )

            else:

                if x_np.ndim < 3:

                    x_np = x_np[
                        None,
                        ...
                    ]

                x = (
                    torch.from_numpy(
                        x_np
                    )
                    .float()
                    .div(255.0)
                )

            # Images should still be [0,1].

            assert (
                x.max() <= 1.0
                and x.min() >= 0.0
            ), (
                "Expected image values "
                f"in [0,1], got "
                f"[{x.min().item()}, "
                f"{x.max().item()}]"
            )

        # ----------------------------------------------------
        # Parents
        # ----------------------------------------------------

        pa = encode_parent_metadata(
            metadata,
            self.parents,
        )

        shortpath = (
            str(
                row["shortpath"]
            )
            if "shortpath"
            in row.index
            else str(
                row["image_path"]
            )
        )

        out = {
            "x": x,
            "pa": pa,
            "shortpath": shortpath,
        }

        if "cache_idx" in row.index:

            out[
                "cache_idx"
            ] = torch.as_tensor(
                int(
                    row[
                        "cache_idx"
                    ]
                ),
                dtype=torch.long,
            )

        return out


# ============================================================
# Transforms
# ============================================================

def build_image_transform(
    cfg: DatasetConfig,
) -> Callable:

    return transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Resize(
                (
                    cfg.img_height,
                    cfg.img_width,
                ),
                antialias=True,
            ),
        ]
    )


def build_latent_transform(
    cfg: DatasetConfig,
) -> Callable | None:

    if cfg.cache_dir is None:

        raise ValueError(
            "build_latent_transform "
            "called with cache_dir=None."
        )

    # Use TRAIN cache statistics
    # even for valid/test.

    train_spec = (
        load_latent_cache_spec(
            cfg.cache_dir,
            "train",
            stats_split="train",
        )
    )

    c, h, w = (
        train_spec.shape
    )

    expected_shape = (
        cfg.img_channels,
        cfg.img_height,
        cfg.img_width,
    )

    actual_shape = (
        c,
        h,
        w,
    )

    if (
        expected_shape
        != actual_shape
    ):

        raise ValueError(
            "Latent shape mismatch.\n"
            f"Model/config expects: "
            f"{expected_shape}\n"
            f"Cache contains: "
            f"{actual_shape}\n\n"
            "For FLUX.2 latent cache "
            "generated from 512x384 images, "
            "you probably want:\n"
            "--img_channels 32 "
            "--img_height 64 "
            "--img_width 48"
        )

    if not cfg.normalize_latents:

        return None

    return transforms.Normalize(
        mean=train_spec.mean,
        std=train_spec.std,
    )


# ============================================================
# Build datasets
# ============================================================

def get_embed(
    cfg: DatasetConfig,
) -> dict[str, EMBED]:

    if cfg.split_dir is None:

        raise ValueError(
            "cfg.split_dir "
            "must be provided."
        )

    parents = validate_parents(
        cfg.parents
    )

    split_dfs = load_split_csvs(
        cfg.split_dir
    )

    latent_mode = (
        is_latent_mode(
            cfg
        )
    )

    # --------------------------------------------------------
    # Latent mode
    # --------------------------------------------------------

    if latent_mode:

        transform = (
            build_latent_transform(
                cfg
            )
        )

    # --------------------------------------------------------
    # Image mode
    # --------------------------------------------------------

    else:

        if cfg.img_channels != 1:

            raise ValueError(
                "Image mode expects "
                f"img_channels=1, "
                f"got {cfg.img_channels}."
            )

        transform = (
            build_image_transform(
                cfg
            )
        )

    datasets = {
        split: EMBED(
            root=cfg.data_dir,
            df=split_dfs[
                split
            ],
            split=split,
            transform=transform,
            parents=parents,
            cache_dir=(
                cfg.cache_dir
                if latent_mode
                else None
            ),
            normalize_age=(
                cfg.normalize_age
            ),
        )
        for split in [
            "train",
            "valid",
            "test",
        ]
    }

    return datasets


# ============================================================
# DataLoaders
# ============================================================

def get_dataloaders(
    cfg: DataLoaderConfig,
    datasets: dict[
        str,
        EMBED,
    ],
) -> dict[
    str,
    torch.utils.data.DataLoader,
]:

    is_dist = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    if is_dist:

        from torch.utils.data.distributed import (
            DistributedSampler
        )

    s = cfg.resume_step

    rank = (
        torch.distributed.get_rank()
        if is_dist
        else 0
    )

    dataloaders = {}

    for split in [
        "train",
        "valid",
        "test",
    ]:

        is_train = (
            split == "train"
        )

        seed = int(
            cfg.seed
            + (
                7654321 * s
                if is_train
                else 0
            )
        )

        sampler = None

        if is_dist:

            sampler = (
                DistributedSampler(
                    datasets[
                        split
                    ],
                    shuffle=is_train,
                    seed=seed,
                )
            )

        g = torch.Generator()

        g.manual_seed(
            seed + rank
        )

        kwargs = dict(
            dataset=datasets[
                split
            ],
            batch_size=cfg.bs,
            shuffle=(
                sampler is None
            )
            and is_train,
            drop_last=is_train,
            sampler=sampler,
            pin_memory=True,
            num_workers=(
                cfg.num_workers
            ),
            worker_init_fn=(
                seed_worker
            ),
            generator=g,
        )

        if cfg.num_workers > 0:

            kwargs[
                "prefetch_factor"
            ] = (
                cfg.prefetch_factor
            )

            kwargs[
                "persistent_workers"
            ] = True

        dataloaders[
            split
        ] = (
            torch.utils.data.DataLoader(
                **kwargs
            )
        )

    return dataloaders


# ============================================================
# Standalone test
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(
            DEFAULT_IMAGE_ROOT
        ),
    )

    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
    )

    # If provided:
    # latent mode.
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--parents",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--img_height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--img_width",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--img_channels",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--normalize_age",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--normalize_latents",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    dataset_cfg = DatasetConfig(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        cache_dir=args.cache_dir,
        parents=args.parents,
        img_height=args.img_height,
        img_width=args.img_width,
        img_channels=args.img_channels,
        vae_ckpt=args.vae_ckpt,
        normalize_age=bool(
            args.normalize_age
        ),
        normalize_latents=bool(
            args.normalize_latents
        ),
    )

    loader_cfg = (
        DataLoaderConfig(
            bs=args.bs,
            num_workers=(
                args.num_workers
            ),
            prefetch_factor=(
                args.prefetch_factor
            ),
            seed=args.seed,
            resume_step=(
                args.resume_step
            ),
        )
    )

    datasets = get_embed(
        dataset_cfg
    )

    dataloaders = (
        get_dataloaders(
            loader_cfg,
            datasets,
        )
    )

    batch = next(
        iter(
            dataloaders[
                "train"
            ]
        )
    )

    expected_shape = (
        args.bs,
        args.img_channels,
        args.img_height,
        args.img_width,
    )

    assert (
        batch["x"].shape
        == expected_shape
    ), (
        f"Expected {expected_shape}, "
        f"got "
        f"{tuple(batch['x'].shape)}"
    )

    print(
        "\nDataset sanity check passed."
    )

    print(
        "mode       :",
        (
            "latent"
            if args.cache_dir
            is not None
            else "image"
        ),
    )

    print(
        f"DEBUG      : {DEBUG}"
    )

    print(
        "x shape    :",
        tuple(
            batch[
                "x"
            ].shape
        ),
    )

    print(
        "x dtype    :",
        batch[
            "x"
        ].dtype,
    )

    print(
        "x min/max  : "
        f"{batch['x'].min().item():.4f} / "
        f"{batch['x'].max().item():.4f}"
    )

    print(
        "x mean/std : "
        f"{batch['x'].mean().item():.4f} / "
        f"{batch['x'].std().item():.4f}"
    )

    print(
        "pa keys    :",
        list(
            batch[
                "pa"
            ].keys()
        ),
    )

    print(
        "Train size :",
        len(
            datasets[
                "train"
            ]
        ),
    )

    print(
        "Valid size :",
        len(
            datasets[
                "valid"
            ]
        ),
    )

    print(
        "Test size  :",
        len(
            datasets[
                "test"
            ]
        ),
    )


if __name__ == "__main__":
    main()