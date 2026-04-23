import argparse
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

DEBUG=False

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
    cache_dir: str | None = None
    parents: list[str] = field(default_factory=list)
    img_height: int = 512
    img_width: int = 384
    img_channels: int = 1
    vae_ckpt: str | None = None


DEFAULT_EMBED_ROOT = Path("/vol/biodata/data/Mammo/EMBED/")
DEFAULT_IMAGE_ROOT = DEFAULT_EMBED_ROOT / "pngs/1024x768"


class Metadata(TypedDict):
    age: float
    view: int
    density: int
    scanner: int
    cview: int


CLASS_SCHEMA: Metadata = {
    "age": 0.0,
    "view": 2,
    "density": 4,
    "scanner": 5,
    "cview": 2,
}


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
        thresh, connectivity=4
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


def validate_parents(parents: list[str] | tuple[str, ...] | None) -> list[str]:
    if parents is None or len(parents) == 0:
        raise ValueError(
            "parents must be provided explicitly, e.g. "
            "--parents age view density scanner cview"
        )

    allowed = list(get_type_hints(Metadata).keys())
    invalid = [p for p in parents if p not in allowed]
    if invalid:
        raise ValueError(f"Invalid parent(s): {invalid}. Allowed: {allowed}")

    seen = set()
    out = []
    for p in parents:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_split_csvs(split_dir: str | os.PathLike) -> dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)
    split_dfs: dict[str, pd.DataFrame] = {}

    for split in ["train", "valid", "test"]:
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        split_dfs[split] = pd.read_csv(path, low_memory=False).reset_index(drop=True)

    return split_dfs


def get_sample(
    root: str | os.PathLike | None,
    row: pd.Series,
    return_image: bool = True,
) -> tuple[np.ndarray, Metadata] | Metadata:
    metadata: Metadata = {k: row[k] for k in get_type_hints(Metadata)}
    if DEBUG:
        metadata["age"] = float(metadata["age"]) / 100.0

    metadata["density"] =3.0-float(metadata["density"]) # #NOTE Flip this for debugging.

    if not return_image:
        return metadata

    image_path = row["image_path"]
    if root is not None and (not os.path.isabs(str(image_path))):
        image_path = os.path.join(root, image_path)

    image = preprocess_breast(image_path)
    return image, metadata


class EMBED(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | os.PathLike | None,
        df: pd.DataFrame,
        split: str,
        transform: Callable | None = None,
        parents: list[str] | None = None,
    ):
        super().__init__()
        self.root = root
        self.df = df.reset_index(drop=True)
        self.split = split
        self.transform = transform
        self.parents = validate_parents(parents)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Tensor | dict[str, Tensor] | str]:
        while True:
            try:
                row = self.df.iloc[idx]
                image, metadata = get_sample(self.root, row, return_image=True)
                break
            except (OSError, ValueError, RuntimeError, FileNotFoundError):
                idx = random.randrange(len(self))

        if image.ndim < 3:
            image = image[None, ...]

        image = torch.from_numpy(image)
        if self.transform is not None:
            image = self.transform(image)

        pa: dict[str, Tensor] = {}
        for k in self.parents:
            spec = CLASS_SCHEMA.get(k)
            if isinstance(spec, int) and spec > 0:
                one_hot = torch.zeros(spec, dtype=torch.float32)
                one_hot[int(metadata[k])] = 1.0
                pa[k] = one_hot
            else:
                pa[k] = torch.as_tensor(metadata[k], dtype=torch.float32).unsqueeze(0)

        shortpath = str(row["shortpath"]) if "shortpath" in row.index else str(row["image_path"])

        return {
            "x": image,
            "pa": pa,
            "shortpath": shortpath,
        }


def get_embed(cfg: DatasetConfig) -> dict[str, EMBED]:
    if cfg.split_dir is None:
        raise ValueError(
            "cfg.split_dir must be provided. "
            "This datasets.py expects pre-generated train.csv/valid.csv/test.csv."
        )

    if cfg.img_channels != 1:
        raise ValueError(
            f"This no-cache version expects img_channels=1, got {cfg.img_channels}"
        )

    parents = validate_parents(cfg.parents)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((cfg.img_height, cfg.img_width)),
            transforms.ToTensor(),
        ]
    )

    split_dfs = load_split_csvs(cfg.split_dir)

    datasets = {
        split: EMBED(
            root=cfg.data_dir,
            df=split_dfs[split],
            split=split,
            transform=transform,
            parents=parents,
        )
        for split in ["train", "valid", "test"]
    }
    return datasets


def get_dataloaders(
    cfg: DataLoaderConfig, datasets: dict[str, EMBED]
) -> dict[str, torch.utils.data.DataLoader]:
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_dist:
        from torch.utils.data.distributed import DistributedSampler

    s = cfg.resume_step
    rank = torch.distributed.get_rank() if is_dist else 0

    dataloaders = {}
    for split in ["train", "valid", "test"]:
        is_train = split == "train"
        seed = int(cfg.seed + (7654321 * s if is_train else 0))
        sampler = None

        if is_dist:
            sampler = DistributedSampler(datasets[split], shuffle=is_train, seed=seed)

        g = torch.Generator()
        g.manual_seed(seed + rank)

        kwargs = dict(
            dataset=datasets[split],
            batch_size=cfg.bs,
            shuffle=(sampler is None) and is_train,
            drop_last=is_train,
            sampler=sampler,
            pin_memory=True,
            num_workers=cfg.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        if cfg.num_workers > 0:
            kwargs["prefetch_factor"] = cfg.prefetch_factor
            kwargs["persistent_workers"] = True

        dataloaders[split] = torch.utils.data.DataLoader(**kwargs)

    return dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--parents", type=str, nargs="+", required=True)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--img_width", type=int, default=384)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset_cfg = DatasetConfig(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        parents=args.parents,
        img_height=args.img_height,
        img_width=args.img_width,
        img_channels=args.img_channels,
    )

    loader_cfg = DataLoaderConfig(
        bs=args.bs,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )

    datasets = get_embed(dataset_cfg)
    dataloaders = get_dataloaders(loader_cfg, datasets)
    batch = next(iter(dataloaders["train"]))

    expected_shape = (args.bs, 1, args.img_height, args.img_width)
    assert batch["x"].shape == expected_shape, (
        f"Expected {expected_shape}, got {tuple(batch['x'].shape)}"
    )

    print("Dataset sanity check passed.")
    print(f"Train size: {len(datasets['train'])}")
    print(f"Valid size: {len(datasets['valid'])}")
    print(f"Test size : {len(datasets['test'])}")