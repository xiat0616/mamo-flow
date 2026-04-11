import argparse
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, get_type_hints

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from torch import Tensor
from torchvision import transforms

sys.path.append("..")
from utils import seed_worker


DEFAULT_EMBED_ROOT = Path("/vol/biodata/data/Mammo/EMBED/")
DEFAULT_IMAGE_ROOT = DEFAULT_EMBED_ROOT / "pngs/1024x768"

DOMAIN_MAP = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

TISSUE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

MODELNAME_MAP = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 4,
    "Senographe Pristina": 1,
}


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
        # sometimes bug in reading images with cv2
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=4
    )

    # fallback: if thresholding finds no foreground, keep the full image
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


def get_target_hw(args: argparse.Namespace) -> tuple[int, int]:
    if hasattr(args, "img_height") and hasattr(args, "img_width"):
        return int(args.img_height), int(args.img_width)

    if hasattr(args, "input_size"):
        input_size = getattr(args, "input_size")
        if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            return int(input_size[0]), int(input_size[1])

    raise AttributeError(
        "Expected args.img_height and args.img_width, or args.input_size=(H, W)."
    )


def normalize_domain_arg(
    domain: int | str | list[int] | list[str] | tuple[int, ...] | None,
) -> list[int] | None:
    if domain in (None, "None"):
        return None

    if isinstance(domain, (list, tuple, set)):
        return [int(d) for d in domain]

    if isinstance(domain, str) and "," in domain:
        return [int(d.strip()) for d in domain.split(",")]

    return [int(domain)]


def normalize_model_arg(
    model: int | str | list[int] | list[str] | tuple[int, ...] | None,
) -> list[int] | None:
    if model in (None, "None"):
        return None

    if isinstance(model, (list, tuple, set)):
        return [int(m) for m in model]

    if isinstance(model, str) and "," in model:
        return [int(m.strip()) for m in model.split(",")]

    return [int(model)]


def validate_parents(parents: list[str] | tuple[str, ...] | None) -> list[str]:
    if parents is None or len(parents) == 0:
        raise ValueError(
            "args.parents must be provided explicitly, e.g. "
            "--parents age view density scanner cview"
        )

    allowed = list(get_type_hints(Metadata).keys())
    invalid = [p for p in parents if p not in allowed]
    if invalid:
        raise ValueError(f"Invalid parent(s): {invalid}. Allowed: {allowed}")

    # deduplicate while preserving order
    seen = set()
    out = []
    for p in parents:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def get_embed_df(
    csv_filepath: str,
    image_root: str | os.PathLike | None = None,
    exclude_cviews: bool = True,
    domain: int | str | list[int] | list[str] | tuple[int, ...] | None = None,
    model: int | str | list[int] | list[str] | tuple[int, ...] | None = None,
    hold_out_model_5: bool = True,
) -> pd.DataFrame:
    image_root = DEFAULT_IMAGE_ROOT if image_root is None else Path(image_root)
    df = pd.read_csv(csv_filepath, low_memory=False)

    df["shortpath"] = df["image_path"].astype(str)
    df["image_path"] = df["image_path"].apply(lambda x: str(image_root / str(x)))
    df["manufacturer_domain"] = df["Manufacturer"].map(DOMAIN_MAP)
    df["density"] = df["tissueden"].map(TISSUE_MAP)
    df["scanner"] = df["ManufacturerModelName"].map(MODELNAME_MAP)
    df["view"] = (df["ViewPosition"] != "MLO").astype(int)
    df["cview"] = (df["FinalImageType"] != "2D").astype(int)
    df["age"] = df["age_at_study"]

    df = df.dropna(
        subset=[
            "age",
            "density",
            "scanner",
            "view",
            "cview",
            "image_path",
            "empi_anon",
        ]
    ).copy()

    df["density"] = df["density"].astype(int)
    df["scanner"] = df["scanner"].astype(int)
    df["view"] = df["view"].astype(int)
    df["cview"] = df["cview"].astype(int)

    if exclude_cviews:
        df = df.loc[df["cview"] == 0].copy()

    if hold_out_model_5:
        df = df.loc[df["scanner"] != 5].copy()

    domains = normalize_domain_arg(domain)
    if domains is not None:
        df = df.loc[df["manufacturer_domain"].isin(domains)].copy()

        if DOMAIN_MAP["HOLOGIC, Inc."] in domains:
            hologic_mask = df["manufacturer_domain"] == DOMAIN_MAP["HOLOGIC, Inc."]
            df = df.loc[
                (~hologic_mask) | (df["ManufacturerModelName"] == "Selenia Dimensions")
            ].copy()

    models = normalize_model_arg(model)
    if models is not None:
        df = df.loc[df["scanner"].isin(models)].copy()

    return df


def split_df(
    df: pd.DataFrame,
    seed: int = 33,
    prop_train: float = 1.0,
    num_valid_patients: int = 600,
) -> dict[str, pd.DataFrame]:
    patient_ids = df["empi_anon"].unique()

    train_id, holdout_id = train_test_split(
        patient_ids,
        test_size=0.25,
        random_state=seed,
    )

    if prop_train < 1.0:
        train_id = np.sort(train_id)
        y = (
            df.loc[df["empi_anon"].isin(train_id)]
            .groupby("empi_anon")["scanner"]
            .unique()
            .apply(lambda x: x[0])
            .sort_index()
        )
        assert y.index[0] == train_id[0]
        train_id, _ = train_test_split(
            train_id,
            train_size=int(prop_train * train_id.shape[0]),
            stratify=y.values,
            random_state=seed,
        )

    n_valid = min(int(num_valid_patients), len(holdout_id))
    valid_id = holdout_id[:n_valid]
    test_id = holdout_id[n_valid:]

    return {
        "train": df.loc[df["empi_anon"].isin(train_id)].copy(),
        "valid": df.loc[df["empi_anon"].isin(valid_id)].copy(),
        "test": df.loc[df["empi_anon"].isin(test_id)].copy(),
    }


def get_sample(
    root: str | os.PathLike | None,
    row: pd.Series,
    return_image: bool = True,
) -> tuple[np.ndarray, Metadata] | Metadata:
    metadata: Metadata = {k: row[k] for k in get_type_hints(Metadata)}
    metadata["age"] = float(metadata["age"]) / 100.0

    if return_image:
        image_path = row["image_path"]
        if root is not None and (not os.path.isabs(str(image_path))):
            image_path = os.path.join(root, image_path)
        image = preprocess_breast(image_path)
        return image, metadata
    else:
        return metadata


class EMBED(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | os.PathLike | None,
        df: pd.DataFrame,
        split: str,
        transform: Callable | None = None,
        parents: list[str] | None = None,
        cache_root: str | None = None,
        image_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.cache_root = cache_root
        self.image_shape = image_shape

        if cache_root is not None:
            print(f"Using {split} memmap...")
            if cache_root.endswith("/flux2_vae"):
                file = f"flux2encoding_float32_{split}.dat"
                self.CHW, self.mm_dtype = (32, 64, 64), np.float32
            elif cache_root.endswith("/raddino_vae"):
                file = f"encoding_float32_{split}.dat"
                self.CHW, self.mm_dtype = (16, 64, 64), np.float32
            elif cache_root.endswith("/cache"):
                assert image_shape is not None
                file = f"cache_uint8_{split}.dat"
                self.CHW, self.mm_dtype = image_shape, np.uint8
            else:
                raise ValueError
            self.cache_root = os.path.join(cache_root, file)
            assert os.path.exists(self.cache_root)

        self.cache = None  # lazy
        self.parents = validate_parents(parents)
        self.df = df.reset_index(drop=True)

    def _maybe_get_cache(self) -> None:
        if (self.cache is None) and (self.cache_root is not None):
            self.cache = np.memmap(
                self.cache_root,
                mode="r",
                dtype=self.mm_dtype,
                shape=(len(self), *self.CHW),
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Tensor | dict[str, float | int] | str]:
        self._maybe_get_cache()  # lazy
        while True:
            try:  # NOTE: hack to handle corrupted images, fix later
                row = self.df.iloc[idx]
                if self.cache is not None:
                    image = np.array(self.cache[idx], copy=True)
                    metadata = get_sample(self.root, row, False)
                else:
                    image, metadata = get_sample(self.root, row)
                break
            except (OSError, ValueError, RuntimeError, FileNotFoundError):
                idx = random.randrange(len(self))

        if image.ndim < 3:
            image = image[None, ...]
        image = torch.from_numpy(image)

        if self.transform is not None:
            image = self.transform(image)

        pa = {k: metadata[k] for k in self.parents}

        return dict(
            x=image,
            pa=pa,
            shortpath=str(row["shortpath"]),
        )


def get_embed(args: argparse.Namespace) -> dict[str, EMBED]:
    input_ch = args.img_channels if hasattr(args, "img_channels") else args.in_channels
    parents = validate_parents(getattr(args, "parents", None))

    if (input_ch > 3) and (args.cache_dir is not None):  # is latent
        assert args.vae_ckpt is not None
        if args.vae_ckpt == "flux2":
            assert input_ch == 32
            mean, std = [-0.061467] * input_ch, [1.633637] * input_ch
        else:  # NOTE: raddino_vae
            assert input_ch == 16
            mean, std = [-0.507020] * input_ch, [3.663423] * input_ch
        transform = transforms.Normalize(mean=mean, std=std)
        image_shape = (input_ch, 64, 64)
    else:
        img_height, img_width = get_target_hw(args)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
            ]
        )
        image_shape = (1, img_height, img_width)

    df = get_embed_df(
        csv_filepath=args.csv_filepath,
        image_root=getattr(args, "data_dir", None),
        exclude_cviews=bool(getattr(args, "exclude_cviews", True)),
        domain=getattr(args, "domain", None),
        model=getattr(args, "model", None),
        hold_out_model_5=bool(getattr(args, "hold_out_model_5", True)),
    )

    split_dfs = split_df(
        df=df,
        seed=getattr(args, "split_seed", 33),
        prop_train=getattr(args, "prop_train", 1.0),
        num_valid_patients=getattr(args, "num_valid_patients", 600),
    )

    datasets = {
        k: EMBED(
            root=getattr(args, "data_dir", None),
            df=split_dfs[k],
            split=k,
            transform=transform,
            parents=parents,
            cache_root=getattr(args, "cache_dir", None),
            image_shape=image_shape,
        )
        for k in ["train", "valid", "test"]
    }
    return datasets


def get_dataloaders(
    args: argparse.Namespace, datasets: dict[str, EMBED]
) -> dict[str, torch.utils.data.DataLoader]:
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_dist:
        from torch.utils.data.distributed import DistributedSampler

    s = int(getattr(args, "resume_step", 0))
    rank = torch.distributed.get_rank() if is_dist else 0

    dataloaders = {}
    for k in ["train", "valid"]:  # NOTE: omitting test set for now
        is_train = k == "train"
        seed, sampler = int(args.seed + (7654321 * s if is_train else 0)), None
        if is_dist:
            sampler = DistributedSampler(datasets[k], shuffle=is_train, seed=seed)
        g = torch.Generator()
        g.manual_seed(seed + rank)
        kwargs = dict(
            dataset=datasets[k],
            batch_size=args.bs,
            shuffle=(sampler is None) and is_train,
            drop_last=is_train,
            sampler=sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.num_workers > 0:
            kwargs["prefetch_factor"] = args.prefetch_factor
            kwargs["persistent_workers"] = True

        dataloaders[k] = torch.utils.data.DataLoader(**kwargs)
    return dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--csv_filepath", type=str, default="joined_simple.csv")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--parents", type=str, nargs="+", required=True)
    parser.add_argument("--domain", type=str, nargs="+", default=None)
    parser.add_argument("--model", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_cviews", type=int, default=1)
    parser.add_argument("--hold_out_model_5", type=int, default=1)
    parser.add_argument("--prop_train", type=float, default=1.0)
    parser.add_argument("--num_valid_patients", type=int, default=600)
    parser.add_argument("--split_seed", type=int, default=33)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--img_width", type=int, default=384)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = get_embed(args)
    dataloaders = get_dataloaders(args, datasets)
    batch = next(iter(dataloaders["train"]))
    img_shape = (1, args.img_height, args.img_width)
    assert batch["x"].shape == (args.bs, *img_shape)