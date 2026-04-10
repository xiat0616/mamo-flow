import argparse
import os
import random
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


def normalize_label_key(label: str) -> str:
    mapping = {
        "tissueden": "density",
        "density": "density",
        "SimpleModelLabel": "scanner",
        "scanner": "scanner",
        "ViewLabel": "view",
        "view": "view",
    }
    if label not in mapping:
        raise ValueError(f"Unsupported label: {label}")
    return mapping[label]


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


def get_embed_df(
    csv_filepath: str,
    image_root: str | os.PathLike | None = None,
    exclude_cviews: bool = True,
    domain: int | str | None = None,
    model: int | str | None = None,
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

    if domain not in (None, "None"):
        domain = int(domain)
        df = df.loc[df["manufacturer_domain"] == domain].copy()
        if domain == DOMAIN_MAP["HOLOGIC, Inc."]:
            df = df.loc[df["ManufacturerModelName"] == "Selenia Dimensions"].copy()

    if model not in (None, "None"):
        df = df.loc[df["scanner"] == int(model)].copy()

    return df


def split_df(
    df: pd.DataFrame,
    label_key: str,
    seed: int = 33,
    prop_train: float = 1.0,
    num_valid_patients: int = 600,
) -> dict[str, pd.DataFrame]:
    patient_ids = df["empi_anon"].unique()

    if label_key == "density":
        y = (
            df.groupby("empi_anon")[label_key]
            .unique()
            .apply(lambda x: x[0])
            .reindex(patient_ids)
            .values
        )
        train_id, holdout_id = train_test_split(
            patient_ids,
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )
    else:
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
        label_key: str = "density",
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.cache_root = cache_root
        self.image_shape = image_shape
        self.label_key = label_key

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
        self.parents = None if parents is None else set(parents)  # set for O(1)
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

    def __getitem__(self, idx: int) -> dict[str, Tensor | Metadata | str]:
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

        if self.parents is not None:
            metadata = {k: v for k, v in metadata.items() if k in self.parents}

        return dict(
            x=image,
            pa=metadata,
            y=torch.tensor(int(row[self.label_key])).long(),
            shortpath=str(row["shortpath"]),
            scanner_int=torch.tensor(int(row["scanner"])).long(),
        )


def get_embed(args: argparse.Namespace) -> dict[str, EMBED]:
    input_ch = args.img_channels if hasattr(args, "img_channels") else args.in_channels

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

    label_key = normalize_label_key(getattr(args, "label", "tissueden"))

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
        label_key=label_key,
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
            parents=getattr(args, "parents", list(CLASS_SCHEMA)),
            cache_root=getattr(args, "cache_dir", None),
            image_shape=image_shape,
            label_key=label_key,
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
    parser.add_argument("--label", type=str, default="tissueden")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
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