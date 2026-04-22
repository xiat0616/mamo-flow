import argparse
import importlib
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
# image preprocessing: match your current image pipeline
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
                transforms.ToTensor(),  # [0, 1], shape (1,H,W)
            ]
        )
        self._validate_df()

    def _validate_df(self) -> None:
        required_cols = {"cache_idx", "image_path"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in split csv: {sorted(missing)}")

        cache_idx = self.df["cache_idx"].to_numpy()
        if not np.issubdtype(cache_idx.dtype, np.integer):
            raise ValueError("cache_idx must be integer typed")

        if len(np.unique(cache_idx)) != len(cache_idx):
            raise ValueError("cache_idx must be unique within each split csv")

        expected = np.arange(len(self.df), dtype=np.int64)
        actual = np.sort(cache_idx.astype(np.int64))
        if not np.array_equal(expected, actual):
            raise ValueError(
                "cache_idx must cover exactly 0..N-1 within each split csv"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        if self.data_dir is not None and (not os.path.isabs(str(image_path))):
            image_path = os.path.join(self.data_dir, image_path)

        image = preprocess_breast(image_path)
        image = self.transform(image)          # (1,H,W), float in [0,1]

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
# encoder wrappers
# ============================================================

def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class BaseEncoderWrapper:
    name = "base"
    default_file_prefix = "encoding_float32"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Flux2EncoderWrapper(BaseEncoderWrapper):
    name = "flux2"
    default_file_prefix = "flux2encoding_float32"

    def __init__(
        self,
        model_id: str,
        subfolder: str | None,
        device: str,
        dtype: torch.dtype,
        sample_posterior: bool = True,
    ):
        from diffusers.models import AutoencoderKLFlux2

        self.device = device
        self.dtype = dtype
        self.sample_posterior = sample_posterior
        self.model = AutoencoderKLFlux2.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # grayscale -> 3 channels for VAE
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            out = self.model.encode(x)
            if hasattr(out, "latent_dist"):
                z = out.latent_dist.sample() if self.sample_posterior else out.latent_dist.mode()
            elif hasattr(out, "latents"):
                z = out.latents
            else:
                raise ValueError("Unexpected FLUX encode output format.")
        return z.float()


class DiffusersKLEncoderWrapper(BaseEncoderWrapper):
    name = "diffusers_kl"
    default_file_prefix = "encoding_float32"

    def __init__(
        self,
        model_id: str,
        subfolder: str | None,
        device: str,
        dtype: torch.dtype,
        repeat_gray_to_three: bool = False,
        sample_posterior: bool = True,
        apply_scaling_factor: bool = False,
    ):
        from diffusers import AutoencoderKL

        self.device = device
        self.dtype = dtype
        self.repeat_gray_to_three = repeat_gray_to_three
        self.sample_posterior = sample_posterior
        self.apply_scaling_factor = apply_scaling_factor

        self.model = AutoencoderKL.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeat_gray_to_three and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            out = self.model.encode(x)
            if not hasattr(out, "latent_dist"):
                raise ValueError("Expected AutoencoderKL.encode(...) to return latent_dist.")
            z = out.latent_dist.sample() if self.sample_posterior else out.latent_dist.mode()

        if self.apply_scaling_factor:
            scaling_factor = getattr(self.model.config, "scaling_factor", None)
            if scaling_factor is None:
                raise ValueError("apply_scaling_factor=True but scaling_factor missing")
            z = z * scaling_factor

        return z.float()


class FactoryEncoderWrapper(BaseEncoderWrapper):
    name = "factory"
    default_file_prefix = "encoding_float32"

    def __init__(
        self,
        factory_spec: str,
        device: str,
        dtype: torch.dtype,
        factory_kwargs: dict[str, Any] | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {} if factory_kwargs is None else factory_kwargs

        module_name, func_name = factory_spec.split(":")
        module = importlib.import_module(module_name)
        factory = getattr(module, func_name)

        self.model = factory(device=device, dtype=dtype, **self.factory_kwargs)
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            if hasattr(self.model, "encode_to_latent"):
                z = self.model.encode_to_latent(x)
            else:
                z = self.model.encode(x)

            if hasattr(z, "latent_dist"):
                z = z.latent_dist.sample()
            elif hasattr(z, "latents"):
                z = z.latents
            elif not torch.is_tensor(z):
                raise ValueError(
                    "Factory encoder output must be a Tensor or expose latent_dist/latents"
                )

        return z.float()


def build_encoder(args: argparse.Namespace) -> BaseEncoderWrapper:
    dtype = resolve_dtype(args.dtype)

    if args.vae_name == "flux2":
        return Flux2EncoderWrapper(
            model_id=args.model_id,
            subfolder=args.subfolder,
            device=args.device,
            dtype=dtype,
            sample_posterior=bool(args.sample_posterior),
        )

    if args.vae_name == "diffusers_kl":
        return DiffusersKLEncoderWrapper(
            model_id=args.model_id,
            subfolder=args.subfolder,
            device=args.device,
            dtype=dtype,
            repeat_gray_to_three=bool(args.repeat_gray_to_three),
            sample_posterior=bool(args.sample_posterior),
            apply_scaling_factor=bool(args.apply_scaling_factor),
        )

    if args.vae_name == "factory":
        factory_kwargs = {}
        if args.factory_ckpt is not None:
            factory_kwargs["ckpt_path"] = args.factory_ckpt
        return FactoryEncoderWrapper(
            factory_spec=args.factory_spec,
            device=args.device,
            dtype=dtype,
            factory_kwargs=factory_kwargs,
        )

    raise ValueError(f"Unsupported vae_name: {args.vae_name}")


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
        # z: (B,C,H,W)
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


def write_split_latents(
    split: str,
    split_csv: str | os.PathLike,
    args: argparse.Namespace,
    encoder: BaseEncoderWrapper,
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
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = args.file_prefix or encoder.default_file_prefix
    out_path = out_dir / f"{file_prefix}_{split}.dat"

    if out_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"{out_path} already exists. Pass --overwrite 1 to replace it.")

    manifest_copy_path = out_dir / f"{split}_manifest.csv"
    dataset.df.to_csv(manifest_copy_path, index=False)

    memmap = None
    latent_shape = None
    stats = None

    for batch in tqdm(loader, desc=f"Encoding {split}", leave=True):
        x = batch["x"].to(args.device, non_blocking=True)
        cache_idx = batch["cache_idx"].cpu().numpy().astype(np.int64)

        # VAE input expected in [-1,1]
        x = x * 2.0 - 1.0

        z = encoder.encode(x)
        if z.ndim != 4:
            raise ValueError(f"Expected latents with shape (B,C,H,W), got {tuple(z.shape)}")

        z_cpu = z.detach().cpu().numpy().astype(np.float32, copy=False)
        bsz, c, h, w = z_cpu.shape

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--vae_name",
        type=str,
        required=True,
        choices=["flux2", "diffusers_kl", "factory"],
    )
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--subfolder", type=str, default=None)

    parser.add_argument("--factory_spec", type=str, default=None)
    parser.add_argument("--factory_ckpt", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=192)

    parser.add_argument("--sample_posterior", type=int, default=1)
    parser.add_argument("--repeat_gray_to_three", type=int, default=0)
    parser.add_argument("--apply_scaling_factor", type=int, default=0)

    parser.add_argument("--file_prefix", type=str, default=None)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    if args.vae_name in {"flux2", "diffusers_kl"} and args.model_id is None:
        raise ValueError("--model_id is required for flux2 and diffusers_kl")
    if args.vae_name == "factory" and args.factory_spec is None:
        raise ValueError("--factory_spec is required for vae_name=factory")

    split_csvs = load_split_csvs(args.split_dir)
    encoder = build_encoder(args)

    all_meta = {
        "vae_name": args.vae_name,
        "model_id": args.model_id,
        "subfolder": args.subfolder,
        "factory_spec": args.factory_spec,
        "factory_ckpt": args.factory_ckpt,
        "device": args.device,
        "dtype": args.dtype,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "sample_posterior": int(args.sample_posterior),
        "repeat_gray_to_three": int(args.repeat_gray_to_three),
        "apply_scaling_factor": int(args.apply_scaling_factor),
        "file_prefix": args.file_prefix or encoder.default_file_prefix,
        "splits": {},
    }

    for split, split_csv in split_csvs.items():
        meta = write_split_latents(
            split=split,
            split_csv=split_csv,
            args=args,
            encoder=encoder,
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