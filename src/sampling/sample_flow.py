import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from src.data_handle.datasets import (
    CLASS_SCHEMA,
    get_embed,
    get_dataloaders,
    DataLoaderConfig,
    DatasetConfig,
)
from src.models.embedder import (
    GlobalCondEmbedder,
    PerAttrCondEmbedder,
    CondEmbedderConfig,
)
from src.models.unet import UNet
from src.flows.flow import Flow, BlockConfig, UNetConfig
from src.utils import ModelEMA, seed_all


def infer_parent_dims_from_batch(
    pa: dict[str, torch.Tensor],
    parents: list[str],
) -> dict[str, int]:
    parent_dims: dict[str, int] = {}
    for k in parents:
        if k not in pa:
            raise KeyError(f"Parent '{k}' not found in batch['pa']")
        v = pa[k]
        parent_dims[k] = 1 if v.ndim == 1 else int(v.shape[1])
    return parent_dims


def to_namespace(d: dict) -> argparse.Namespace:
    return argparse.Namespace(**d)


def select_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def build_dataloaders_from_train_args(
    train_args: argparse.Namespace,
    batch_size: int,
) -> dict[str, torch.utils.data.DataLoader]:
    datasets = get_embed(
        DatasetConfig(
            data_dir=train_args.data_dir,
            csv_filepath=train_args.csv_filepath,
            cache_dir=getattr(train_args, "cache_dir", None),
            parents=train_args.parents,
            domain=getattr(train_args, "domain", None),
            scanner_model=getattr(train_args, "scanner_model", None),
            exclude_cviews=getattr(train_args, "exclude_cviews", 1),
            hold_out_model_5=getattr(train_args, "hold_out_model_5", 1),
            prop_train=getattr(train_args, "prop_train", 1.0),
            valid_frac=getattr(train_args, "valid_frac", 0.125),
            test_frac=getattr(train_args, "test_frac", 0.125),
            split_seed=getattr(train_args, "split_seed", 33),
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            img_channels=train_args.img_channels,
            vae_ckpt=getattr(train_args, "vae_ckpt", None),
        )
    )

    dataloaders = get_dataloaders(
        DataLoaderConfig(
            bs=batch_size,
            num_workers=getattr(train_args, "num_workers", 4),
            prefetch_factor=getattr(train_args, "prefetch_factor", 2),
            seed=getattr(train_args, "seed", 0),
            resume_step=0,
        ),
        datasets,
    )
    return dataloaders


def build_flow_model_from_ckpt_args(
    train_args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    amp_dtype = select_amp_dtype(device)

    dataloaders = build_dataloaders_from_train_args(train_args, batch_size=2)
    sample_batch = next(iter(dataloaders["train"]))
    parent_dims = infer_parent_dims_from_batch(sample_batch["pa"], train_args.parents)

    unet_cfg = UNetConfig(
        img_height=train_args.img_height,
        img_width=train_args.img_width,
        img_channels=train_args.img_channels,
        cond_embed_dim=train_args.cond_embed_dim,
        model_channels=train_args.model_channels,
        channel_mult=tuple(train_args.channel_mult),
        channel_mult_time=getattr(train_args, "channel_mult_time", None),
        channel_mult_emb=getattr(train_args, "channel_mult_emb", None),
        num_blocks=train_args.num_blocks,
        attn_resolutions=tuple(train_args.attn_resolutions),
        label_balance=train_args.label_balance,
        concat_balance=train_args.concat_balance,
    )

    block_cfg = BlockConfig(
        resample_filter=tuple(train_args.resample_filter),
        channels_per_head=train_args.channels_per_head,
        dropout=train_args.dropout,
        res_balance=train_args.res_balance,
        attn_balance=train_args.attn_balance,
        clip_act=train_args.clip_act,
    )

    forward_nn = UNet(**vars(unet_cfg), **vars(block_cfg))

    cond_embedder = None
    if train_args.cond_embedder != "none" and len(train_args.parents) > 0:
        embedder_cfg = CondEmbedderConfig(
            parents=train_args.parents,
            parent_dims=parent_dims,
            cond_embed_dim=train_args.cond_embed_dim,
        )
        if train_args.cond_embedder == "per_attr":
            cond_embedder = PerAttrCondEmbedder(embedder_cfg)
        elif train_args.cond_embedder == "global":
            cond_embedder = GlobalCondEmbedder(embedder_cfg)
        else:
            raise ValueError(f"Unknown cond_embedder: {train_args.cond_embedder}")

    model = Flow(
        forward_nn=forward_nn,
        cond_embedder=cond_embedder,
        sigma=train_args.sigma,
        alpha=train_args.alpha,
        p_uncond=train_args.p_uncond,
        amp_dtype=amp_dtype,
    )
    return model.to(device)


def maybe_apply_ema(
    model: nn.Module,
    ckpt: dict,
    ema_rate: float,
    use_ema: bool,
) -> None:
    if not use_ema:
        return

    ema_state = ckpt.get("ema_state", None)
    if ema_state is None:
        print("EMA state not found in checkpoint; using raw model weights.")
        return

    ema = ModelEMA(model.parameters(), rate=ema_rate)
    ema.load_state_dict(ema_state)
    ema.apply()


def get_iterator(
    train_args: argparse.Namespace,
    batch_size: int,
    split: str,
):
    dataloaders = build_dataloaders_from_train_args(train_args, batch_size=batch_size)
    if split not in dataloaders:
        raise KeyError(f"Unknown split '{split}'. Available: {list(dataloaders.keys())}")
    return iter(dataloaders[split])


def move_pa_to_device(
    pa: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in pa.items()}


def preprocess_x_for_sampling(
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = x.float().to(device, non_blocking=True)
    channels = x.shape[1]
    if channels <= 3:
        x = x * 2 - 1
    return x


def clone_pa(pa: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in pa.items()}


def _schema_num_classes(do_key: str) -> int | None:
    if do_key not in CLASS_SCHEMA:
        raise KeyError(f"Unknown intervention key '{do_key}'. Available: {list(CLASS_SCHEMA.keys())}")

    spec = CLASS_SCHEMA[do_key]
    if isinstance(spec, float):
        return None
    return int(spec)


def _as_index_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1).round().long()


def apply_single_intervention(
    pa: dict[str, torch.Tensor],
    pa_rand: dict[str, torch.Tensor] | None,
    do_key: str,
    do_mode: str,
) -> dict[str, torch.Tensor]:
    if do_key not in pa:
        raise KeyError(f"Intervention key '{do_key}' not found in pa. Available: {list(pa.keys())}")

    pa_cf = clone_pa(pa)
    ref = pa_cf[do_key]
    num_classes = _schema_num_classes(do_key)

    # Continuous variable, e.g. age.
    if num_classes is None:
        if do_mode == "flip":
            raise ValueError(
                f"'flip' is not defined for continuous key '{do_key}'. "
                f"Use --do_mode random instead."
            )
        if pa_rand is None or do_key not in pa_rand:
            raise ValueError(f"Random intervention for '{do_key}' requires a random source batch.")
        pa_cf[do_key] = pa_rand[do_key].clone()
        return pa_cf

    # Discrete categorical variable stored as scalar-coded labels.
    orig_idx = _as_index_tensor(ref)

    if do_mode == "flip":
        # Binary: 0 <-> 1
        if num_classes == 2:
            new_idx = 1 - orig_idx
        else:
            # Multiclass: deterministic next-class cyclic shift.
            new_idx = (orig_idx + 1) % num_classes

        pa_cf[do_key] = new_idx.to(ref.dtype).view_as(ref)
        return pa_cf

    if do_mode == "random":
        if pa_rand is None or do_key not in pa_rand:
            raise ValueError(f"Random intervention for '{do_key}' requires a random source batch.")

        rand_idx = _as_index_tensor(pa_rand[do_key])

        # Clamp in case the sampled values come in float-coded form.
        rand_idx = rand_idx.clamp(min=0, max=num_classes - 1)

        # Ensure the discrete counterfactual actually changes the value.
        same = rand_idx == orig_idx
        if same.any():
            rand_idx[same] = (orig_idx[same] + 1) % num_classes

        pa_cf[do_key] = rand_idx.to(ref.dtype).view_as(ref)
        return pa_cf

    raise ValueError(f"Unknown do_mode: {do_mode}")


def sample_batch(
    model: nn.Module,
    noise: torch.Tensor,
    pa: dict[str, torch.Tensor] | None,
    steps: int,
):
    with torch.inference_mode():
        try:
            return model.sample(noise, steps=steps, pa=pa)
        except TypeError:
            return model.sample(noise, pa=pa, T=steps)


def invert_to_noise(
    model: nn.Module,
    x: torch.Tensor,
    pa_src: dict[str, torch.Tensor] | None,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
) -> torch.Tensor:
    if not hasattr(model, "ode_solve"):
        raise AttributeError(
            "Counterfactual generation requires model.ode_solve(...), "
            "but the current Flow model does not expose it."
        )

    t = torch.tensor([0.0, 1.0], device=x.device)
    with torch.no_grad():
        traj = model.ode_solve(
            x,
            pa=pa_src,
            t=t,
            method=ode_method,
            atol=ode_atol,
            rtol=ode_rtol,
        )
    return traj[-1]


def generate_from_inverted_noise(
    model: nn.Module,
    noise: torch.Tensor,
    pa_cf: dict[str, torch.Tensor] | None,
    sample_steps: int,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    use_ode_generation: bool,
) -> torch.Tensor:
    with torch.no_grad():
        if use_ode_generation:
            if not hasattr(model, "ode_solve"):
                raise AttributeError(
                    "ODE-based counterfactual generation requires model.ode_solve(...), "
                    "but the current Flow model does not expose it."
                )
            t = torch.tensor([1.0, 0.0], device=noise.device)
            traj = model.ode_solve(
                noise,
                pa=pa_cf,
                t=t,
                method=ode_method,
                atol=ode_atol,
                rtol=ode_rtol,
            )
            return traj[-1]

        return sample_batch(
            model=model,
            noise=noise,
            pa=pa_cf,
            steps=sample_steps,
        )


def save_random_samples(
    samples: torch.Tensor,
    save_dir: Path,
    start_idx: int,
    make_batch_grid: bool = True,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    vis = ((samples.clamp(-1, 1) + 1.0) / 2.0).cpu()

    for i in range(vis.shape[0]):
        save_image(vis[i], save_dir / f"sample_{start_idx + i:06d}.png")

    if make_batch_grid:
        nrow = int(math.sqrt(vis.shape[0]))
        nrow = max(nrow, 1)
        grid = make_grid(vis, nrow=nrow, pad_value=1.0)
        save_image(grid, save_dir / f"grid_{start_idx:06d}.png")


def save_counterfactual_samples(
    x_src: torch.Tensor,
    x_cf: torch.Tensor,
    save_dir: Path,
    start_idx: int,
    make_batch_grid: bool = True,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    src_vis = ((x_src.clamp(-1, 1) + 1.0) / 2.0).cpu()
    cf_vis = ((x_cf.clamp(-1, 1) + 1.0) / 2.0).cpu()

    for i in range(src_vis.shape[0]):
        idx = start_idx + i
        save_image(src_vis[i], save_dir / f"src_{idx:06d}.png")
        save_image(cf_vis[i], save_dir / f"cf_{idx:06d}.png")

        pair = make_grid(
            torch.stack([src_vis[i], cf_vis[i]], dim=0),
            nrow=2,
            pad_value=1.0,
        )
        save_image(pair, save_dir / f"pair_{idx:06d}.png")

    if make_batch_grid:
        panels = []
        for i in range(src_vis.shape[0]):
            panels.append(src_vis[i])
            panels.append(cf_vis[i])
        grid = make_grid(torch.stack(panels, dim=0), nrow=2, pad_value=1.0)
        save_image(grid, save_dir / f"grid_{start_idx:06d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--use_ema", action="store_true", default=False)

    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "cf"],
        help="random: sample from Gaussian noise; cf: generate counterfactuals from real images.",
    )
    parser.add_argument(
        "--cond_source",
        type=str,
        default="dataset",
        choices=["dataset", "none"],
        help="How to obtain conditioning variables for random conditional sampling.",
    )
    parser.add_argument("--do_key", type=str, default=None)
    parser.add_argument(
        "--do_mode",
        type=str,
        default="flip",
        choices=["flip", "random"],
        help="flip: binary flip or cyclic next-class for multiclass; random: resample from dataset.",
    )
    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument(
        "--cf_use_ode_generation",
        action="store_true",
        default=False,
        help="Use ODE solve for generation after inversion instead of model.sample(...).",
    )

    args = parser.parse_args()

    if args.mode == "cf" and args.do_key is None:
        parser.error("--mode cf requires --do_key")

    seed_all(args.seed, determ=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = to_namespace(ckpt["args"])

    model = build_flow_model_from_ckpt_args(train_args, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    maybe_apply_ema(
        model=model,
        ckpt=ckpt,
        ema_rate=getattr(train_args, "ema_rate", 0.9999),
        use_ema=args.use_ema,
    )
    model.eval()

    sample_steps = (
        args.sample_steps
        if args.sample_steps is not None
        else getattr(train_args, "T", 100)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cond_iter = None
    if args.mode == "random" and args.cond_source == "dataset":
        cond_iter = get_iterator(train_args, args.batch_size, args.split)

    src_iter = None
    rand_iter = None
    if args.mode == "cf":
        src_iter = get_iterator(train_args, args.batch_size, args.split)
        if args.do_mode == "random":
            rand_iter = get_iterator(train_args, args.batch_size, args.split)

    meta = {
        "ckpt": args.ckpt,
        "mode": args.mode,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "sample_steps": sample_steps,
        "seed": args.seed,
        "split": args.split,
        "use_ema": args.use_ema,
        "cond_source": args.cond_source,
        "do_key": args.do_key,
        "do_mode": args.do_mode,
        "ode_method": args.ode_method,
        "ode_atol": args.ode_atol,
        "ode_rtol": args.ode_rtol,
        "cf_use_ode_generation": args.cf_use_ode_generation,
    }
    with open(save_dir / "sampling_args.json", "w") as f:
        json.dump(meta, f, indent=2)

    produced = 0
    while produced < args.num_samples:
        bs = min(args.batch_size, args.num_samples - produced)

        if args.mode == "random":
            noise = torch.randn(
                bs,
                train_args.img_channels,
                train_args.img_height,
                train_args.img_width,
                device=device,
            )

            pa = None
            if cond_iter is not None:
                try:
                    batch = next(cond_iter)
                except StopIteration:
                    cond_iter = get_iterator(train_args, args.batch_size, args.split)
                    batch = next(cond_iter)

                pa = move_pa_to_device(batch["pa"], device)
                pa = {k: v[:bs] for k, v in pa.items()}

            samples = sample_batch(
                model=model,
                noise=noise,
                pa=pa,
                steps=sample_steps,
            )

            save_random_samples(samples, save_dir, produced, make_batch_grid=True)
            produced += bs
            print(f"Saved {produced}/{args.num_samples} random samples to {save_dir}")

        else:
            try:
                batch = next(src_iter)
            except StopIteration:
                src_iter = get_iterator(train_args, args.batch_size, args.split)
                batch = next(src_iter)

            x_src = preprocess_x_for_sampling(batch["x"][:bs], device)
            pa_src = move_pa_to_device(batch["pa"], device)
            pa_src = {k: v[:bs] for k, v in pa_src.items()}

            pa_rand = None
            if args.do_mode == "random":
                try:
                    rand_batch = next(rand_iter)
                except StopIteration:
                    rand_iter = get_iterator(train_args, args.batch_size, args.split)
                    rand_batch = next(rand_iter)

                pa_rand = move_pa_to_device(rand_batch["pa"], device)
                pa_rand = {k: v[:bs] for k, v in pa_rand.items()}

            pa_cf = apply_single_intervention(
                pa=pa_src,
                pa_rand=pa_rand,
                do_key=args.do_key,
                do_mode=args.do_mode,
            )

            noise = invert_to_noise(
                model=model,
                x=x_src,
                pa_src=pa_src,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
            )

            x_cf = generate_from_inverted_noise(
                model=model,
                noise=noise,
                pa_cf=pa_cf,
                sample_steps=sample_steps,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                use_ode_generation=args.cf_use_ode_generation,
            )

            save_counterfactual_samples(
                x_src=x_src,
                x_cf=x_cf,
                save_dir=save_dir,
                start_idx=produced,
                make_batch_grid=True,
            )
            produced += bs
            print(f"Saved {produced}/{args.num_samples} counterfactual pairs to {save_dir}")

    print(f"Done. Samples saved to: {save_dir}")


if __name__ == "__main__":
    main()