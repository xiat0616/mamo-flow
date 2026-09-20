import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from src.models.embedder import (
    GlobalCondEmbedder,
    PerAttrCondEmbedder,
    CondEmbedderConfig,
    infer_parent_dims_from_batch,
)
from src.models.unet import UNet
from src.flows.flow import Flow, BlockConfig, UNetConfig
from src.utils import ModelEMA, seed_all


plt.switch_backend("Agg")


DENSITY_LABELS = {0: "A", 1: "B", 2: "C", 3: "D"}
VIEW_LABELS = {0: "MLO", 1: "CC"}
CVIEW_LABELS = {0: "2D", 1: "CView"}
CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def to_namespace(d: dict) -> argparse.Namespace:
    return argparse.Namespace(**d)


def select_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def _format_float_tag(x: float) -> str:
    s = f"{x:.0e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s


def build_time_grid(
    direction: str,
    device: torch.device,
    ode_steps: int | None,
) -> torch.Tensor:
    if ode_steps is None:
        if direction == "forward":
            return torch.tensor([0.0, 1.0], device=device)
        if direction == "backward":
            return torch.tensor([1.0, 0.0], device=device)
        raise ValueError(f"Unknown direction: {direction}")

    if ode_steps < 1:
        raise ValueError(f"ode_steps must be >= 1, got {ode_steps}")

    if direction == "forward":
        return torch.linspace(0.0, 1.0, ode_steps + 1, device=device)
    if direction == "backward":
        return torch.linspace(1.0, 0.0, ode_steps + 1, device=device)
    raise ValueError(f"Unknown direction: {direction}")


def build_dataloaders_from_train_args(
    train_args: argparse.Namespace,
    batch_size: int,
) -> dict[str, torch.utils.data.DataLoader]:
    dataset_name = getattr(train_args, "dataset", "embed")

    # NOTE: Monkey code now, as the previously trained flow matching assume EMBED by default.
    if dataset_name is None:
        dataset_name = "embed"

    if dataset_name == "embed":
        from src.data_handle.embed import (
            get_embed,
            get_dataloaders,
            DataLoaderConfig,
            DatasetConfig,
        )
        assert train_args.img_channels == 1, "Embed dataset currently only supports single-channel images. Please set --img_channels 1 when using the embed dataset."
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

        # # Check the dataset batch['x'] before building dataloaders, to catch any potential issues early.
        # sample = datasets["train"][0]
        # print(f"Sample 'x' shape: {sample['x'].shape}, dtype: {sample['x'].dtype}, value range: [{sample['x'].min().item():.3f}, {sample['x'].max().item():.3f}]")
        # plt.imshow(sample['x'][0].numpy(), cmap='gray')
        # plt.title("Sample 'x' Visualization")
        # plt.axis("off")
        # plt.savefig("/vol/biomedic3/tx1215/mamo-flow/sample_dataset_image.png")
        # plt.close()

        dataloaders = get_dataloaders(
            DataLoaderConfig(
                bs=batch_size,
                num_workers=4,
                prefetch_factor=getattr(train_args, "prefetch_factor", 2),
                seed=getattr(train_args, "seed", 0),
                resume_step=0,
            ),
            datasets,
        )
        return dataloaders

    if dataset_name == "cifar10":
        from src.data_handle.cifar import (
            get_cifar10,
            get_dataloaders,
            DataLoaderConfig,
            DatasetConfig,
        )

        datasets = get_cifar10(
            DatasetConfig(
                data_dir=train_args.data_dir,
                valid_frac=getattr(train_args, "valid_frac", 0.05),
                split_seed=getattr(train_args, "split_seed", 33),
                img_height=train_args.img_height,
                img_width=train_args.img_width,
                img_channels=train_args.img_channels,
                use_labels_as_pa=True,
            )
        )

        dataloaders = get_dataloaders(
            DataLoaderConfig(
                bs=batch_size,
                num_workers=4,
                prefetch_factor=getattr(train_args, "prefetch_factor", 2),
                seed=getattr(train_args, "seed", 0),
                resume_step=0,
            ),
            datasets,
        )
        return dataloaders

    raise ValueError(f"Unknown dataset: {dataset_name}")


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
    """
    Match the training preview path:
        dataloader image [0, 1] -> model image [-1, 1]

    Do not clamp here. This is model input preprocessing, not visualization.
    """
    x = x.float().to(device, non_blocking=True)
    channels = x.shape[1]
    if channels <= 3:
        x = x * 2.0 - 1.0
    return x


def clone_pa(pa: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in pa.items()}


def get_class_schema(train_args: argparse.Namespace) -> dict[str, float | int]:
    dataset_name = getattr(train_args, "dataset", "embed")

    if dataset_name == "embed":
        from src.data_handle.embed import CLASS_SCHEMA
        return CLASS_SCHEMA

    if dataset_name == "cifar10":
        return {"y": 10}

    return {}


def _schema_num_classes(
    do_key: str,
    ref: torch.Tensor,
    class_schema: dict[str, float | int],
) -> int | None:
    if do_key in class_schema:
        spec = class_schema[do_key]
        if isinstance(spec, float):
            return None
        return int(spec)

    # Fallback: infer from one-hot tensor.
    if ref.shape[-1] > 1:
        return int(ref.shape[-1])

    raise KeyError(
        f"Unknown intervention key '{do_key}'. Available schema keys: {list(class_schema.keys())}"
    )


def _as_index_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] > 1:  # one-hot encoded
        return x.argmax(dim=-1)
    return x.view(-1).round().long()


def apply_single_intervention(
    pa: dict[str, torch.Tensor],
    pa_rand: dict[str, torch.Tensor] | None,
    do_key: str | None,
    do_mode: str,
    class_schema: dict[str, float | int],
) -> dict[str, torch.Tensor]:
    if do_mode == "null":
        return clone_pa(pa)

    if do_key is None:
        raise ValueError(f"do_key must be provided for do_mode='{do_mode}'")

    if do_key not in pa:
        raise KeyError(f"Intervention key '{do_key}' not found in pa. Available: {list(pa.keys())}")

    pa_cf = clone_pa(pa)
    ref = pa_cf[do_key]
    num_classes = _schema_num_classes(do_key, ref, class_schema)

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

    orig_idx = _as_index_tensor(ref)

    def _to_pa(new_idx: torch.Tensor) -> torch.Tensor:
        if ref.shape[-1] > 1:  # one-hot
            one_hot = torch.zeros_like(ref)
            one_hot.scatter_(-1, new_idx.unsqueeze(-1), 1.0)
            return one_hot
        return new_idx.to(ref.dtype).view_as(ref)

    if do_mode == "flip":
        if num_classes == 2:
            new_idx = 1 - orig_idx
        else:
            new_idx = (orig_idx + 1) % num_classes

        pa_cf[do_key] = _to_pa(new_idx)
        return pa_cf

    if do_mode == "random":
        if pa_rand is None or do_key not in pa_rand:
            raise ValueError(f"Random intervention for '{do_key}' requires a random source batch.")

        rand_idx = _as_index_tensor(pa_rand[do_key]).clamp(min=0, max=num_classes - 1)
        same = rand_idx == orig_idx

        if same.any():
            num_same = int(same.sum().item())
            rand_offset = torch.randint(
                low=1,
                high=num_classes,
                size=(num_same,),
                device=orig_idx.device,
            )
            rand_idx[same] = (orig_idx[same] + rand_offset) % num_classes

        pa_cf[do_key] = _to_pa(rand_idx)
        return pa_cf

    raise ValueError(f"Unknown do_mode: {do_mode}")


def generate_random_from_noise(
    model: nn.Module,
    noise: torch.Tensor,
    pa: dict[str, torch.Tensor] | None,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> torch.Tensor:
    if not hasattr(model, "ode_solve"):
        raise AttributeError(
            "Random generation requires model.ode_solve(...), "
            "but the current Flow model does not expose it."
        )

    t = build_time_grid("forward", noise.device, ode_steps)
    traj = model.ode_solve(
        noise,
        pa=pa,
        t=t,
        method=ode_method,
        atol=ode_atol,
        rtol=ode_rtol,
    )
    return traj[-1]


def invert_to_noise(
    model: nn.Module,
    x: torch.Tensor,
    pa_src: dict[str, torch.Tensor] | None,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> torch.Tensor:
    if not hasattr(model, "ode_solve"):
        raise AttributeError(
            "Counterfactual generation requires model.ode_solve(...), "
            "but the current Flow model does not expose it."
        )

    t = build_time_grid("backward", x.device, ode_steps)
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
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> torch.Tensor:
    if not hasattr(model, "ode_solve"):
        raise AttributeError(
            "Counterfactual generation requires model.ode_solve(...), "
            "but the current Flow model does not expose it."
        )

    t = build_time_grid("forward", noise.device, ode_steps)
    traj = model.ode_solve(
        noise,
        pa=pa_cf,
        t=t,
        method=ode_method,
        atol=ode_atol,
        rtol=ode_rtol,
    )
    return traj[-1]


def get_ckpt_tag(ckpt_path: str) -> str:
    return Path(ckpt_path).stem


def get_exp_name(train_args: argparse.Namespace) -> str:
    return getattr(train_args, "exp_name", "unknown_exp")


def get_sampler_tag(
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> str:
    if ode_steps is not None:
        return f"ode-{ode_method}_steps-{ode_steps}"

    return (
        f"ode-{ode_method}"
        f"_atol-{_format_float_tag(ode_atol)}"
        f"_rtol-{_format_float_tag(ode_rtol)}"
    )


def build_sampling_root(
    save_root: str,
    ckpt_path: str,
    train_args: argparse.Namespace,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> Path:
    exp_name = get_exp_name(train_args)
    ckpt_tag = get_ckpt_tag(ckpt_path)
    sampler_tag = get_sampler_tag(
        ode_method=ode_method,
        ode_atol=ode_atol,
        ode_rtol=ode_rtol,
        ode_steps=ode_steps,
    )
    return Path(save_root) / exp_name / ckpt_tag / sampler_tag


def build_random_save_dirs(
    save_root: str,
    ckpt_path: str,
    train_args: argparse.Namespace,
    cond_source: str,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> dict[str, Path]:
    root = build_sampling_root(
        save_root=save_root,
        ckpt_path=ckpt_path,
        train_args=train_args,
        ode_method=ode_method,
        ode_atol=ode_atol,
        ode_rtol=ode_rtol,
        ode_steps=ode_steps,
    )
    cond_tag = "cond_dataset" if cond_source == "dataset" else "uncond"
    random_root = root / "randoms" / cond_tag
    return {
        "root": random_root,
        "rs": random_root / "rs",
        "rs_visual": random_root / "rs_visual",
    }


def build_cf_save_dirs(
    save_root: str,
    ckpt_path: str,
    train_args: argparse.Namespace,
    do_key: str | None,
    do_mode: str,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
) -> dict[str, Path]:
    root = build_sampling_root(
        save_root=save_root,
        ckpt_path=ckpt_path,
        train_args=train_args,
        ode_method=ode_method,
        ode_atol=ode_atol,
        ode_rtol=ode_rtol,
        ode_steps=ode_steps,
    )

    if do_mode == "null":
        cf_root = root / "reconstructions" / "null"
    else:
        cf_root = root / "cfs" / str(do_key) / str(do_mode)

    return {
        "root": cf_root,
        "inputs": cf_root / "inputs",
        "cfs": cf_root / "cfs",
        "cf_visuals": cf_root / "cf_visuals",
    }


def _get_pa_scalar(pa: dict[str, torch.Tensor], key: str, idx: int) -> float:
    v = pa[key]
    if v.ndim == 0:
        return float(v.detach().cpu().item())
    vi = v[idx]
    if vi.numel() > 1:  # one-hot encoded
        return float(vi.argmax().item())
    return float(vi.item())


def _format_pa_value(key: str, value: float) -> str:
    if key == "age":
        return f"{value * 100:.1f}"

    ivalue = int(round(value))
    if key == "density":
        return DENSITY_LABELS.get(ivalue, str(ivalue))
    if key == "view":
        return VIEW_LABELS.get(ivalue, str(ivalue))
    if key == "cview":
        return CVIEW_LABELS.get(ivalue, str(ivalue))
    if key == "y":
        return CIFAR10_LABELS.get(ivalue, str(ivalue))
    return str(ivalue)


def _format_attr_block(
    title: str,
    pa: dict[str, torch.Tensor],
    idx: int,
    parents: list[str],
    items_per_line: int = 2,
) -> str:
    items = []
    for key in parents:
        value = _get_pa_scalar(pa, key, idx)
        items.append(f"{key}={_format_pa_value(key, value)}")

    lines = [title]
    for start in range(0, len(items), items_per_line):
        lines.append(" | ".join(items[start:start + items_per_line]))
    return "\n".join(lines)


def _format_random_hparam_title(
    meta: dict,
    global_idx: int,
) -> str:
    return (
        f"random sample #{global_idx:06d}\n"
        f"mode={meta['mode']} | cond_source={meta['cond_source']}"
    )


def _format_random_hparam_xlabel(meta: dict) -> str:
    if meta["ode_steps"] is not None:
        solver = f"solver={meta['ode_method']} | steps={meta['ode_steps']}"
    else:
        solver = (
            f"solver={meta['ode_method']} | "
            f"atol={meta['ode_atol']:.1e} | rtol={meta['ode_rtol']:.1e}"
        )

    run_cfg = (
        f"split={meta['split']} | batch_size={meta['batch_size']} | "
        f"seed={meta['seed']} | use_ema={meta['use_ema']}"
    )
    return f"{solver}\n{run_cfg}"


def _format_random_hparam_ylabel(
    pa: dict[str, torch.Tensor] | None,
    idx: int,
    parents: list[str],
) -> str:
    if pa is None:
        return "conditioning\nnone (unconditional)"

    return _format_attr_block(
        title="conditioning",
        pa=pa,
        idx=idx,
        parents=parents,
        items_per_line=2,
    )


def _show_image_matplotlib(ax: plt.Axes, img: torch.Tensor) -> None:
    """
    Matplotlib display matching the training plotting style.

    For single-channel images, intentionally do not pass vmin/vmax.
    Matplotlib then uses auto display scaling, which is why the training
    plots looked less washed out than direct PIL/to_pil_image rendering.
    """
    img = img.detach().cpu().float()
    # print(f"Image shape: {tuple(img.shape)}, dtype: {img.dtype}, value range: [{img.min().item():.3f}, {img.max().item():.3f}]")
    if img.ndim == 2:
        ax.imshow(img.numpy(), cmap="gray")
        return

    if img.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [H,W], got shape {tuple(img.shape)}")

    if img.shape[0] == 1:
        ax.imshow(img[0].numpy(), cmap="gray")
    elif img.shape[0] in {3, 4}:
        ax.imshow(img[:3].permute(1, 2, 0).clamp(0, 1).numpy())
    else:
        raise ValueError(f"Unsupported channel count: {img.shape[0]}")


def _save_random_visual(
    img: torch.Tensor,
    save_path: Path,
    pa: dict[str, torch.Tensor] | None,
    idx: int,
    global_idx: int,
    parents: list[str],
    meta: dict,
) -> None:
    title = _format_random_hparam_title(meta, global_idx)
    xlabel = _format_random_hparam_xlabel(meta)
    ylabel = _format_random_hparam_ylabel(pa, idx, parents)

    fig, ax = plt.subplots(figsize=(6, 6))
    _show_image_matplotlib(ax, img)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, rotation=0, labelpad=55, va="center")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _to_gray_np(img: torch.Tensor) -> np.ndarray:
    img_np = img.detach().cpu().float().numpy()

    if img_np.ndim == 3:
        if img_np.shape[0] == 1:
            return img_np[0]
        return img_np.mean(axis=0)

    if img_np.ndim == 2:
        return img_np

    raise ValueError(f"Expected [C,H,W] or [H,W], got shape {tuple(img.shape)}")


def _save_cf_visual_with_diff_matplotlib(
    src_img: torch.Tensor,
    cf_img: torch.Tensor,
    pa_src: dict[str, torch.Tensor],
    pa_cf: dict[str, torch.Tensor],
    idx: int,
    parents: list[str],
    save_path: Path,
) -> None:
    """
    Save input / counterfactual / difference with Matplotlib.

    src_img and cf_img should already be display tensors in [0, 1].
    The input and CF panels use the same Matplotlib auto-display style as
    the training plots. The difference panel is computed from the raw linear
    [0, 1] tensors, not from any windowed or rescaled image.
    """
    src_gray = _to_gray_np(src_img)
    cf_gray = _to_gray_np(cf_img)

    diff = cf_gray - src_gray
    diff_vmax = float(np.max(np.abs(diff)))
    diff_vmax = max(diff_vmax, 1e-8)

    # Match training-style effect-map scale.
    diff_display = diff * 255.0
    amax = float(np.max(np.abs(diff_display)))
    amax = max(amax, 1e-8)

    text_left = _format_attr_block("attrs", pa_src, idx, parents)
    text_mid = _format_attr_block("attrs", pa_cf, idx, parents)
    text_right = (
        "red: cf > input\n"
        "blue: cf < input\n"
        f"max|diff|={diff_vmax:.3f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    
    # print(f"src_img shape: {tuple(src_img.shape)}, dtype: {src_img.dtype}, value range: [{src_img.min().item():.3f}, {src_img.max().item():.3f}]")
    _show_image_matplotlib(axes[0], src_img)
    axes[0].set_title("input", fontsize=11)
    axes[0].set_xlabel(text_left, fontsize=8)

    # print(f"cf_img shape: {tuple(cf_img.shape)}, dtype: {cf_img.dtype}, value range: [{cf_img.min().item():.3f}, {cf_img.max().item():.3f}]")
    _show_image_matplotlib(axes[1], cf_img)
    axes[1].set_title("cf", fontsize=11)
    axes[1].set_xlabel(text_mid, fontsize=8)

    im = axes[2].imshow(
        diff_display,
        cmap="RdBu_r",
        vmin=-amax,
        vmax=amax,
    )
    axes[2].set_title("difference (cf - input)", fontsize=11)
    axes[2].set_xlabel(text_right, fontsize=8)

    cbar = fig.colorbar(
        im,
        ax=axes[2],
        orientation="horizontal",
        fraction=0.046,
        pad=0.12,
    )
    cbar.outline.set_visible(False)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_random_samples(
    samples: torch.Tensor,
    save_dirs: dict[str, Path],
    start_idx: int,
    pa: dict[str, torch.Tensor] | None,
    parents: list[str],
    meta: dict,
) -> None:
    for d in save_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    vis = ((samples.clamp(-1, 1) + 1.0) / 2.0).cpu()

    assert vis.min() >= 0 and vis.max() <= 1, f"vis has out-of-range values: [{vis.min().item():.3f}, {vis.max().item():.3f}]"

    pa_cpu = None if pa is None else {k: v.detach().cpu() for k, v in pa.items()}

    for i in range(vis.shape[0]):
        idx = start_idx + i
        save_image(vis[i], save_dirs["rs"] / f"{idx:06d}_rand.png")
        _save_random_visual(
            img=vis[i],
            save_path=save_dirs["rs_visual"] / f"{idx:06d}_viz.png",
            pa=pa_cpu,
            idx=i,
            global_idx=idx,
            parents=parents,
            meta=meta,
        )


def save_counterfactual_samples(
    x_src: torch.Tensor,
    x_cf: torch.Tensor,
    pa_src: dict[str, torch.Tensor],
    pa_cf: dict[str, torch.Tensor],
    parents: list[str],
    save_dirs: dict[str, Path],
    start_idx: int,
) -> None:
    for d in save_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    src_vis = ((x_src.clamp(-1, 1) + 1.0) / 2.0).cpu() #[0, 1] for visualization
    cf_vis = ((x_cf.clamp(-1, 1) + 1.0) / 2.0).cpu() #[0, 1] for visualization

    assert src_vis.min() >= 0 and src_vis.max() <= 1, f"src_vis has out-of-range values: [{src_vis.min().item():.3f}, {src_vis.max().item():.3f}]"
    assert cf_vis.min() >= 0 and cf_vis.max() <= 1, f"cf_vis has out-of-range values: [{cf_vis.min().item():.3f}, {cf_vis.max().item():.3f}]"

    pa_src_cpu = {k: v.detach().cpu() for k, v in pa_src.items()}
    pa_cf_cpu = {k: v.detach().cpu() for k, v in pa_cf.items()}

    for i in range(src_vis.shape[0]):
        idx = start_idx + i

        # Raw linear image saves. These are useful for quantitative inspection,
        # but they may look different from Matplotlib auto-displayed figures.
        # print(src_vis[i].shape, src_vis[i].dtype, src_vis[i].min().item(), src_vis[i].max().item())
        # print(cf_vis[i].shape, cf_vis[i].dtype, cf_vis[i].min().item(), cf_vis[i].max().item())
        save_image(src_vis[i], save_dirs["inputs"] / f"{idx:06d}_input.png")
        save_image(cf_vis[i], save_dirs["cfs"] / f"{idx:06d}_cf.png")

        # Human-readable visual, now using Matplotlib like the training plot.
        _save_cf_visual_with_diff_matplotlib(
            src_img=src_vis[i],
            cf_img=cf_vis[i],
            pa_src=pa_src_cpu,
            pa_cf=pa_cf_cpu,
            idx=i,
            parents=parents,
            save_path=save_dirs["cf_visuals"] / f"{idx:06d}_viz.png",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Override the split_dir stored in the checkpoint.",
    )
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--use_ema", action="store_true", default=False)

    parser.add_argument(
        "--mode",
        type=str,
        default="rs",
        choices=["rs", "cf"],
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
        choices=["null", "flip", "random"],
        help=(
            "null: keep conditioning unchanged and reconstruct the input; "
            "flip: binary flip or cyclic next-class for multiclass; "
            "random: resample from dataset."
        ),
    )
    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="Number of intervals on [0,1] for the external time grid. Especially useful for fixed-step solvers.",
    )

    args = parser.parse_args()

    if args.mode == "cf" and args.do_mode != "null" and args.do_key is None:
        parser.error("--mode cf requires --do_key unless --do_mode null")

    seed_all(args.seed, deterministic=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = to_namespace(ckpt["args"])
    if args.split_dir is not None:
        train_args.split_dir = args.split_dir

    model = build_flow_model_from_ckpt_args(train_args, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    maybe_apply_ema(
        model=model,
        ckpt=ckpt,
        ema_rate=getattr(train_args, "ema_rate", 0.9999),
        use_ema=args.use_ema,
    )
    model.eval()

    class_schema = get_class_schema(train_args)

    if args.mode == "rs":
        random_save_dirs = build_random_save_dirs(
            save_root=args.save_dir,
            ckpt_path=args.ckpt,
            train_args=train_args,
            cond_source=args.cond_source,
            ode_method=args.ode_method,
            ode_atol=args.ode_atol,
            ode_rtol=args.ode_rtol,
            ode_steps=args.ode_steps,
        )
        save_dir = random_save_dirs["root"]
        for d in random_save_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        cf_save_dirs = None
    elif args.mode == "cf":
        cf_save_dirs = build_cf_save_dirs(
            save_root=args.save_dir,
            ckpt_path=args.ckpt,
            train_args=train_args,
            do_key=args.do_key,
            do_mode=args.do_mode,
            ode_method=args.ode_method,
            ode_atol=args.ode_atol,
            ode_rtol=args.ode_rtol,
            ode_steps=args.ode_steps,
        )
        save_dir = cf_save_dirs["root"]
        for d in cf_save_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        random_save_dirs = None
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    cond_iter = None
    if args.mode == "rs" and args.cond_source == "dataset":
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
        "seed": args.seed,
        "split": args.split,
        "use_ema": args.use_ema,
        "cond_source": args.cond_source,
        "do_key": args.do_key,
        "do_mode": args.do_mode,
        "ode_method": args.ode_method,
        "ode_atol": None if args.ode_steps is not None else args.ode_atol,
        "ode_rtol": None if args.ode_steps is not None else args.ode_rtol,
        "ode_steps": args.ode_steps,
    }

    with open(save_dir / "sampling_args.json", "w") as f:
        json.dump(meta, f, indent=2)

    produced = 0
    while produced < args.num_samples:
        bs = min(args.batch_size, args.num_samples - produced)

        if args.mode == "rs":
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

            samples = generate_random_from_noise(
                model=model,
                noise=noise,
                pa=pa,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )

            save_random_samples(
                samples=samples,
                save_dirs=random_save_dirs,
                start_idx=produced,
                pa=pa,
                parents=train_args.parents,
                meta=meta,
            )
            produced += bs
            print(f"Saved {produced}/{args.num_samples} random samples to {save_dir}")

        elif args.mode == "cf":
            try:
                batch = next(src_iter)
            except StopIteration:
                src_iter = get_iterator(train_args, args.batch_size, args.split)
                batch = next(src_iter)

            # # Save one image for sanity checking the input image
            # plt.imshow(batch["x"][0].permute(1, 2, 0).cpu(), cmap="gray")
            # plt.title("Sample input image (before preprocessing)")
            # plt.axis("off")
            # plt.savefig("/vol/biomedic3/tx1215/mamo-flow/sample_input_image.png")
            # plt.close()

            x_src = preprocess_x_for_sampling(batch["x"][:bs], device)
            assert x_src.max() <= 1.0 and x_src.min() >= -1.0, "Preprocessed source images should be in [-1, 1]"
   
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
                class_schema=class_schema,
            )

            noise = invert_to_noise(
                model=model,
                x=x_src,
                pa_src=pa_src,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )

            x_cf = generate_from_inverted_noise(
                model=model,
                noise=noise,
                pa_cf=pa_cf,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
            )
    
            save_counterfactual_samples(
                x_src=x_src,
                x_cf=x_cf,
                pa_src=pa_src,
                pa_cf=pa_cf,
                parents=train_args.parents,
                save_dirs=cf_save_dirs,
                start_idx=produced,
            )
            produced += bs
            print(f"Saved {produced}/{args.num_samples} outputs to {save_dir}")

    print(f"Done. Samples saved to: {save_dir}")


if __name__ == "__main__":
    main()
