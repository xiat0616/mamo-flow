import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from src.models.embedder import (
    GlobalCondEmbedder,
    PerAttrCondEmbedder,
    CondEmbedderConfig,
    infer_parent_dims_from_batch,
)
from src.models.unet import UNet
from src.flows.meanflow import (
    MeanFlow,
    MeanFlowConfig,
    SampleConfig,
    BlockConfig,
    UNetConfig,
)
from src.utils import ModelEMA, seed_all


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

    if dataset_name is None:
        dataset_name = "embed"

    if dataset_name == "embed":
        from src.data_handle.embed import (
            get_embed,
            get_dataloaders,
            DataLoaderConfig,
            DatasetConfig,
        )

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


def _get_mf_arg(train_args: argparse.Namespace, mf_name: str, plain_name: str, default):
    if hasattr(train_args, mf_name):
        return getattr(train_args, mf_name)
    if hasattr(train_args, plain_name):
        return getattr(train_args, plain_name)
    return default


def build_meanflow_model_from_ckpt_args(
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
    if getattr(train_args, "cond_embedder", "none") != "none" and len(train_args.parents) > 0:
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

    mf_config = MeanFlowConfig(
        ratio_r_neq_t=_get_mf_arg(train_args, "mf_ratio_r_neq_t", "ratio_r_neq_t", 0.25),
        time_sampler=_get_mf_arg(train_args, "mf_time_sampler", "time_sampler", "lognorm"),
        lognorm_mu=_get_mf_arg(train_args, "mf_lognorm_mu", "lognorm_mu", -0.4),
        lognorm_sigma=_get_mf_arg(train_args, "mf_lognorm_sigma", "lognorm_sigma", 1.0),
        adaptive_weight_p=_get_mf_arg(train_args, "mf_adaptive_weight_p", "adaptive_weight_p", 1.0),
        adaptive_weight_eps=_get_mf_arg(train_args, "mf_adaptive_weight_eps", "adaptive_weight_eps", 1e-3),
    )

    model = MeanFlow(
        forward_nn=forward_nn,
        cond_embedder=cond_embedder,
        p_uncond=getattr(train_args, "p_uncond", 0.0),
        amp_dtype=amp_dtype,
        mf_config=mf_config,
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

    if ref.shape[-1] > 1:
        return int(ref.shape[-1])

    raise KeyError(
        f"Unknown intervention key '{do_key}'. Available schema keys: {list(class_schema.keys())}"
    )


def _as_index_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] > 1:
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
        if ref.shape[-1] > 1:
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


def make_sample_args(
    cfg_mode: str,
    cfg_scale: float,
    null_keys: list[str] | None,
) -> SampleConfig:
    return SampleConfig(
        cfg_mode=cfg_mode,
        cfg_scale=cfg_scale,
        null_keys=None if not null_keys else set(null_keys),
    )


def generate_random_from_noise(
    model: nn.Module,
    noise: torch.Tensor,
    pa: dict[str, torch.Tensor] | None,
    sample_steps: int,
    sample_args: SampleConfig,
) -> torch.Tensor:
    if not hasattr(model, "sample"):
        raise AttributeError(
            "Random generation requires model.sample(...), "
            "but the current MeanFlow model does not expose it."
        )
    return model.sample(
        noise,
        steps=sample_steps,
        pa=pa,
        sample_args=sample_args,
    )


def invert_to_noise_ode(
    model: nn.Module,
    x: torch.Tensor,
    pa_src: dict[str, torch.Tensor] | None,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
    sample_args: SampleConfig | None = None,
) -> torch.Tensor:
    if not hasattr(model, "ode_solve"):
        raise AttributeError(
            "Counterfactual inversion requires model.ode_solve(...), "
            "but the current MeanFlow model does not expose it."
        )

    t = build_time_grid("backward", x.device, ode_steps)
    traj = model.ode_solve(
        x,
        pa=pa_src,
        sample_args=sample_args,
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


def get_cfg_tag(cfg_mode: str, cfg_scale: float, null_keys: list[str] | None) -> str:
    if cfg_mode == "none":
        base = "no_cfg"
    else:
        base = f"{cfg_mode}_w-{cfg_scale:g}"

    if null_keys:
        joined = "_".join(sorted(null_keys))
        base = f"{base}_null-{joined}"
    return base


def get_rs_sampler_tag(
    sample_steps: int,
    cfg_mode: str,
    cfg_scale: float,
    null_keys: list[str] | None,
) -> str:
    cfg_tag = get_cfg_tag(cfg_mode, cfg_scale, null_keys)
    return f"mf_steps-{sample_steps}_{cfg_tag}"


def get_cf_sampler_tag(
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    ode_steps: int | None,
    sample_steps: int,
    cfg_mode: str,
    cfg_scale: float,
    null_keys: list[str] | None,
) -> str:
    if ode_steps is not None:
        inv_tag = f"invode-{ode_method}_steps-{ode_steps}"
    else:
        inv_tag = (
            f"invode-{ode_method}"
            f"_atol-{_format_float_tag(ode_atol)}"
            f"_rtol-{_format_float_tag(ode_rtol)}"
        )

    rs_tag = f"mf_steps-{sample_steps}"
    cfg_tag = get_cfg_tag(cfg_mode, cfg_scale, null_keys)
    return f"{inv_tag}_{rs_tag}_{cfg_tag}"


def build_sampling_root(
    save_root: str,
    ckpt_path: str,
    train_args: argparse.Namespace,
    sampler_tag: str,
) -> Path:
    exp_name = get_exp_name(train_args)
    ckpt_tag = get_ckpt_tag(ckpt_path)
    return Path(save_root) / exp_name / ckpt_tag / sampler_tag


def build_random_save_dirs(
    save_root: str,
    ckpt_path: str,
    train_args: argparse.Namespace,
    cond_source: str,
    sampler_tag: str,
) -> dict[str, Path]:
    root = build_sampling_root(
        save_root=save_root,
        ckpt_path=ckpt_path,
        train_args=train_args,
        sampler_tag=sampler_tag,
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
    sampler_tag: str,
) -> dict[str, Path]:
    root = build_sampling_root(
        save_root=save_root,
        ckpt_path=ckpt_path,
        train_args=train_args,
        sampler_tag=sampler_tag,
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
    if vi.numel() > 1:
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


def _format_random_hparam_title(meta: dict, global_idx: int) -> str:
    return (
        f"random sample #{global_idx:06d}\n"
        f"mode={meta['mode']} | cond_source={meta['cond_source']}"
    )


def _format_random_hparam_xlabel(meta: dict) -> str:
    line1 = f"sample_steps={meta['sample_steps']} | cfg={meta['cfg_mode']} | cfg_scale={meta['cfg_scale']}"
    line2 = (
        f"split={meta['split']} | batch_size={meta['batch_size']} | "
        f"seed={meta['seed']} | use_ema={meta['use_ema']}"
    )
    return f"{line1}\n{line2}"


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

    img_np = img.detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    if img_np.ndim == 3 and img_np.shape[0] == 1:
        ax.imshow(img_np[0], cmap="gray", vmin=0.0, vmax=1.0)
    elif img_np.ndim == 3:
        ax.imshow(np.transpose(img_np, (1, 2, 0)))
    else:
        ax.imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, rotation=0, labelpad=55, va="center")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_diff_heatmap(
    src_img: torch.Tensor,
    cf_img: torch.Tensor,
) -> tuple[Image.Image, float]:
    src_np = src_img.detach().cpu().float().numpy()
    cf_np = cf_img.detach().cpu().float().numpy()

    if src_np.ndim == 3:
        src_gray = src_np.mean(axis=0)
        cf_gray = cf_np.mean(axis=0)
    else:
        src_gray = src_np
        cf_gray = cf_np

    diff = cf_gray - src_gray
    vmax = float(np.max(np.abs(diff)))
    vmax = max(vmax, 1e-8)

    diff_norm = (diff / vmax + 1.0) / 2.0
    cmap = colormaps["coolwarm"]
    diff_rgb = (cmap(diff_norm)[..., :3] * 255).astype(np.uint8)

    return Image.fromarray(diff_rgb), vmax


def _render_cf_visual_with_diff(
    src_img: torch.Tensor,
    cf_img: torch.Tensor,
    pa_src: dict[str, torch.Tensor],
    pa_cf: dict[str, torch.Tensor],
    idx: int,
    parents: list[str],
    gutter: int = 12,
    pad: int = 6,
) -> Image.Image:
    src_pil = to_pil_image(src_img).convert("RGB")
    cf_pil = to_pil_image(cf_img).convert("RGB")
    diff_pil, diff_vmax = _make_diff_heatmap(src_img, cf_img)

    font = ImageFont.load_default()

    title_left = "input"
    title_mid = "cf"
    title_right = "difference (cf - input)"

    text_left = _format_attr_block("attrs", pa_src, idx, parents)
    text_mid = _format_attr_block("attrs", pa_cf, idx, parents)
    text_right = (
        "heatmap\n"
        "red: cf > input\n"
        "blue: cf < input\n"
        f"max|diff|={diff_vmax:.3f}"
    )

    probe = Image.new("RGB", (10, 10), "white")
    draw_probe = ImageDraw.Draw(probe)

    def text_size(txt: str) -> tuple[int, int]:
        bbox = draw_probe.multiline_textbbox((0, 0), txt, font=font, spacing=2)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    col_w = max(src_pil.width, cf_pil.width, diff_pil.width)

    _, title_h_left = text_size(title_left)
    _, title_h_mid = text_size(title_mid)
    _, title_h_right = text_size(title_right)
    title_h = max(title_h_left, title_h_mid, title_h_right)

    _, text_h_left = text_size(text_left)
    _, text_h_mid = text_size(text_mid)
    _, text_h_right = text_size(text_right)
    text_h = max(text_h_left, text_h_mid, text_h_right)

    img_h = max(src_pil.height, cf_pil.height, diff_pil.height)

    total_w = col_w * 3 + gutter * 2 + pad * 2
    total_h = pad + title_h + pad + img_h + pad + text_h + pad

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    cols_x = [
        pad,
        pad + col_w + gutter,
        pad + 2 * (col_w + gutter),
    ]

    titles = [title_left, title_mid, title_right]
    images = [src_pil, cf_pil, diff_pil]
    texts = [text_left, text_mid, text_right]

    for x0, title, img, txt in zip(cols_x, titles, images, texts):
        title_bbox = draw.multiline_textbbox((0, 0), title, font=font, spacing=2)
        title_w = title_bbox[2] - title_bbox[0]
        draw.multiline_text(
            (x0 + (col_w - title_w) // 2, pad),
            title,
            fill="black",
            font=font,
            spacing=2,
        )

        img_x = x0 + (col_w - img.width) // 2
        img_y = pad + title_h + pad
        canvas.paste(img, (img_x, img_y))

        draw.multiline_text(
            (x0, img_y + img_h + pad),
            txt,
            fill="black",
            font=font,
            spacing=2,
        )

    return canvas


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

    src_vis = ((x_src.clamp(-1, 1) + 1.0) / 2.0).cpu()
    cf_vis = ((x_cf.clamp(-1, 1) + 1.0) / 2.0).cpu()

    pa_src_cpu = {k: v.detach().cpu() for k, v in pa_src.items()}
    pa_cf_cpu = {k: v.detach().cpu() for k, v in pa_cf.items()}

    for i in range(src_vis.shape[0]):
        idx = start_idx + i

        save_image(src_vis[i], save_dirs["inputs"] / f"{idx:06d}_input.png")
        save_image(cf_vis[i], save_dirs["cfs"] / f"{idx:06d}_cf.png")

        viz_img = _render_cf_visual_with_diff(
            src_img=src_vis[i],
            cf_img=cf_vis[i],
            pa_src=pa_src_cpu,
            pa_cf=pa_cf_cpu,
            idx=i,
            parents=parents,
        )
        viz_img.save(save_dirs["cf_visuals"] / f"{idx:06d}_viz.png")


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
        help="rs: random sampling with MeanFlow.sample(...); cf: experimental ODE inversion + MeanFlow sampling.",
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
            "null: keep conditioning unchanged and reconstruct approximately; "
            "flip: binary flip or cyclic next-class for multiclass; "
            "random: resample from dataset."
        ),
    )

    parser.add_argument("--sample_steps", type=int, default=1)

    parser.add_argument(
        "--cfg_mode",
        type=str,
        default="none",
        choices=["none", "cfg"],
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--null_keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional conditioning keys to null during CFG.",
    )

    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="Number of intervals on [0,1] for ODE inversion in cf mode.",
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

    model = build_meanflow_model_from_ckpt_args(train_args, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    maybe_apply_ema(
        model=model,
        ckpt=ckpt,
        ema_rate=getattr(train_args, "ema_rate", 0.9999),
        use_ema=args.use_ema,
    )
    model.eval()

    class_schema = get_class_schema(train_args)

    sample_args = make_sample_args(
        cfg_mode=args.cfg_mode,
        cfg_scale=args.cfg_scale,
        null_keys=args.null_keys,
    )
    # ode_solve in your current MeanFlow mainly uses pa/null_keys;
    # keep this conservative for inversion.
    ode_sample_args = make_sample_args(
        cfg_mode="none",
        cfg_scale=1.0,
        null_keys=args.null_keys,
    )

    if args.mode == "rs":
        sampler_tag = get_rs_sampler_tag(
            sample_steps=args.sample_steps,
            cfg_mode=args.cfg_mode,
            cfg_scale=args.cfg_scale,
            null_keys=args.null_keys,
        )
        random_save_dirs = build_random_save_dirs(
            save_root=args.save_dir,
            ckpt_path=args.ckpt,
            train_args=train_args,
            cond_source=args.cond_source,
            sampler_tag=sampler_tag,
        )
        save_dir = random_save_dirs["root"]
        for d in random_save_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        cf_save_dirs = None
    elif args.mode == "cf":
        sampler_tag = get_cf_sampler_tag(
            ode_method=args.ode_method,
            ode_atol=args.ode_atol,
            ode_rtol=args.ode_rtol,
            ode_steps=args.ode_steps,
            sample_steps=args.sample_steps,
            cfg_mode=args.cfg_mode,
            cfg_scale=args.cfg_scale,
            null_keys=args.null_keys,
        )
        cf_save_dirs = build_cf_save_dirs(
            save_root=args.save_dir,
            ckpt_path=args.ckpt,
            train_args=train_args,
            do_key=args.do_key,
            do_mode=args.do_mode,
            sampler_tag=sampler_tag,
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
        "sample_steps": args.sample_steps,
        "cfg_mode": args.cfg_mode,
        "cfg_scale": args.cfg_scale,
        "null_keys": args.null_keys,
        "ode_method": args.ode_method if args.mode == "cf" else None,
        "ode_atol": args.ode_atol if args.mode == "cf" and args.ode_steps is None else None,
        "ode_rtol": args.ode_rtol if args.mode == "cf" and args.ode_steps is None else None,
        "ode_steps": args.ode_steps if args.mode == "cf" else None,
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
                sample_steps=args.sample_steps,
                sample_args=sample_args,
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
                class_schema=class_schema,
            )

            noise = invert_to_noise_ode(
                model=model,
                x=x_src,
                pa_src=pa_src,
                ode_method=args.ode_method,
                ode_atol=args.ode_atol,
                ode_rtol=args.ode_rtol,
                ode_steps=args.ode_steps,
                sample_args=ode_sample_args,
            )

            x_cf = generate_random_from_noise(
                model=model,
                noise=noise,
                pa=pa_cf,
                sample_steps=args.sample_steps,
                sample_args=sample_args,
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