import builtins
import copy
import datetime
import inspect
import os
import random
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torchvision.utils import make_grid


class ModelEMA:
    def __init__(self, params: Iterable[torch.nn.Parameter], rate: float = 0.999):
        self.rate = rate
        self.params = list(params)  # reference
        self.ema_params = [
            copy.deepcopy(p).detach().requires_grad_(False) for p in self.params
        ]

    @torch.no_grad()
    def update(self):
        for ema_p, p in zip(self.ema_params, self.params, strict=True):
            ema_p.mul_(self.rate).add_(p, alpha=1 - self.rate)

    @torch.no_grad()
    def apply(self):
        self.stored_params = [p.clone() for p in self.params]
        for p, ema_p in zip(self.params, self.ema_params, strict=True):
            p.copy_(ema_p)

    @torch.no_grad()
    def restore(self):
        assert self.stored_params is not None
        for p, stored_p in zip(self.params, self.stored_params, strict=True):
            p.copy_(stored_p)
        del self.stored_params

    def state_dict(self) -> list[torch.Tensor]:
        return [p.clone() for p in self.ema_params]

    def load_state_dict(self, state: list[torch.Tensor]) -> None:
        for ema_p, saved_p in zip(self.ema_params, state, strict=True):
            ema_p.copy_(saved_p)


def seed_all(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:  # single GPU
        local_rank = rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    init_pg_kwargs = dict(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
    )
    # `device_id` was added in newer torch releases; keep compatibility with 2.1.x.
    if "device_id" in inspect.signature(torch.distributed.init_process_group).parameters:
        init_pg_kwargs["device_id"] = local_rank
    torch.distributed.init_process_group(**init_pg_kwargs)
    print(
        f"WORLD_SIZE: {world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}, "
        + f"MASTER: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )
    if rank != 0:
        # NOTE: hack to print only on master gpu (0)
        def print_pass(*args):
            pass

        builtins.print = print_pass
    return local_rank, rank, world_size


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    while True:
        if hasattr(model, "_orig_mod"):  # compiled wrapper
            model = model._orig_mod
            continue
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            continue
        return model


def get_pretrained_flux2vae() -> torch.nn.Module:
    from diffusers.models import AutoencoderKLFlux2

    print("\nLoading FLUX.2 vae...")
    model = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    # NOTE: monkey patch to support greyscale
    orig_encode = model.encode
    model.encode = lambda x: orig_encode(x.repeat(1, 3, 1, 1)).latent_dist.sample()
    orig_decode = model.decode
    model.decode = lambda z: orig_decode(z).sample.mean(dim=1, keepdim=True)
    model.register_buffer("mean", torch.tensor(-0.061467))
    model.register_buffer("std", torch.tensor(1.633637))
    model.requires_grad_(False)
    model.eval()
    return model


@torch.inference_mode()
def get_mc_stats(
    samples: torch.Tensor | list[torch.Tensor], prefix: str | None = None
) -> dict[str, torch.Tensor]:
    x = samples if torch.is_tensor(samples) else torch.stack(samples)  # [K, ...]
    x = x.detach().float()
    x = x.unsqueeze(0) if x.ndim == 0 else x
    mc_samples = int(x.shape[0])
    assert mc_samples >= 1
    mean = x.mean(dim=0)
    std = x.std(dim=0) if mc_samples > 1 else torch.zeros_like(mean)
    out = {"mean": mean, "std": std, "se": std / (mc_samples**0.5)}
    return {f"{prefix}_{k}": v for k, v in out.items()} if prefix else out


@torch.inference_mode()
def plot_samples(
    u: Tensor,
    pa: dict[str, Tensor],
    model: torch.nn.Module,
    vae: torch.nn.Module | None = None,
    steps: int | None = None,
    file_path: str | None = None,
    **kwargs,
) -> None:
    if steps is not None:
        kwargs = dict(  # default sched solver
            method="euler",
            t=model.schedule(torch.linspace(0.0, 1.0, steps).to(u.device), model.alpha),
        )
    x = model.ode_solve(x=u, pa=pa, **kwargs)[-1]  # [-1,1]; type: ignore
    if isinstance(vae, torch.nn.Module):
        # NOTE: trying cpu inference to prevent VRAM spike
        # vae.to("cpu")
        vae_dev = next(vae.parameters()).device
        x = x.to(vae_dev)
        z = x * vae.std + vae.mean
        with torch.autocast(z.device.type, dtype=torch.bfloat16):
            x = vae.decode(z)
    x = (x.float().clamp(min=-1, max=1) + 1) * 0.5  # [0,1]
    c, s = u.shape[0], 8
    plt.figure(figsize=(c * s, s * c // 6))
    plt.axis("off")
    plt.imshow(make_grid(x.cpu(), padding=0, nrow=6).permute(1, 2, 0))
    plt.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def value_to_name(pa: dict[str, Tensor]) -> dict[str, list[str]]:
    idx_to_name = {
        "dataset": ["nan", "MIC", "REX", "CXP", "PAD", "NIH", "BRX", "VIN"],
        "view": ["nan", "AP", "PA", "LATERAL"],
        "race": ["nan", "Asian", "Black", "White"],
        "sex": ["nan", "Male", "Female"],
        "disease": ["nan", "N", "P"],
        # Add dataset-specific mappings here if you want prettier labels:
        # "density": ["nan", "A", "B", "C", "D"],
        # "cview": [...],
        # "scanner": [...],
    }

    out = {k: [] for k in pa.keys()}

    def _to_scalar(x: Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}")
        return x.item()

    def _is_nan(x) -> bool:
        return isinstance(x, float) and np.isnan(x)

    for k, v in pa.items():
        for i in range(v.shape[0]):  # batch size
            val = _to_scalar(v[i])

            if k == "age":
                if _is_nan(val):
                    out[k].append("nan")
                else:
                    out[k].append(str(int(round(float(val) * 100))))
                continue

            # Known categorical keys with explicit name maps
            if k in idx_to_name:
                if _is_nan(val):
                    out[k].append("nan")
                else:
                    idx = int(round(float(val)))
                    names = idx_to_name[k]
                    if 0 <= idx < len(names):
                        out[k].append(names[idx])
                    else:
                        out[k].append(str(idx))
                continue

            # Fallback for unknown keys:
            # 1) If value looks like an integer class, show the integer.
            # 2) Otherwise show a rounded float.
            if _is_nan(val):
                out[k].append("nan")
            else:
                fval = float(val)
                if abs(fval - round(fval)) < 1e-6:
                    out[k].append(str(int(round(fval))))
                else:
                    out[k].append(f"{fval:.3f}")

    return out


@torch.inference_mode()
def plot_counterfactuals(
    x: Tensor,
    pa: dict[str, Tensor],
    model: torch.nn.Module,
    vae: torch.nn.Module | None = None,
    steps: int | None = None,
    file_path: str | None = None,
    **kwargs,
) -> None:
    import numbers
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from data.datasets import CLASS_SCHEMA

    bs = x.shape[0]
    do_pa = {k: v.clone() for k, v in pa.items()}
    do_i: list[str] = []
    pa_keys = list(pa.keys())

    if len(pa_keys) == 0:
        raise ValueError("No parent keys found for counterfactual plotting.")

    # Build batch of random interventions
    for i in range(bs):
        k = pa_keys[i % len(pa_keys)]  # wraps around
        do_i.append(k)

        y = do_pa[k][i].clone()
        schema_val = CLASS_SCHEMA.get(k, None)
        is_categorical = isinstance(schema_val, numbers.Integral)

        if is_categorical:
            num_classes = int(schema_val)

            # If there is no alternative class, keep original
            if num_classes <= 1:
                do_y = y.clone()
            else:
                # Sample from [0, num_classes)
                do_y = torch.randint(0, num_classes, y.shape, device=y.device)

                # Re-sample until it differs from the original label
                max_tries = 32
                tries = 0
                while (do_y == y).any() and tries < max_tries:
                    do_y = torch.randint(0, num_classes, y.shape, device=y.device)
                    tries += 1

                # Fallback in pathological cases
                if (do_y == y).any():
                    do_y = (y.long() + 1) % num_classes
                    do_y = do_y.to(y.dtype)

            do_pa[k][i] = do_y
        else:
            # Continuous variable, e.g. normalized age
            do_pa[k][i] = torch.rand_like(do_pa[k][i])

    sched_solver = steps is not None
    if sched_solver:
        kwargs = dict(
            method="euler",
            t=model.schedule(torch.linspace(1.0, 0.0, steps, device=x.device), model.alpha),
        )
    else:
        kwargs["t"] = torch.tensor([1.0, 0.0], device=x.device)

    u = model.ode_solve(x, pa=pa, **kwargs)[-1]  # [-1,1]; type: ignore 

    if sched_solver:
        kwargs["t"] = model.schedule(torch.linspace(0.0, 1.0, steps, device=x.device), model.alpha)
    else:
        kwargs["t"] = torch.tensor([0.0, 1.0], device=x.device)

    cf_x = model.ode_solve(u, pa=do_pa, **kwargs)[-1]  # [-1,1]; type: ignore

    if isinstance(vae, torch.nn.Module):
        t0 = time.time()
        vae_dev = next(vae.parameters()).device
        x, cf_x = x.to(vae_dev), cf_x.to(vae_dev)
        z = x * vae.std + vae.mean
        cf_z = cf_x * vae.std + vae.mean

        if z.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(z.device.type, dtype=torch.bfloat16):
                out = vae.decode(torch.cat([z, cf_z], dim=0))
        else:
            out = vae.decode(torch.cat([z, cf_z], dim=0))

        rec, cf_x = out.chunk(2, dim=0)
        x = (rec.float().clamp(min=-1, max=1) + 1) * 0.5 # [0,1]
        cf_x = (cf_x.float().clamp(min=-1, max=1) + 1) * 0.5 # [0,1]

        eval_elapsed = time.time() - t0
        print(f"VAE inference time elapsed: {eval_elapsed:.2f}s")
    else:
        x = (x.float().clamp(min=-1, max=1) + 1) * 0.5 # [0,1]
        cf_x = (cf_x.float().clamp(min=-1, max=1) + 1) * 0.5 # [0,1]

    x, cf_x = x[:, 0, ...].cpu(), cf_x[:, 0, ...].cpu()  # (b, h, w)
    effect = (cf_x - x) * 255
    imgs = [x, cf_x, effect]

    _pa, _do_pa = value_to_name(pa), value_to_name(do_pa)

    fs = 20
    plt.rcParams["font.size"] = fs
    c, s = 6, 8
    nrows = 3 * int(np.ceil(bs / c))
    fig, axes = plt.subplots(nrows, c, figsize=(s * c, s * nrows + s))

    if nrows == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    count = [0, 0, 0]
    for idx, ax in enumerate(axes.flatten()):
        row = idx // c
        img_group = row % 3

        if count[img_group] < imgs[img_group].shape[0]:
            sample_idx = count[img_group]
            k = do_i[sample_idx]
            img = imgs[img_group][sample_idx].squeeze()

            if img_group == 0:  # observation
                ax.imshow(img, cmap="gray")
                ax.set_xlabel(f"{k}: {_pa[k][sample_idx]}", labelpad=8)

            elif img_group == 1:  # counterfactual
                ax.imshow(img, cmap="gray")
                ax.set_xlabel(f"do({k}={_do_pa[k][sample_idx]})", labelpad=8)

            else:  # effect
                amax = float(img.abs().max())
                if amax == 0:
                    amax = 1.0
                eff = ax.imshow(img, cmap="RdBu_r", vmin=-amax, vmax=amax)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="3%", pad=0.08)
                cbar = fig.colorbar(eff, cax=cax, orientation="horizontal")
                cbar.outline.set_visible(False)  # type: ignore
                cbar.ax.tick_params(labelsize=fs // 2)
                ticks = np.linspace(np.ceil(-amax), np.floor(amax), 5)
                cbar.set_ticks(np.round(ticks))

            count[img_group] += 1

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(wspace=0.055, hspace=0.01)
    if file_path is not None:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.inference_mode()
def save_plots(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    steps: int,
    save_path: str,
    vae: torch.nn.Module | None = None,
) -> None:
    base = unwrap(model)
    device = next(base.parameters()).device
    num_samples = min(6 * round(batch_size / 6), 12)
    dataloader = torch.utils.data.DataLoader(dataset, num_samples, shuffle=True)
    batch = next(iter(dataloader))
    x = batch["x"].float().to(device) * 2 - 1
    pa = {k: v.to(device) for k, v in batch["pa"].items()}
    kwargs = dict(pa=pa, model=base, vae=vae, steps=steps)
    plot_samples(torch.randn_like(x), file_path=save_path + ".pdf", **kwargs)
    plot_counterfactuals(x, file_path=save_path + "_cf.pdf", **kwargs)


@torch.inference_mode()
def plot_samples_mf(
    noise: Tensor,
    pa: dict[str, Tensor],
    model: torch.nn.Module,
    vae: torch.nn.Module | None = None,
    steps: int = 1,
    file_path: str | None = None,
) -> None:
    x = model.sample(noise, steps=steps, pa=pa)
    if isinstance(vae, torch.nn.Module):
        vae_dev = next(vae.parameters()).device
        x = x.to(vae_dev)
        z = x * vae.std + vae.mean
        with torch.autocast(z.device.type, dtype=torch.bfloat16):
            x = vae.decode(z)
    x = (x.float().clamp(-1, 1) + 1) * 0.5
    c, s = noise.shape[0], 8
    plt.figure(figsize=(c * s, s * c // 6))
    plt.axis("off")
    plt.imshow(make_grid(x.cpu(), padding=0, nrow=6).permute(1, 2, 0))
    plt.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.inference_mode()
def plot_counterfactuals_mf(
    x: Tensor,
    pa: dict[str, Tensor],
    model: torch.nn.Module,
    vae: torch.nn.Module | None = None,
    steps: int = 1,                  # MeanFlow sampling steps
    file_path: str | None = None,
    **kwargs,
) -> None:
    import numbers
    import time
    from data.datasets import CLASS_SCHEMA

    bs = x.shape[0]
    do_pa = {k: v.clone() for k, v in pa.items()}
    do_i: list[str] = []
    pa_keys = list(pa.keys())

    if len(pa_keys) == 0:
        raise ValueError("No parent keys found for counterfactual plotting.")

    for i in range(bs):
        k = pa_keys[i % len(pa_keys)]
        do_i.append(k)
        y = do_pa[k][i].clone()
        schema_val = CLASS_SCHEMA.get(k, None)
        is_categorical = isinstance(schema_val, numbers.Integral)

        if is_categorical:
            num_classes = int(schema_val)
            if num_classes <= 1:
                do_y = y.clone()
            else:
                do_y = torch.randint(0, num_classes, y.shape, device=y.device)
                max_tries = 32
                tries = 0
                while (do_y == y).any() and tries < max_tries:
                    do_y = torch.randint(0, num_classes, y.shape, device=y.device)
                    tries += 1
                if (do_y == y).any():
                    do_y = (y.long() + 1) % num_classes
                    do_y = do_y.to(y.dtype)
            do_pa[k][i] = do_y
        else:
            do_pa[k][i] = torch.rand_like(do_pa[k][i])

    # Separate MeanFlow sample args from ODE kwargs
    inv_sample_args = kwargs.pop("inv_sample_args", None)
    gen_sample_args = kwargs.pop("gen_sample_args", None)

    # Use a separate inversion step count if provided; otherwise default to 150
    inv_steps = kwargs.pop("inv_steps", 150)

    # Remaining kwargs go to model.ode_solve(..., **ode_kwargs)
    ode_kwargs = dict(kwargs)

    sched_solver = inv_steps is not None
    if sched_solver:
        ode_kwargs.setdefault("method", "euler")
        ode_kwargs.setdefault(
            "t",
            torch.linspace(0.0, 1.0, inv_steps, device=x.device),
        )
    else:
        ode_kwargs.setdefault("t", torch.tensor([0.0, 1.0], device=x.device))

    # Encode data -> noise via ODE inversion on the diagonal field r=t
    noise = model.ode_solve(
        x,
        pa=pa,
        sample_args=inv_sample_args,
        **ode_kwargs,
    )[-1]

    # Generate CF from the same recovered noise using native MeanFlow sampling
    cf_x = model.sample(
        noise,
        steps=steps,
        pa=do_pa,
        sample_args=gen_sample_args,
    )

    if isinstance(vae, torch.nn.Module):
        t0 = time.time()
        vae_dev = next(vae.parameters()).device
        x = x.to(vae_dev)
        cf_x = cf_x.to(vae_dev)

        z = x * vae.std + vae.mean
        cf_z = cf_x * vae.std + vae.mean

        if z.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(z.device.type, dtype=torch.bfloat16):
                out = vae.decode(torch.cat([z, cf_z], dim=0))
        else:
            out = vae.decode(torch.cat([z, cf_z], dim=0))

        rec, cf_x = out.chunk(2, dim=0)
        x = (rec.float().clamp(min=-1, max=1) + 1) * 0.5
        cf_x = (cf_x.float().clamp(min=-1, max=1) + 1) * 0.5
        print(f"VAE inference time elapsed: {time.time() - t0:.2f}s")
    else:
        x = (x.float().clamp(min=-1, max=1) + 1) * 0.5
        cf_x = (cf_x.float().clamp(min=-1, max=1) + 1) * 0.5

    x = x[:, 0, ...].cpu()
    cf_x = cf_x[:, 0, ...].cpu()
    effect = (cf_x - x) * 255
    imgs = [x, cf_x, effect]

    _pa, _do_pa = value_to_name(pa), value_to_name(do_pa)

    fs = 20
    plt.rcParams["font.size"] = fs
    c, s = 6, 8
    nrows = 3 * int(np.ceil(bs / c))
    fig, axes = plt.subplots(nrows, c, figsize=(s * c, s * nrows + s))

    if nrows == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    count = [0, 0, 0]
    for idx, ax in enumerate(axes.flatten()):
        row = idx // c
        img_group = row % 3

        if count[img_group] < imgs[img_group].shape[0]:
            sample_idx = count[img_group]
            k = do_i[sample_idx]
            img = imgs[img_group][sample_idx].squeeze()

            if img_group == 0:
                ax.imshow(img, cmap="gray")
                ax.set_xlabel(f"{k}: {_pa[k][sample_idx]}", labelpad=8)

            elif img_group == 1:
                ax.imshow(img, cmap="gray")
                ax.set_xlabel(f"do({k}={_do_pa[k][sample_idx]})", labelpad=8)

            else:
                amax = float(img.abs().max())
                if amax == 0:
                    amax = 1.0
                eff = ax.imshow(img, cmap="RdBu_r", vmin=-amax, vmax=amax)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="3%", pad=0.08)
                cbar = fig.colorbar(eff, cax=cax, orientation="horizontal")
                cbar.outline.set_visible(False)
                cbar.ax.tick_params(labelsize=fs // 2)
                ticks = np.linspace(np.ceil(-amax), np.floor(amax), 5)
                cbar.set_ticks(np.round(ticks))

            count[img_group] += 1

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(wspace=0.055, hspace=0.01)
    if file_path is not None:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    
@torch.inference_mode()
def save_plots_mf(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    steps: int,
    save_path: str,
    vae: torch.nn.Module | None = None,
) -> None:
    base = unwrap(model)
    device = next(base.parameters()).device
    num_samples = min(6 * round(batch_size / 6), 12)
    dataloader = torch.utils.data.DataLoader(dataset, num_samples, shuffle=True)
    batch = next(iter(dataloader))
    x = batch["x"].float().to(device) * 2 - 1 # [0,1] -> [-1,1]
    pa = {k: v.to(device) for k, v in batch["pa"].items()}
    plot_samples_mf(torch.randn_like(x), pa=pa, model=base, vae=vae, steps=steps, file_path=save_path + ".pdf")
    plot_counterfactuals_mf(x, pa=pa, model=base, vae=vae, steps=steps, file_path=save_path + "_cf.pdf")


@torch.inference_mode()
def save_reconstructions(
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    file_path: str | None = None,
) -> None:
    base = unwrap(model)
    device = next(base.parameters()).device
    num_samples = min(6 * round(batch_size / 6), 18)
    dataloader = torch.utils.data.DataLoader(dataset, num_samples, shuffle=True)
    x = next(iter(dataloader))["x"].float().to(device) * 2 - 1
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        rec = base.decode(base.encode(x))  # for vae only
        interleaved = torch.stack((x, rec), dim=1).reshape(-1, *x.shape[1:])
        interleaved = (interleaved.clamp(min=-1, max=1) + 1) * 0.5  # [0, 1]
    c, s = x.shape[0], 8
    plt.figure(figsize=(c * s, 2 * s * c // 6))
    plt.axis("off")
    plt.imshow(make_grid(interleaved.cpu(), padding=0, nrow=6).permute(1, 2, 0))
    plt.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
