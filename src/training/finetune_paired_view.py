#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from src.data_handle.embed import DatasetConfig, get_embed
from src.models.embedder import (
    CondEmbedderConfig,
    GlobalCondEmbedder,
    PerAttrCondEmbedder,
    infer_parent_dims_from_batch,
)
from src.utils import ModelEMA, seed_all, setup_distributed, unwrap


# ============================================================
# Dataset
# ============================================================

def unwrap_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset


def build_cache_idx_lookup(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base = unwrap_dataset(dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        df = base.df.iloc[indices].reset_index(drop=True)
    else:
        df = dataset.df.reset_index(drop=True)

    if "cache_idx" not in df.columns:
        raise ValueError("Dataset dataframe has no cache_idx column.")

    return {int(cache_idx): i for i, cache_idx in enumerate(df["cache_idx"].astype(int))}


def get_sample_by_cache_idx(dataset, lookup, cache_idx):
    cache_idx = int(cache_idx)

    if cache_idx not in lookup:
        raise KeyError(f"cache_idx={cache_idx} not found in dataset.")

    return dataset[lookup[cache_idx]]


class PairedViewDataset(Dataset):
    """
    Each real CC/MLO pair is used in both directions:

        CC  -> MLO
        MLO -> CC

    pa_cf keeps the source attributes except for cf_keys, which are
    replaced by the corresponding target-view attributes.
    """

    def __init__(self, dataset, pair_df, cf_keys=("view",)):
        self.dataset = dataset
        self.pairs = pair_df.reset_index(drop=True)
        self.lookup = build_cache_idx_lookup(dataset)
        self.cf_keys = set(cf_keys)

    def __len__(self):
        return 2 * len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx // 2]

        cc = get_sample_by_cache_idx(self.dataset, self.lookup, row["cc_cache_idx"])
        mlo = get_sample_by_cache_idx(self.dataset, self.lookup, row["mlo_cache_idx"])

        src, target = (cc, mlo) if idx % 2 == 0 else (mlo, cc)

        pa_src = {k: v.clone() for k, v in src["pa"].items()}
        pa_target = {k: v.clone() for k, v in target["pa"].items()}

        pa_cf = {
            k: pa_target[k].clone() if k in self.cf_keys else pa_src[k].clone()
            for k in pa_src
        }

        return {
            "x_src": src["x"].float(),
            "x_target": target["x"].float(),
            "pa_src": pa_src,
            "pa_cf": pa_cf,
        }


# ============================================================
# Model construction
# ============================================================

def select_amp_dtype(device):
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def build_flow_model(train_args, datasets, device, amp_dtype):
    from src.flows.flow import Flow

    sample = datasets["train"][0]
    pa_batch = {k: v.unsqueeze(0) for k, v in sample["pa"].items()}
    parent_dims = infer_parent_dims_from_batch(pa_batch, train_args.parents)

    cond_embedder = None

    if train_args.cond_embedder != "none":
        embedder_config = CondEmbedderConfig(
            parents=train_args.parents,
            parent_dims=parent_dims,
            cond_embed_dim=train_args.cond_embed_dim,
        )

        if train_args.cond_embedder == "per_attr":
            cond_embedder = PerAttrCondEmbedder(embedder_config)
        elif train_args.cond_embedder == "global":
            cond_embedder = GlobalCondEmbedder(embedder_config)
        else:
            raise ValueError(f"Unknown cond_embedder: {train_args.cond_embedder}")

    if train_args.model == "dit":
        from src.models.DiT import DiT

        forward_nn = DiT(
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            patch_size=train_args.patch_size,
            in_channels=train_args.img_channels,
            hidden_size=train_args.hidden_size,
            depth=train_args.depth,
            num_heads=train_args.num_heads,
            mlp_ratio=train_args.mlp_ratio,
            cond_embed_dim=train_args.cond_embed_dim if train_args.cond_embedder != "none" else 0,
            grad_checkpointing=train_args.grad_checkpointing,
        )

    elif train_args.model == "unet":
        from src.flows.flow import BlockConfig, UNetConfig
        from src.models.unet import UNet

        unet_config = UNetConfig(
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            img_channels=train_args.img_channels,
            cond_embed_dim=train_args.cond_embed_dim,
            model_channels=train_args.model_channels,
            channel_mult=tuple(train_args.channel_mult),
            channel_mult_time=train_args.channel_mult_time,
            channel_mult_emb=train_args.channel_mult_emb,
            num_blocks=train_args.num_blocks,
            attn_resolutions=tuple(train_args.attn_resolutions),
            label_balance=train_args.label_balance,
            concat_balance=train_args.concat_balance,
        )

        block_config = BlockConfig(
            resample_filter=tuple(train_args.resample_filter),
            channels_per_head=train_args.channels_per_head,
            dropout=train_args.dropout,
            res_balance=train_args.res_balance,
            attn_balance=train_args.attn_balance,
            clip_act=train_args.clip_act,
        )

        forward_nn = UNet(**vars(unet_config), **vars(block_config))

    else:
        raise ValueError(f"Unknown model: {train_args.model}")

    model = Flow(
        forward_nn=forward_nn,
        cond_embedder=cond_embedder,
        sigma=train_args.sigma,
        alpha=train_args.alpha,
        p_uncond=train_args.p_uncond,
        amp_dtype=amp_dtype,
    )

    return model.to(device)


# ============================================================
# Fine-tuning objective
# ============================================================

class PairedFineTuneModel(nn.Module):
    """
    Original FM:
        u ~ N(0, I)
        x_t = (1-t) * u + t * x_src

    Supervised one-step branch:
        with probability p_cf:
            condition = pa_cf
            target    = x_target

        otherwise:
            condition = pa_src
            target    = x_src

        x_pred = x_t + (1-t) * v_theta(x_t, t, condition)
        L_sft  = d(x_pred, target)

    Total:
        L = L_FM + lambda_pair * L_sft

    Note:
        The reconstruction part of L_sft is closely related to the
        original flow-matching objective when sigma=0.

        For the same u and t, reconstruction endpoint MSE is equivalent
        to the FM velocity MSE up to a factor of (1-t)^2.

        Therefore, when L_FM is retained, the reconstruction branch may
        be partly redundant. It is kept for now as additional rehearsal
        during paired fine-tuning.

        A future simplification is to set p_cf=1.0 so that L_sft contains
        only paired counterfactual supervision, while L_FM preserves the
        original flow-matching behaviour.
    """

    def __init__(
        self,
        flow,
        *,
        alpha=1.0,
        lambda_pair=1.0,
        p_cf=0.8,
        t_max=0.8,
        endpoint_loss="mse",
    ):
        super().__init__()

        if not 0.0 <= p_cf <= 1.0:
            raise ValueError("p_cf must be in [0, 1].")

        if not 0.0 < t_max < 1.0:
            raise ValueError("t_max must be in (0, 1).")

        self.flow = flow
        self.alpha = float(alpha)
        self.lambda_pair = float(lambda_pair)
        self.p_cf = float(p_cf)
        self.t_max = float(t_max)
        self.endpoint_loss = endpoint_loss

    def sample_t(self, batch_size, device, g=None):
        t = torch.rand(batch_size, device=device, generator=g) * self.t_max

        if self.alpha != 1.0:
            t = t / (self.alpha - (self.alpha - 1.0) * t)

        return t

    def predict_velocity(self, x_t, t, pa):
        cond_emb = None if self.flow.cond_embedder is None else self.flow.cond_embedder(pa)
        amp_dtype = getattr(self.flow, "amp_dtype", None)

        if amp_dtype is not None:
            with torch.autocast(x_t.device.type, dtype=amp_dtype):
                v = self.flow.forward_nn(x_t, t, cond_emb)
            return v.float()

        return self.flow.forward_nn(x_t, t, cond_emb)

    @staticmethod
    def mix_pa(pa_src, pa_cf, is_cf):
        pa_goal = {}

        for key in pa_src:
            mask = is_cf.reshape(-1, *([1] * (pa_src[key].ndim - 1)))
            pa_goal[key] = torch.where(mask, pa_cf[key], pa_src[key])

        return pa_goal

    def endpoint_error(self, x_pred, x_goal):
        if self.endpoint_loss == "mse":
            return (x_pred - x_goal).square().flatten(1).mean(1)

        if self.endpoint_loss == "l1":
            return (x_pred - x_goal).abs().flatten(1).mean(1)

        raise ValueError(f"Unknown endpoint loss: {self.endpoint_loss}")

    def forward(self, x_src, x_target, pa_src, pa_cf, g=None):
        # 1. Original flow-matching objective.
        #
        # This preserves the original generative flow behaviour during
        # paired fine-tuning.
        fm_out = self.flow(x_src, pa_src, g)
        loss_fm = fm_out[0] if isinstance(fm_out, tuple) else fm_out

        bs = x_src.shape[0]

        # 2. Draw Gaussian noise and construct source-derived x_t.
        u = torch.randn(
            x_src.shape,
            device=x_src.device,
            dtype=x_src.dtype,
            generator=g,
        )

        t = self.sample_t(bs, x_src.device, g)
        t_bc = t.reshape(-1, *([1] * (x_src.ndim - 1)))
        x_t = (1.0 - t_bc) * u + t_bc * x_src

        # 3. Choose CF supervision or reconstruction per sample.
        #
        # The reconstruction branch provides additional source-condition
        # rehearsal. With sigma=0, it is closely related to the original
        # flow-matching objective already included in loss_fm, so it may
        # be partly redundant.
        #
        # Keep it for now. A future simplification is to set p_cf=1.0,
        # making the endpoint loss purely paired CF supervision.
        is_cf = torch.rand(bs, device=x_src.device, generator=g) < self.p_cf
        x_mask = is_cf.reshape(-1, *([1] * (x_src.ndim - 1)))

        x_goal = torch.where(x_mask, x_target, x_src)
        pa_goal = self.mix_pa(pa_src, pa_cf, is_cf)

        # 4. Predict velocity using selected conditioning.
        v = self.predict_velocity(x_t, t, pa_goal)

        # 5. One-step endpoint prediction.
        x_pred = x_t + (1.0 - t_bc) * v

        # 6. Endpoint supervision.
        # Scale the endpoint error by (1-t)^2 to match the original flow-matching objective when sigma=0.
        errors = self.endpoint_error(x_pred, x_goal)/ (1.0 - t).clamp_min(1e-4).square()
        loss_endpoint = errors.mean()

        loss_cf = (
            errors[is_cf].mean()
            if is_cf.any()
            else torch.zeros((), device=x_src.device)
        )

        loss_recon = (
            errors[~is_cf].mean()
            if (~is_cf).any()
            else torch.zeros((), device=x_src.device)
        )

        # 7. Final objective.
        #
        # L_FM preserves the original flow objective.
        # L_endpoint adds paired-view supervision.
        #
        # The reconstruction subset of L_endpoint partially overlaps
        # with L_FM and may be removed in a future ablation.
        loss = loss_fm + self.lambda_pair * loss_endpoint

        return loss, {
            "loss": loss.detach(),
            "fm": loss_fm.detach(),
            "endpoint": loss_endpoint.detach(),
            "cf": loss_cf.detach(),
            "recon": loss_recon.detach(),
            "cf_frac": is_cf.float().mean().detach(),
            "t": t.mean().detach(),
        }


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(model, dataloader, device, seed=12345):
    model.eval()

    total = torch.zeros(7, device=device)
    n = torch.tensor(0.0, device=device)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    for batch in dataloader:
        x_src = batch["x_src"].to(device, non_blocking=True)
        x_target = batch["x_target"].to(device, non_blocking=True)

        pa_src = {
            k: v.to(device, non_blocking=True)
            for k, v in batch["pa_src"].items()
        }

        pa_cf = {
            k: v.to(device, non_blocking=True)
            for k, v in batch["pa_cf"].items()
        }

        _, stats = model(
            x_src,
            x_target,
            pa_src,
            pa_cf,
            g=g,
        )

        bs = x_src.shape[0]

        values = torch.stack([
            stats["loss"],
            stats["fm"],
            stats["endpoint"],
            stats["cf"],
            stats["recon"],
            stats["cf_frac"],
            stats["t"],
        ])

        total += values * bs
        n += bs

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
        dist.all_reduce(n)

    return (total / n).tolist()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--pair_csv", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)

    # Supervised fine-tuning
    parser.add_argument("--lambda_pair", type=float, default=1.0)
    parser.add_argument("--p_cf", type=float, default=0.8)
    parser.add_argument("--endpoint_loss", choices=["mse", "l1"], default="mse")
    parser.add_argument("--t_max", type=float, default=0.8)
    parser.add_argument("--cf_keys", nargs="+", default=["view"])

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--dist", action="store_true")

    args = parser.parse_args()

    if not 0.0 <= args.p_cf <= 1.0:
        parser.error("--p_cf must be in [0, 1].")

    if not 0.0 < args.t_max < 1.0:
        parser.error("--t_max must be in (0, 1).")

    # ========================================================
    # Distributed setup
    # ========================================================

    if args.dist:
        device, rank, world_size = setup_distributed()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rank, world_size = 0, 1

    is_dist = args.dist and dist.is_available() and dist.is_initialized()

    seed_all(args.seed + rank, False)
    amp_dtype = select_amp_dtype(device)

    # ========================================================
    # Load pretrained checkpoint
    # ========================================================

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])

    if abs(float(train_args.sigma)) > 1e-12:
        raise NotImplementedError(
            "The one-step endpoint formulation currently assumes sigma=0, "
            f"but checkpoint has sigma={train_args.sigma}."
        )

    # ========================================================
    # Load cached latent dataset
    # ========================================================

    datasets = get_embed(
        DatasetConfig(
            data_dir=train_args.data_dir,
            split_dir=train_args.split_dir,
            cache_dir=train_args.cache_dir,
            parents=train_args.parents,
            img_height=train_args.img_height,
            img_width=train_args.img_width,
            img_channels=train_args.img_channels,
            vae_ckpt=train_args.vae_ckpt,
        )
    )

    # ========================================================
    # Load paired CC/MLO CSV
    # ========================================================

    pair_csv = (
        Path(train_args.cache_dir) / "pair_csv_latent.csv"
        if args.pair_csv is None
        else Path(args.pair_csv)
    )

    pairs = pd.read_csv(pair_csv, low_memory=False)

    required = {"split", "cc_cache_idx", "mlo_cache_idx"}
    missing = required - set(pairs.columns)

    if missing:
        raise ValueError(f"Pair CSV missing columns: {sorted(missing)}")

    train_pairs = pairs.loc[pairs["split"] == "train"].reset_index(drop=True)
    valid_pairs = pairs.loc[pairs["split"] == "valid"].reset_index(drop=True)

    if len(train_pairs) == 0:
        raise ValueError("No training pairs found.")

    train_dataset = PairedViewDataset(
        datasets["train"],
        train_pairs,
        cf_keys=args.cf_keys,
    )

    valid_dataset = (
        PairedViewDataset(
            datasets["valid"],
            valid_pairs,
            cf_keys=args.cf_keys,
        )
        if len(valid_pairs) > 0
        else None
    )

    # ========================================================
    # Data loaders
    # ========================================================

    train_sampler = (
        DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=args.seed,
        )
        if is_dist
        else None
    )

    valid_sampler = (
        DistributedSampler(
            valid_dataset,
            shuffle=False,
        )
        if is_dist and valid_dataset is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = None

    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.bs,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=True,
        )

    # ========================================================
    # Build model
    # ========================================================

    flow = build_flow_model(
        train_args,
        datasets,
        device,
        amp_dtype,
    )

    flow.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = PairedFineTuneModel(
        flow,
        alpha=float(train_args.alpha),
        lambda_pair=args.lambda_pair,
        p_cf=args.p_cf,
        t_max=args.t_max,
        endpoint_loss=args.endpoint_loss,
    ).to(device)

    # ========================================================
    # EMA
    # ========================================================

    ema = ModelEMA(
        model.parameters(),
        rate=args.ema_rate,
    )

    if ckpt.get("ema_state") is not None:
        try:
            ema.load_state_dict(ckpt["ema_state"])

            if rank == 0:
                print("Loaded EMA state from pretrained checkpoint.")

        except Exception:
            if rank == 0:
                print("Could not load pretrained EMA state; starting new EMA.")

    # ========================================================
    # DDP
    # ========================================================

    if is_dist:
        model = DistributedDataParallel(
            model,
            device_ids=[device],
            bucket_cap_mb=150,
        )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    step = 0
    best_endpoint = float("inf")

    # ========================================================
    # Print config
    # ========================================================

    if rank == 0:
        print("=" * 80)
        print("Paired-view supervised fine-tuning")
        print("=" * 80)
        print("Checkpoint        :", args.ckpt)
        print("Pair CSV          :", pair_csv)
        print("Train pairs       :", len(train_pairs))
        print("Train directions  :", len(train_dataset))
        print("Valid pairs       :", len(valid_pairs))
        print("CF keys           :", args.cf_keys)
        print("p_cf              :", args.p_cf)
        print("p_reconstruction  :", 1.0 - args.p_cf)
        print("lambda_pair       :", args.lambda_pair)
        print("endpoint loss     :", args.endpoint_loss)
        print("t_max             :", args.t_max)
        print("learning rate     :", args.lr)
        print("=" * 80)

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()

        loader = tqdm(
            train_loader,
            disable=rank != 0,
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )

        for batch in loader:
            x_src = batch["x_src"].to(device, non_blocking=True)
            x_target = batch["x_target"].to(device, non_blocking=True)

            pa_src = {
                k: v.to(device, non_blocking=True)
                for k, v in batch["pa_src"].items()
            }

            pa_cf = {
                k: v.to(device, non_blocking=True)
                for k, v in batch["pa_cf"].items()
            }

            optimizer.zero_grad(set_to_none=True)

            loss, stats = model(
                x_src,
                x_target,
                pa_src,
                pa_cf,
            )

            loss.backward()

            gnorm = nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            optimizer.step()
            ema.update()

            step += 1

            if rank == 0:
                loader.set_postfix(
                    loss=f"{stats['loss'].item():.5f}",
                    fm=f"{stats['fm'].item():.5f}",
                    ep=f"{stats['endpoint'].item():.5f}",
                    cf=f"{stats['cf'].item():.5f}",
                    recon=f"{stats['recon'].item():.5f}",
                    pcf=f"{stats['cf_frac'].item():.2f}",
                    t=f"{stats['t'].item():.2f}",
                    gn=f"{float(gnorm):.2f}",
                )

            # =================================================
            # Validation
            # =================================================

            if valid_loader is not None and step % args.eval_freq == 0:
                if is_dist:
                    dist.barrier()

                eval_model = unwrap(model)

                (
                    valid_loss,
                    valid_fm,
                    valid_endpoint,
                    valid_cf,
                    valid_recon,
                    valid_cf_frac,
                    valid_t,
                ) = evaluate(
                    eval_model,
                    valid_loader,
                    device,
                    seed=args.seed + step,
                )

                if rank == 0:
                    print(
                        f"\nstep={step} "
                        f"valid_loss={valid_loss:.6f} "
                        f"fm={valid_fm:.6f} "
                        f"endpoint={valid_endpoint:.6f} "
                        f"cf={valid_cf:.6f} "
                        f"recon={valid_recon:.6f} "
                        f"cf_frac={valid_cf_frac:.3f}"
                    )

                    state = {
                        "model_state_dict": eval_model.flow.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "ema_state": ema.state_dict(),
                        "args": vars(train_args),
                        "finetune_args": vars(args),
                        "step": step,
                        "epoch": epoch,
                    }

                    torch.save(
                        state,
                        Path(args.save_dir) / "last_checkpoint.pt",
                    )

                    if valid_endpoint < best_endpoint:
                        best_endpoint = valid_endpoint

                        torch.save(
                            state,
                            Path(args.save_dir) / "best_checkpoint.pt",
                        )

                        print(
                            f"=> best checkpoint saved, "
                            f"endpoint loss={best_endpoint:.6f}"
                        )

                if is_dist:
                    dist.barrier()

                model.train()

        if rank == 0:
            now = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
            print(f"{now} finished epoch {epoch + 1}")

    # ========================================================
    # Final save
    # ========================================================

    if rank == 0:
        base = unwrap(model)

        state = {
            "model_state_dict": base.flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state": ema.state_dict(),
            "args": vars(train_args),
            "finetune_args": vars(args),
            "step": step,
            "epoch": args.epochs,
        }

        path = Path(args.save_dir) / "last_checkpoint.pt"
        torch.save(state, path)

        print("Saved:", path)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()