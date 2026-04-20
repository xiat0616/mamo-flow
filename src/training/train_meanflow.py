import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

import wandb
from data_handle.datasets import (
    CLASS_SCHEMA,
    get_embed,
    get_dataloaders,
    DataLoaderConfig,
    DatasetConfig,
)
from utils import (
    ModelEMA,
    get_mc_stats,
    save_plots_mf,
    seed_all,
    setup_distributed,
    unwrap,
)


def parse_hw(s: str) -> tuple[int, int]:
    try:
        h, w = s.lower().split("x")
        return int(h), int(w)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{s}'. Expected format like 128x96."
        ) from e


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


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: argparse.Namespace,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        ema: ModelEMA | None = None,
        vae: nn.Module | None = None,
    ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.vae = vae
        self.device = next(model.parameters()).device
        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.step, self.epoch = 0, 0
        self.best_loss = 1e6
        self.eval_mc = 8
        self.tqdm_kwargs = dict(
            disable=(self.rank != 0),
            mininterval=float(os.environ.get("TQDM_MININTERVAL", 1)),
        )

    def train_epoch(self, dataloaders: dict[str, torch.utils.data.DataLoader]) -> float:
        missing = {"train", "valid"} - dataloaders.keys()
        assert not missing, f"Missing dataloader(s): {sorted(missing)}"

        self.model.train()
        dataloader = dataloaders["train"]
        loader = tqdm(enumerate(dataloader), total=len(dataloader), **self.tqdm_kwargs)

        total_loss = torch.tensor(0.0, device=self.device)
        n = torch.tensor(0, device=self.device)

        for _, batch in loader:
            x, pa = batch["x"], batch["pa"]
            bs, channels = x.shape[:2]

            x = x.float().to(self.device, non_blocking=True)
            pa = {k: v.to(self.device, non_blocking=True) for k, v in pa.items()}

            # Dequantize image inputs during training, then map to [-1, 1]
            if channels <= 3:
                x = (x + (torch.rand_like(x) - 0.5) / 255.0).clamp(0, 1) * 2 - 1

            self.optimizer.zero_grad(set_to_none=True)
            result = self.model(x, pa, g=None)
            loss, raw_mse = result if isinstance(result, tuple) else (result, None)
            loss.backward()

            stats = {
                "gnorm": nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
            }
            if raw_mse is not None:
                stats["raw_mse"] = raw_mse.item()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
                stats["lr"] = self.scheduler.get_last_lr()[0]

            if self.ema is not None:
                self.ema.update()

            n += bs
            total_loss += loss.detach() * bs
            self.step += 1

            if self.rank == 0:
                elapsed = max(loader.format_dict["elapsed"], 1e-6)
                world = dist.get_world_size() if self.is_dist else 1
                tok_ps = bs * 1024 * world * (loader.n / elapsed)
                wandb.log(stats | {"tokens/s": tok_ps}, self.step)
                loader.set_postfix({"tok/s": f"{tok_ps:,.0f}"}, refresh=False)
                loader.set_description(
                    f"train loss: {total_loss / n:.7f}, "
                    + ", ".join(f"{k}: {v:7f}" for k, v in stats.items()),
                    refresh=False,
                )

            if (self.step % self.args.eval_freq) == 0:
                t0 = time.time()
                self.model.eval()
                if self.ema is not None:
                    self.ema.apply()

                g = torch.Generator(device=self.device)
                mc_losses = []
                for k in range(self.eval_mc):
                    g.manual_seed(self.args.seed + k)
                    mc_losses.append(self.eval_epoch(dataloaders["valid"], g))

                if self.rank == 0:
                    mc_stats = get_mc_stats(mc_losses, prefix="valid_loss")
                    mc_stats["valid_loss"] = mc_stats.pop("valid_loss_mean")
                    mc_stats = {k: v.item() for k, v in mc_stats.items()}
                    print("\n" + ", ".join(f"{k}: {v:7f}" for k, v in mc_stats.items()))
                    wandb.log(mc_stats | {"valid_mc": self.eval_mc}, self.step)
                    self.save_checkpoint(mc_stats["valid_loss"])

                    save_plots_mf(
                        batch_size=bs,
                        dataset=dataloaders["valid"].dataset,
                        model=self.model,
                        vae=self.vae,
                        steps=self.args.sample_steps,
                        save_path=os.path.join(self.args.save_dir, f"{self.step}"),
                    )

                    eval_elapsed = time.time() - t0
                    print(f"Eval time elapsed: {eval_elapsed:.2f}s")
                    loader.start_t += eval_elapsed

                if self.is_dist:
                    dist.barrier()

                del mc_losses
                if self.ema is not None:
                    self.ema.restore()
                self.model.train()

        self.epoch += 1

        if self.is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

        return (total_loss / n).item()

    # CRITICAL FIX: use @torch.no_grad() instead of @torch.inference_mode().
    # torch.func.jvp uses forward-mode AD which requires dual tensors —
    # inference_mode() creates inference tensors that block this.
    # no_grad() only disables the reverse-mode tape, which is fine.
    @torch.no_grad()
    def eval_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        g: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        loader = tqdm(enumerate(dataloader), total=len(dataloader), **self.tqdm_kwargs)

        total_loss = torch.tensor(0.0, device=self.device)
        n = torch.tensor(0, device=self.device)

        for _, batch in loader:
            x, pa = batch["x"], batch["pa"]
            bs, channels = x.shape[:2]

            x = x.float().to(self.device, non_blocking=True)
            if channels <= 3:
                x = x * 2 - 1
            pa = {k: v.to(self.device, non_blocking=True) for k, v in pa.items()}

            loss = self.model(x, pa, g=g)

            n += bs
            total_loss += loss.detach() * bs

            if self.rank == 0:
                elapsed = max(loader.format_dict["elapsed"], 1e-6)
                world = dist.get_world_size() if self.is_dist else 1
                tok_ps = bs * 1024 * world * (loader.n / elapsed)
                loader.set_description(f"eval loss: {total_loss / n:.7f}", refresh=False)
                loader.set_postfix({"tok/s": f"{tok_ps:,.0f}"}, refresh=False)

        if self.is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

        return (total_loss / n).detach()

    def save_checkpoint(self, loss: float) -> None:
        ckpt = {
            "model_state_dict": unwrap(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "args": vars(self.args),
            "step": self.step,
            "epoch": self.epoch,
        }
        last_path = os.path.join(self.args.save_dir, "last_checkpoint.pt")
        torch.save(ckpt, last_path)
        print(f"=> step: {self.step}, last model saved: {last_path}")

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.args.save_dir, "best_checkpoint.pt")
            torch.save(ckpt, best_path)
            print(f"=> step: {self.step}, best model saved: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--csv_filepath", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--parents", nargs="+", type=str, default=list(CLASS_SCHEMA))
    parser.add_argument("--domain", nargs="+", type=str, default=None)
    parser.add_argument("--scanner_model", nargs="+", type=str, default=None)
    parser.add_argument("--exclude_cviews", type=int, default=1)
    parser.add_argument("--hold_out_model_5", type=int, default=1)
    parser.add_argument("--prop_train", type=float, default=1.0)
    parser.add_argument("--valid_frac", type=float, default=0.125)
    parser.add_argument("--test_frac", type=float, default=0.125)
    parser.add_argument("--split_seed", type=int, default=33)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--img_width", type=int, default=384)
    parser.add_argument("--img_channels", type=int, default=1)

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="meanflow_smoke")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup", type=int, default=2000)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.99])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--determ", action="store_true", default=False)
    parser.add_argument("--dist", action="store_true", default=False)

    # ------------------------------------------------------------------
    # SAMPLING / CFG
    # ------------------------------------------------------------------
    parser.add_argument(
        "--sample_steps", "--T",
        dest="sample_steps",
        type=int,
        default=1,
        help="Number of MeanFlow sampling intervals used for eval plots.",
    )
    parser.add_argument("--p_uncond", type=float, default=0.2)
    parser.add_argument(
        "--cond_embedder",
        type=str,
        default="per_attr",
        choices=["per_attr", "global", "none"],
    )

    # ------------------------------------------------------------------
    # MEANFLOW-SPECIFIC
    # ------------------------------------------------------------------
    parser.add_argument("--mf_ratio_r_neq_t", type=float, default=0.25)
    parser.add_argument(
        "--mf_time_sampler",
        type=str,
        default="lognorm",
        choices=["uniform", "lognorm"],
    )
    parser.add_argument("--mf_lognorm_mu", type=float, default=-0.4)
    parser.add_argument("--mf_lognorm_sigma", type=float, default=1.0)
    parser.add_argument("--mf_adaptive_weight_p", type=float, default=1.0)
    parser.add_argument("--mf_adaptive_weight_eps", type=float, default=1e-3)

    # ------------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------------
    sub = parser.add_subparsers(dest="model", required=False)
    p_unet = sub.add_parser("unet")
    p_unet.add_argument("--model_channels", type=int, default=192)
    p_unet.add_argument("--channel_mult", nargs="+", type=int, default=[1, 2, 3, 4])
    p_unet.add_argument("--cond_embed_dim", type=int, default=256)
    p_unet.add_argument("--channel_mult_time", type=int, default=None)
    p_unet.add_argument("--channel_mult_emb", type=int, default=None)
    p_unet.add_argument("--num_blocks", type=int, default=3)
    p_unet.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=parse_hw,
        default=[(16, 12)],
        help="Attention resolutions as HxW pairs, e.g. 128x96 64x48",
    )
    p_unet.add_argument("--label_balance", type=float, default=0.5)
    p_unet.add_argument("--concat_balance", type=float, default=0.5)
    p_unet.add_argument("--resample_filter", nargs="+", type=int, default=[1, 1])
    p_unet.add_argument("--channels_per_head", type=int, default=64)
    p_unet.add_argument("--dropout", type=float, default=0.0)
    p_unet.add_argument("--res_balance", type=float, default=0.3)
    p_unet.add_argument("--attn_balance", type=float, default=0.3)
    p_unet.add_argument("--clip_act", type=int, default=256)

    args = parser.parse_args()

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        parser.set_defaults(**ckpt["args"])
        args = parser.parse_args()
        args.resume_step = int(ckpt.get("step", 0))

    if args.model is None:
        parser.error("Missing required subcommand: model (i.e. unet or transformer)")

    # ------------------------------------------------------------------
    # Device / AMP
    # ------------------------------------------------------------------
    _dist_out = setup_distributed() if args.dist else None
    if _dist_out is not None:
        _local_rank, rank, world_size = _dist_out
        device = torch.device(f"cuda:{_local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rank, world_size = 0, 1
    is_dist = args.dist and dist.is_available() and dist.is_initialized()

    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None

    seed_all(args.seed, args.determ)
    if args.resume:
        runtime_seed = int(args.seed + 7654321 * args.resume_step + rank)
        torch.manual_seed(runtime_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(runtime_seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    datasets = get_embed(
        DatasetConfig(
            data_dir=args.data_dir,
            csv_filepath=args.csv_filepath,
            cache_dir=args.cache_dir,
            parents=args.parents,
            domain=args.domain,
            scanner_model=args.scanner_model,
            exclude_cviews=args.exclude_cviews,
            hold_out_model_5=args.hold_out_model_5,
            prop_train=args.prop_train,
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
            split_seed=args.split_seed,
            img_height=args.img_height,
            img_width=args.img_width,
            img_channels=args.img_channels,
            vae_ckpt=args.vae_ckpt,
        )
    )
    dataloaders = get_dataloaders(
        DataLoaderConfig(
            bs=args.bs,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed,
            resume_step=getattr(args, "resume_step", 0),
        ),
        datasets,
    )

    # ------------------------------------------------------------------
    # Build MeanFlow model
    # ------------------------------------------------------------------
    if args.model == "unet":
        from models.embedder import (
            GlobalCondEmbedder,
            PerAttrCondEmbedder,
            CondEmbedderConfig,
        )
        from models.unet_mf import UNet
        from src.flows.meanflow import BlockConfig, MeanFlow, MeanFlowConfig, UNetConfig

        sample_batch = next(iter(dataloaders["train"]))
        parent_dims = infer_parent_dims_from_batch(sample_batch["pa"], args.parents)

        unet_cfg = UNetConfig(
            img_height=args.img_height,
            img_width=args.img_width,
            img_channels=args.img_channels,
            cond_embed_dim=args.cond_embed_dim,
            model_channels=args.model_channels,
            channel_mult=tuple(args.channel_mult),
            channel_mult_time=args.channel_mult_time,
            channel_mult_emb=args.channel_mult_emb,
            num_blocks=args.num_blocks,
            attn_resolutions=tuple(args.attn_resolutions),
            label_balance=args.label_balance,
            concat_balance=args.concat_balance,
        )

        block_cfg = BlockConfig(
            resample_filter=tuple(args.resample_filter),
            channels_per_head=args.channels_per_head,
            dropout=args.dropout,
            res_balance=args.res_balance,
            attn_balance=args.attn_balance,
            clip_act=args.clip_act,
        )

        forward_nn = UNet(**vars(unet_cfg), **vars(block_cfg))

        cond_embedder = None
        if args.cond_embedder != "none" and len(args.parents) > 0:
            embedder_cfg = CondEmbedderConfig(
                parents=args.parents,
                parent_dims=parent_dims,
                cond_embed_dim=args.cond_embed_dim,
            )
            if args.cond_embedder == "per_attr":
                cond_embedder = PerAttrCondEmbedder(embedder_cfg)
            elif args.cond_embedder == "global":
                cond_embedder = GlobalCondEmbedder(embedder_cfg)
            else:
                raise ValueError(f"Unknown cond_embedder: {args.cond_embedder}")

        mf_cfg = MeanFlowConfig(
            ratio_r_neq_t=args.mf_ratio_r_neq_t,
            time_sampler=args.mf_time_sampler,
            lognorm_mu=args.mf_lognorm_mu,
            lognorm_sigma=args.mf_lognorm_sigma,
            adaptive_weight_p=args.mf_adaptive_weight_p,
            adaptive_weight_eps=args.mf_adaptive_weight_eps,
        )

        model = MeanFlow(
            forward_nn=forward_nn,
            cond_embedder=cond_embedder,
            p_uncond=args.p_uncond,
            amp_dtype=amp_dtype,
            mf_config=mf_cfg,
        )

    elif args.model == "transformer":
        raise NotImplementedError("Transformer model not implemented yet")
    else:
        raise NotImplementedError

    if args.resume:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model = model.to(device)
    ema = ModelEMA(model.parameters(), rate=args.ema_rate)
    if args.resume and ckpt.get("ema_state") is not None:
        ema.load_state_dict(ckpt["ema_state"])

    if is_dist:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=ddp_device_ids, bucket_cap_mb=150)

    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        print(
            f"We do not use torch.compile as it reports errors, "
            f"but device {device} has CUDA capability {torch.cuda.get_device_capability(device)}"
        )
    else:
        print(
            f"Skipping torch.compile: device {device} has CUDA capability "
            f"{torch.cuda.get_device_capability(device) if device.type == 'cuda' else 'CPU'}, "
            f"requires >= 7.0"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=tuple(args.betas),
        eps=args.eps,
    )

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / args.lr_warmup,
            total_iters=args.lr_warmup,
        )

    vae = None
    if rank == 0:
        wandb.init(project="mammo_flow", name=args.exp_name, config=vars(args))
        for k, v in vars(args).items():
            print(f"--{k}={v}")

        num_params = sum(p.numel() for p in unwrap(model).parameters() if p.requires_grad)
        print(f"#params: {num_params:,}")

        if args.vae_ckpt == "flux2":
            from utils import get_pretrained_flux2vae
            vae = get_pretrained_flux2vae()

    print("\ntorch:", torch.__version__)
    print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
    print("amp dtype:", amp_dtype)
    print("matmul precision:", torch.get_float32_matmul_precision())

    if torch.cuda.is_available():
        print(
            "sdpa backends enabled:",
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )

    if is_dist:
        dist.barrier()

    trainer = Trainer(
        model,
        args,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        vae=vae,
    )

    if args.resume:
        trainer.step, trainer.epoch = ckpt.get("step", 0), ckpt.get("epoch", 0)

    for i in range(trainer.epoch, trainer.epoch + args.epochs):
        if is_dist:
            dataloaders["train"].sampler.set_epoch(i)

        now = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        print(f"\n{now}, Epoch {i + 1}:")
        train_loss = trainer.train_epoch(dataloaders)

        if rank == 0:
            wandb.log({"train_loss": train_loss}, trainer.step)

    if is_dist:
        dist.destroy_process_group()