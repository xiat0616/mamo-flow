#!/bin/bash

base_name="${1:-flow_dit}"
partition="${2:-local}"

project_root="/vol/biomedic3/tx1215/mamo-flow"
ckpt_root="${project_root}/checkpoints"

# ============================================================
# Resume config
#
# Leave empty for fresh training:
# resume_exp_name=""
#
# Put experiment folder name here for resume training:
# ============================================================

resume_exp_name=""

mkdir -p "$ckpt_root"

# ============================================================
# Resume mode
# Everything follows resume_exp_name.
# You only need to change resume_exp_name above.
# ============================================================
if [ -n "$resume_exp_name" ]; then
    exp_name="$resume_exp_name"
    save_dir="${ckpt_root}/${exp_name}"
    resume_ckpt="${save_dir}/best_checkpoint.pt"

    if [ ! -f "$resume_ckpt" ]; then
        echo "Resume checkpoint not found: $resume_ckpt"
        exit 1
    fi

    echo "Resume mode enabled"
    echo "  resume_exp_name = $resume_exp_name"
    echo "  resume_ckpt     = $resume_ckpt"
    echo "  exp_name        = $exp_name"
    echo "  save_dir        = $save_dir"

    ARGS=(
        --resume="$resume_ckpt"
        --exp_name="$exp_name"
        --save_dir="$save_dir"
    )

# ============================================================
# Fresh training mode
# Used only when resume_exp_name=""
# ============================================================
else
    dataset="embed"

    # ---- pixel-space defaults ----
    # For latent space: set img_height/img_width to latent spatial dims,
    # img_channels to latent depth (e.g. 16 for raddino, 32 for flux2),
    # and add --cache_dir pointing to your memmap cache.
    img_height=128
    img_width=96
    img_channels=1

    cond_embedder="per_attr"
    p_uncond=0.2

    epochs=10000
    bs=64
    lr=1e-4

    # ---- DiT architecture ----
    # patch_size=8 on 128x96 → 16x12=192 patches (good for pixel space)
    # For latent 32x24: patch_size=2 → 16x12=192 patches
    # For latent 32x24: patch_size=4 → 8x6=48 patches
    patch_size=8
    hidden_size=768   # DiT-B: 768 / DiT-L: 1024 / DiT-XL: 1152
    depth=12          # DiT-B: 12  / DiT-L: 24   / DiT-XL: 28
    num_heads=12      # hidden_size must be divisible by num_heads
    mlp_ratio=2.6667  # 8/3
    cond_embed_dim=256

    exp_name="${dataset}_${base_name}_${img_height}_${img_width}_h${hidden_size}_d${depth}_p${patch_size}_puncond_${p_uncond}"
    save_dir="${ckpt_root}/${exp_name}"

    echo "Fresh training mode enabled"
    echo "  exp_name = $exp_name"
    echo "  save_dir = $save_dir"

    ARGS=(
    # DATA
        --dataset="$dataset"
        --data_dir="/vol/biodata/data/Mammo/EMBED/pngs/1024x768"
        --split_dir="${project_root}/assets/embed_splits_v1"
        --save_dir="$save_dir"
        --parents age view density scanner cview
        --img_height=$img_height
        --img_width=$img_width
        --img_channels=$img_channels
        # --cache_dir="/path/to/latent/cache"   # uncomment for latent mode

    # TRAIN
        --resume=""
        --exp_name="$exp_name"
        --seed=6
        --epochs=$epochs
        --bs=$bs
        --lr=$lr
        --lr_warmup=5000
        --wd=0.0
        --betas 0.9 0.999
        --eps=1e-8
        --ema_rate=0.9999
        --eval_freq=1000
        --num_workers=8
        --prefetch_factor=4
        --dist

    # FLOW
        --alpha=1.0
        --sigma=0.0
        --T=150
        --p_uncond=$p_uncond
        --cond_embedder=$cond_embedder

    # MODEL
        dit
        --hidden_size=$hidden_size
        --depth=$depth
        --num_heads=$num_heads
        --patch_size=$patch_size
        --mlp_ratio=$mlp_ratio
        --cond_embed_dim=$cond_embed_dim
        # --grad_checkpointing   # uncomment for large models (DiT-L/XL) to save memory
    )
fi

mkdir -p "$save_dir"

NPROC_PER_NODE=2

if [ "$partition" = "gpus48" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log

source ~/.bashrc
cd ${project_root}
uv sync --frozen

nvidia-smi
export OMP_NUM_THREADS=${NPROC_PER_NODE}
export TQDM_MININTERVAL=300
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)
export NCCL_P2P_DISABLE=1

srun uv run torchrun \\
    --nnodes=1 \\
    --nproc_per_node=${NPROC_PER_NODE} \\
    --rdzv_id="\$SLURM_JOB_ID" \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \\
    -m src.training.train_flow ${ARGS[@]} | tee "${save_dir}/log.out"
EOF

elif [ "$partition" = "gpus24" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log

source ~/.bashrc
cd ${project_root}
uv sync --frozen

nvidia-smi
export OMP_NUM_THREADS=${NPROC_PER_NODE}
export TQDM_MININTERVAL=300
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)

srun uv run torchrun \\
    --nnodes=1 \\
    --nproc_per_node=${NPROC_PER_NODE} \\
    --rdzv_id="\$SLURM_JOB_ID" \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \\
    -m src.training.train_flow ${ARGS[@]} | tee "${save_dir}/log.out"
EOF

else
    NPROC_PER_NODE=8
    RDZV_ID="${RDZV_ID:-$(date +%s)-$$}"

    export OMP_NUM_THREADS=1
    export TQDM_MININTERVAL=300
    export MASTER_ADDR=localhost
    export MASTER_PORT=$(shuf -i 10001-29500 -n 1)

    uv run torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv_id="${RDZV_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m src.training.train_flow "${ARGS[@]}" | tee "${save_dir}/log.out"
fi
