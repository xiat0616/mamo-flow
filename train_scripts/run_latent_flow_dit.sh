#!/bin/bash

base_name="${1:-latent_flow}"
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

# Example:
# resume_exp_name="embed_latent_flow_flux2_dit_b2_64_48_condemb_per_attr_puncond_0.2"


mkdir -p "$ckpt_root"


# ============================================================
# Resume mode
#
# Everything follows resume_exp_name.
# Only change resume_exp_name above.
# ============================================================

if [ -n "$resume_exp_name" ]; then

    exp_name="$resume_exp_name"
    save_dir="${ckpt_root}/${exp_name}"

    # resume_ckpt="${save_dir}/best_checkpoint.pt"
    resume_ckpt="${save_dir}/last_checkpoint.pt"

    if [ ! -f "$resume_ckpt" ]; then
        echo "Resume checkpoint not found: $resume_ckpt"
        exit 1
    fi

    echo "========================================"
    echo "Resume mode enabled"
    echo "  resume_exp_name = $resume_exp_name"
    echo "  resume_ckpt     = $resume_ckpt"
    echo "  exp_name        = $exp_name"
    echo "  save_dir        = $save_dir"
    echo "========================================"

    ARGS=(
        --resume="$resume_ckpt"
        --exp_name="$exp_name"
        --save_dir="$save_dir"
        --lr=1e-4
    )


# ============================================================
# Fresh training
# ============================================================

else

    # ========================================================
    # Dataset
    # ========================================================

    dataset="embed"

    data_dir="/vol/biodata/data/Mammo/EMBED/pngs/1024x768"
    split_dir="${project_root}/assets/embed_splits_v1"

    # FLUX.2 latent cache generated from 512x384 images
    cache_dir="${project_root}/cache/flux2_vae_512x384"


    # ========================================================
    # Latent dimensions
    #
    # Original mammogram:
    #
    #   1 x 512 x 384
    #
    # FLUX.2 latent:
    #
    #   32 x 64 x 48
    # ========================================================

    img_channels=32
    img_height=64
    img_width=48


    # ========================================================
    # Conditioning
    # ========================================================

    cond_embedder="per_attr"
    cond_embed_dim=256

    p_uncond=0.2


    # ========================================================
    # DiT configuration
    #
    # DiT-B/2 style
    #
    # latent:
    #     64 x 48
    #
    # patch size:
    #     2
    #
    # token grid:
    #     32 x 24
    #
    # number tokens:
    #     768
    # ========================================================

    patch_size=2

    hidden_size=768
    depth=12
    num_heads=12
    mlp_ratio=2.6666666666666665


    # ========================================================
    # Training
    # ========================================================

    epochs=10000

    # Per-GPU batch size.
    #
    # With 2 GPUs:
    #
    #     bs=16
    #     global batch = 32
    #
    bs=16

    lr=1e-4

    lr_warmup=5000

    wd=1e-4

    eval_freq=5000

    ema_rate=0.9999


    # ========================================================
    # Flow
    # ========================================================

    alpha=1.0
    sigma=0.0

    T=150


    # ========================================================
    # Experiment name
    # ========================================================

    exp_name="${dataset}_${base_name}_flux2_dit_b2_${img_height}_${img_width}_condemb_${cond_embedder}_cdim_${cond_embed_dim}_puncond_${p_uncond}"

    save_dir="${ckpt_root}/${exp_name}"


    echo "========================================"
    echo "Fresh latent-flow training"
    echo
    echo "  dataset        = $dataset"
    echo "  cache_dir      = $cache_dir"
    echo
    echo "  latent shape   = ${img_channels}x${img_height}x${img_width}"
    echo
    echo "  model          = DiT"
    echo "  patch_size     = $patch_size"
    echo "  hidden_size    = $hidden_size"
    echo "  depth          = $depth"
    echo "  num_heads      = $num_heads"
    echo
    echo "  batch/GPU      = $bs"
    echo "  lr             = $lr"
    echo
    echo "  exp_name       = $exp_name"
    echo "  save_dir       = $save_dir"
    echo "========================================"


    # ========================================================
    # Arguments
    #
    # IMPORTANT:
    #
    # General arguments MUST come before:
    #
    #     dit
    #
    # DiT-specific arguments MUST come after:
    #
    #     dit
    # ========================================================

    ARGS=(

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

        --dataset="$dataset"

        --data_dir="$data_dir"

        --split_dir="$split_dir"

        --cache_dir="$cache_dir"

        --save_dir="$save_dir"

        --vae_ckpt=flux2

        --parents age view density scanner cview

        --img_height=$img_height
        --img_width=$img_width
        --img_channels=$img_channels


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

        --resume=""

        --exp_name="$exp_name"

        --seed=8

        --epochs=$epochs

        --bs=$bs

        --lr=$lr

        --lr_warmup=$lr_warmup

        --wd=$wd

        --betas 0.9 0.99

        --eps=1e-8

        --ema_rate=$ema_rate

        --eval_freq=$eval_freq

        --num_workers=8

        --prefetch_factor=4

        --dist


    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

        --alpha=$alpha

        --sigma=$sigma

        --T=$T

        --p_uncond=$p_uncond

        --cond_embedder=$cond_embedder


    # --------------------------------------------------------
    # MODEL
    #
    # argparse subcommand
    # --------------------------------------------------------

        dit

        --hidden_size=$hidden_size

        --depth=$depth

        --num_heads=$num_heads

        --patch_size=$patch_size

        --mlp_ratio=$mlp_ratio

        --cond_embed_dim=$cond_embed_dim

        --grad_checkpointing
    )

fi


# ============================================================
# Output directory
# ============================================================

mkdir -p "$save_dir"


# ============================================================
# Number GPUs
# ============================================================

NPROC_PER_NODE=2


# ============================================================
# gpus48
# ============================================================

if [ "$partition" = "gpus48" ]; then

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=latent_dit
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log

set -euo pipefail

source ~/.bashrc

cd ${project_root}

uv sync --frozen

echo "========================================"
echo "Host: \$(hostname)"
echo "Job:  \$SLURM_JOB_ID"
echo "GPUs: ${NPROC_PER_NODE}"
echo "========================================"

nvidia-smi

export OMP_NUM_THREADS=${NPROC_PER_NODE}
export TQDM_MININTERVAL=300

export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)

# Keep this for BioMedIA gpus48 if P2P causes issues.
export NCCL_P2P_DISABLE=1

srun uv run torchrun \\
    --nnodes=1 \\
    --nproc_per_node=${NPROC_PER_NODE} \\
    --rdzv_id="\$SLURM_JOB_ID" \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \\
    -m src.training.train_flow ${ARGS[@]} \\
    2>&1 | tee "${save_dir}/log.out"

EOF


# ============================================================
# gpus24
# ============================================================

elif [ "$partition" = "gpus24" ]; then

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=latent_dit
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log

set -euo pipefail

source ~/.bashrc

cd ${project_root}

uv sync --frozen

echo "========================================"
echo "Host: \$(hostname)"
echo "Job:  \$SLURM_JOB_ID"
echo "GPUs: ${NPROC_PER_NODE}"
echo "========================================"

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
    -m src.training.train_flow ${ARGS[@]} \\
    2>&1 | tee "${save_dir}/log.out"

EOF


# ============================================================
# Local
# ============================================================

else

    # Change if you don't want to use all 8 GPUs locally.
    NPROC_PER_NODE=8

    RDZV_ID="${RDZV_ID:-$(date +%s)-$$}"

    export OMP_NUM_THREADS=1
    export TQDM_MININTERVAL=300

    export MASTER_ADDR=localhost
    export MASTER_PORT=$(shuf -i 10001-29500 -n 1)

    echo "========================================"
    echo "Local latent-flow training"
    echo "GPUs: ${NPROC_PER_NODE}"
    echo "========================================"

    uv run torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv_id="${RDZV_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m src.training.train_flow \
        "${ARGS[@]}" \
        2>&1 | tee "${save_dir}/log.out"

fi