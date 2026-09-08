#!/bin/bash

base_name="${1:-latent_flow}"
partition="${2:-local}"

project_root="/vol/biomedic3/tx1215/mamo-flow"
ckpt_root="${project_root}/checkpoints"


# ============================================================
# Resume config
#
# Leave empty for fresh training:
#
#   resume_exp_name=""
#
# Put experiment folder name here for resume training.
# ============================================================

resume_exp_name=""

# Example:
#
# resume_exp_name="embed_latent_flow_1024_flux2_dit_b4_128_96_condemb_per_attr_cdim_160_puncond_0.2"


mkdir -p "$ckpt_root"


# ============================================================
# Number of GPUs
#
# Current SLURM setup:
#     2 GPUs
#
# For global batch = 72:
#
#     36 per GPU × 2 GPUs = 72
# ============================================================

NPROC_PER_NODE=2


# ============================================================
# Resume mode
# ============================================================

if [ -n "$resume_exp_name" ]; then

    exp_name="$resume_exp_name"
    save_dir="${ckpt_root}/${exp_name}"

    resume_ckpt="${save_dir}/last_checkpoint.pt"

    # Alternatively:
    # resume_ckpt="${save_dir}/best_checkpoint.pt"

    if [ ! -f "$resume_ckpt" ]; then
        echo "Resume checkpoint not found: $resume_ckpt"
        exit 1
    fi

    echo "========================================"
    echo "Resume mode enabled"
    echo
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

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This must be a NEW cache generated from the full
    # 1024 x 768 mammograms.
    #
    # Do NOT reuse:
    #
    #   flux2_vae_512x384
    #
    # because its latent shape is 32 x 64 x 48.
    # --------------------------------------------------------

    cache_dir="${project_root}/cache/flux2_vae_1024x768"


    # ========================================================
    # Latent dimensions
    #
    # Mammogram:
    #
    #     1 x 1024 x 768
    #
    # FLUX.2 VAE:
    #
    #     spatial downsampling = 8
    #
    # FLUX latent:
    #
    #     32 x 128 x 96
    # ========================================================

    img_channels=32
    img_height=128
    img_width=96


    # ========================================================
    # Conditioning
    # ========================================================

    cond_embedder="per_attr"

    # 160 / 5 parents = 32 dimensions per parent.
    cond_embed_dim=160

    p_uncond=0.2


    # ========================================================
    # DiT
    # ========================================================

    patch_size=2
    # hidden_size=1024
    # depth=16
    # num_heads=16
    hidden_size=1536
    depth=32
    num_heads=24
    mlp_ratio=2.6666666666666665


    # ========================================================
    # Training
    # ========================================================

    epochs=1000000

    # --------------------------------------------------------
    # Batch size
    #
    # --bs is PER GPU / PER DDP PROCESS.
    #
    # 2 GPUs:
    #
    #     36 × 2 = 72 global batch
    #
    # This gives a RadiT-B-like global batch size.
    # --------------------------------------------------------

    bs=12
    lr=1e-4
    # RadiT-style transformer training.
    lr_warmup=5000
    wd=1e-4
    eval_freq=5000
    ema_rate=0.9999


    # ========================================================
    # Flow
    # ========================================================

    alpha=1.0
    sigma=0.0

    # Number of ODE integration points used for
    # generation / counterfactual plotting.
    T=150


    # ========================================================
    # Experiment name
    # ========================================================

    exp_name="${dataset}_${base_name}_flux2_dit_b4_${img_height}_${img_width}_condemb_${cond_embedder}_cdim_${cond_embed_dim}_puncond_${p_uncond}"

    save_dir="${ckpt_root}/${exp_name}"


    # ========================================================
    # Information
    # ========================================================

    global_bs=$((bs * NPROC_PER_NODE))

    echo "========================================"
    echo "Fresh 1024x768 latent-flow training"
    echo
    echo "  dataset          = $dataset"
    echo "  data_dir         = $data_dir"
    echo "  cache_dir        = $cache_dir"
    echo
    echo "  image resolution = 1024x768"
    echo "  latent shape     = ${img_channels}x${img_height}x${img_width}"
    echo
    echo "  model            = DiT"
    echo "  patch_size       = $patch_size"
    echo "  token grid       = 32x24"
    echo "  tokens/image     = 768"
    echo
    echo "  hidden_size      = $hidden_size"
    echo "  depth            = $depth"
    echo "  num_heads        = $num_heads"
    echo "  mlp_ratio        = $mlp_ratio"
    echo
    echo "  GPUs             = $NPROC_PER_NODE"
    echo "  batch/GPU        = $bs"
    echo "  global batch     = $global_bs"
    echo
    echo "  lr               = $lr"
    echo "  warmup           = $lr_warmup"
    echo "  weight decay     = $wd"
    echo
    echo "  alpha            = $alpha"
    echo "  sigma            = $sigma"
    echo "  ODE steps        = $T"
    echo
    echo "  exp_name         = $exp_name"
    echo "  save_dir         = $save_dir"
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

        --betas 0.9 0.999
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
# gpus48
# ============================================================

if [ "$partition" = "gpus48" ]; then

    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=latent_dit_1024
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log


source ~/.bashrc

cd ${project_root}

uv sync --frozen


echo "========================================"
echo "Host: \$(hostname)"
echo "Job:  \$SLURM_JOB_ID"
echo "GPUs: ${NPROC_PER_NODE}"
echo "========================================"

nvidia-smi


# ============================================================
# Runtime environment
# ============================================================

# Avoid CPU oversubscription:
#
# 2 DDP processes × 1 OMP thread each
#
export OMP_NUM_THREADS=1

export TQDM_MININTERVAL=300


# ------------------------------------------------------------
# Disable Weights & Biases
# ------------------------------------------------------------

export WANDB_MODE=disabled


# ------------------------------------------------------------
# DDP rendezvous
# ------------------------------------------------------------

export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)


# ------------------------------------------------------------
# Keep this for BioMedIA gpus48 if P2P causes issues.
# ------------------------------------------------------------

export NCCL_P2P_DISABLE=1


echo "MASTER_ADDR=\$MASTER_ADDR"
echo "MASTER_PORT=\$MASTER_PORT"


# ============================================================
# Training
# ============================================================

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

#SBATCH --job-name=latent_dit_1024
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log


source ~/.bashrc

cd ${project_root}

uv sync --frozen


echo "========================================"
echo "Host: \$(hostname)"
echo "Job:  \$SLURM_JOB_ID"
echo "GPUs: ${NPROC_PER_NODE}"
echo "========================================"

nvidia-smi


# ============================================================
# Runtime environment
# ============================================================

export OMP_NUM_THREADS=1
export TQDM_MININTERVAL=300

export WANDB_MODE=disabled


# ------------------------------------------------------------
# DDP rendezvous
# ------------------------------------------------------------

export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)


echo "MASTER_ADDR=\$MASTER_ADDR"
echo "MASTER_PORT=\$MASTER_PORT"


# ============================================================
# Training
# ============================================================

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

    # Keep 2 GPUs here as well so that:
    #
    #     bs=36
    #
    # always means:
    #
    #     global batch = 72
    #
    # If you change this to 8 GPUs, reduce/increase bs
    # depending on your desired GLOBAL batch size.

    NPROC_PER_NODE=2

    RDZV_ID="${RDZV_ID:-$(date +%s)-$$}"


    export OMP_NUM_THREADS=1
    export TQDM_MININTERVAL=300

    export WANDB_MODE=disabled

    export MASTER_ADDR=localhost
    export MASTER_PORT=$(shuf -i 10001-29500 -n 1)


    echo "========================================"
    echo "Local 1024x768 latent-flow training"
    echo
    echo "GPUs:             ${NPROC_PER_NODE}"
    echo "Batch/GPU:        ${bs:-resume}"
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