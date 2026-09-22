#!/bin/bash

# Usage:
#
#   bash scripts/finetune_paired_view.sh gpus48
#   bash scripts/finetune_paired_view.sh gpus24
#   bash scripts/finetune_paired_view.sh local

partition="${1:-local}"

project_root="/vol/biomedic3/tx1215/mamo-flow"
ckpt_root="${project_root}/checkpoints"


# ============================================================
# Pretrained experiment
# ============================================================

base_exp_name="embed_latent_flow_small_bs_flux2_dit_b2_64_48_condemb_per_attr_cdim_160_puncond_0.2"

ckpt_file="last_checkpoint.pt"
ckpt="${ckpt_root}/${base_exp_name}/${ckpt_file}"

if [ ! -f "$ckpt" ]; then
    echo "Checkpoint not found: $ckpt"
    exit 1
fi


# ============================================================
# Paired data
# ============================================================

pair_csv="${project_root}/cache/flux2_vae_1024x768/pair_csv_latent.csv"

if [ ! -f "$pair_csv" ]; then
    echo "Pair CSV not found: $pair_csv"
    exit 1
fi


# ============================================================
# Fine-tuning
# ============================================================

epochs=20

# --bs is PER GPU / PER DDP process.
bs=16

lr=1e-5
wd=1e-4

lambda_pair=1.0
p_cf=0.8
endpoint_loss="mse"
t_max=0.6

cf_keys="view"

num_workers=8
prefetch_factor=4

seed=8
ema_rate=0.9999
grad_clip=1.0
eval_freq=1000


# ============================================================
# Distributed
#
# Total GPUs:
#
#     NNODES × NPROC_PER_NODE
#
# Global batch:
#
#     bs × NNODES × NPROC_PER_NODE
# ============================================================

NNODES=1
NPROC_PER_NODE=2

world_size=$((NNODES * NPROC_PER_NODE))
global_bs=$((bs * world_size))


# ============================================================
# Experiment name
# ============================================================

exp_name="${base_exp_name}_paired_ft_pcf_${p_cf}_lambda_${lambda_pair}_tmax_${t_max}"
save_dir="${ckpt_root}/${exp_name}"

mkdir -p "$save_dir"


# ============================================================
# Information
# ============================================================

echo "========================================"
echo "Paired-view supervised fine-tuning"
echo
echo "  partition         = $partition"
echo
echo "  base experiment   = $base_exp_name"
echo "  checkpoint        = $ckpt"
echo "  pair CSV          = $pair_csv"
echo
echo "  nodes             = $NNODES"
echo "  GPUs/node         = $NPROC_PER_NODE"
echo "  total GPUs        = $world_size"
echo "  batch/GPU         = $bs"
echo "  global batch      = $global_bs"
echo
echo "  epochs            = $epochs"
echo "  lr                = $lr"
echo "  weight decay      = $wd"
echo
echo "  lambda_pair       = $lambda_pair"
echo "  p_cf              = $p_cf"
echo "  endpoint loss     = $endpoint_loss"
echo "  t_max             = $t_max"
echo "  cf_keys           = $cf_keys"
echo
echo "  exp_name          = $exp_name"
echo "  save_dir          = $save_dir"
echo "========================================"


# ============================================================
# Arguments
# ============================================================

ARGS=(
    --ckpt="$ckpt"
    --save_dir="$save_dir"
    --pair_csv="$pair_csv"

    --epochs=$epochs
    --bs=$bs
    --lr=$lr
    --wd=$wd

    --lambda_pair=$lambda_pair
    --p_cf=$p_cf
    --endpoint_loss=$endpoint_loss
    --t_max=$t_max
    --cf_keys $cf_keys

    --num_workers=$num_workers
    --prefetch_factor=$prefetch_factor

    --seed=$seed
    --ema_rate=$ema_rate
    --grad_clip=$grad_clip
    --eval_freq=$eval_freq

    --dist
)


# ============================================================
# gpus48
# ============================================================

if [ "$partition" = "gpus48" ]; then

    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=paired_view_ft
#SBATCH --partition=gpus48
#SBATCH --nodes=${NNODES}
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log


source ~/.bashrc

cd ${project_root}

uv sync --frozen


echo "========================================"
echo "Host:        \$(hostname)"
echo "Job:         \$SLURM_JOB_ID"
echo "Nodes:       ${NNODES}"
echo "GPUs/node:   ${NPROC_PER_NODE}"
echo "Total GPUs:  ${world_size}"
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

export NCCL_P2P_DISABLE=1

echo "MASTER_ADDR=\$MASTER_ADDR"
echo "MASTER_PORT=\$MASTER_PORT"


# ============================================================
# Fine-tuning
#
# One torchrun launcher is started on each node.
# Each torchrun starts NPROC_PER_NODE GPU processes.
# ============================================================

srun \\
    --nodes=${NNODES} \\
    --ntasks=${NNODES} \\
    --ntasks-per-node=1 \\
    bash -c 'uv run torchrun \\
        --nnodes=${NNODES} \\
        --nproc_per_node=${NPROC_PER_NODE} \\
        --node_rank=\$SLURM_NODEID \\
        --rdzv_id="\$SLURM_JOB_ID" \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \\
        -m src.training.finetune_paired_view ${ARGS[@]}' \\
    2>&1 | tee "${save_dir}/log.out"

EOF


# ============================================================
# gpus24
# ============================================================

elif [ "$partition" = "gpus24" ]; then

    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=paired_view_ft
#SBATCH --partition=gpus24
#SBATCH --nodes=${NNODES}
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=${save_dir}/slurm.%j.log


source ~/.bashrc

cd ${project_root}

uv sync --frozen


echo "========================================"
echo "Host:        \$(hostname)"
echo "Job:         \$SLURM_JOB_ID"
echo "Nodes:       ${NNODES}"
echo "GPUs/node:   ${NPROC_PER_NODE}"
echo "Total GPUs:  ${world_size}"
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
# Fine-tuning
# ============================================================

srun \\
    --nodes=${NNODES} \\
    --ntasks=${NNODES} \\
    --ntasks-per-node=1 \\
    bash -c 'uv run torchrun \\
        --nnodes=${NNODES} \\
        --nproc_per_node=${NPROC_PER_NODE} \\
        --node_rank=\$SLURM_NODEID \\
        --rdzv_id="\$SLURM_JOB_ID" \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \\
        -m src.training.finetune_paired_view ${ARGS[@]}' \\
    2>&1 | tee "${save_dir}/log.out"

EOF


# ============================================================
# Local
# ============================================================

else

    if [ "$NNODES" -ne 1 ]; then
        echo "Local mode currently requires NNODES=1."
        exit 1
    fi

    RDZV_ID="${RDZV_ID:-$(date +%s)-$$}"

    export OMP_NUM_THREADS=1
    export TQDM_MININTERVAL=300
    export WANDB_MODE=disabled

    export MASTER_ADDR=localhost
    export MASTER_PORT=$(shuf -i 10001-29500 -n 1)


    echo "========================================"
    echo "Local paired-view fine-tuning"
    echo
    echo "Nodes:            ${NNODES}"
    echo "GPUs/node:        ${NPROC_PER_NODE}"
    echo "Total GPUs:       ${world_size}"
    echo "Batch/GPU:        ${bs}"
    echo "Global batch:     ${global_bs}"
    echo "========================================"


    cd "${project_root}" || exit 1

    uv run torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --node_rank=0 \
        --rdzv_id="${RDZV_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m src.training.finetune_paired_view \
        "${ARGS[@]}" \
        2>&1 | tee "${save_dir}/log.out"

fi