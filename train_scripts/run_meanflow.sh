#!/bin/bash

base_name="${1:-meanflow}"
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

    img_height=256
    img_width=192
    img_channels=1

    cond_embedder="per_attr"
    model_channels=64
    cond_embed_dim=160
    p_uncond=0.2

    epochs=10000
    bs=32
    lr=1e-4

    mf_ratio_r_neq_t=0.25
    mf_time_sampler="lognorm"
    mf_lognorm_mu=-0.4
    mf_lognorm_sigma=1.0
    mf_adaptive_weight_p=1.0
    mf_adaptive_weight_eps=1e-3

    sample_steps=1

    exp_name="${dataset}_${base_name}_${img_height}_${img_width}_condemb_${cond_embedder}_mchannel_${model_channels}_puncond_${p_uncond}_rneqt_${mf_ratio_r_neq_t}_${mf_time_sampler}"
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

    # TRAIN
        --resume=""
        --exp_name="$exp_name"
        --seed=6
        --epochs=$epochs
        --bs=$bs
        --lr=$lr
        --lr_warmup=2000
        --wd=0.0
        --betas 0.9 0.999
        --eps=1e-8
        --ema_rate=0.9999
        --eval_freq=5000
        --num_workers=8
        --prefetch_factor=4
        --dist

    # SAMPLING / CFG
        --sample_steps=$sample_steps
        --p_uncond=$p_uncond
        --cond_embedder=$cond_embedder

    # MEANFLOW
        --mf_ratio_r_neq_t=$mf_ratio_r_neq_t
        --mf_time_sampler=$mf_time_sampler
        --mf_lognorm_mu=$mf_lognorm_mu
        --mf_lognorm_sigma=$mf_lognorm_sigma
        --mf_adaptive_weight_p=$mf_adaptive_weight_p
        --mf_adaptive_weight_eps=$mf_adaptive_weight_eps

    # MODEL
        unet
        --model_channels=$model_channels
        --channel_mult 1 2 3 4 5
        --cond_embed_dim=$cond_embed_dim
        --num_blocks=3
        --attn_resolutions 16x12
        --label_balance=0.5
        --concat_balance=0.5
        --resample_filter 1 1
        --channels_per_head=64
        --dropout=0.0
        --res_balance=0.3
        --attn_balance=0.3
        --clip_act=256
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
    -m src.training.train_meanflow ${ARGS[@]} | tee "${save_dir}/log.out"
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
    -m src.training.train_meanflow ${ARGS[@]} | tee "${save_dir}/log.out"
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
        -m src.training.train_meanflow "${ARGS[@]}" | tee "${save_dir}/log.out"
fi
