#!/bin/bash

base_name="${1:-flow}"
partition="${2:-gpus24}"

# ----------------------------
# CIFAR-10 config
# ----------------------------
dataset="cifar10"
data_dir="/vol/biomedic3/tx1215/mamo-flow/assets/cifar10"

img_height=32
img_width=32
img_channels=3

parents=(y)
cond_embedder="per_attr"
model_channels=128
cond_embed_dim=128
p_uncond=0.2

epochs=10000
bs=128
lr=1e-4

valid_frac=0.05
split_seed=33

exp_name="${base_name}_${dataset}_${img_height}_${img_width}_condemb_${cond_embedder}_mchannel_${model_channels}_puncond_${p_uncond}"

mkdir -p ./checkpoints
mkdir -p "./checkpoints/$exp_name"

ARGS=(
# DATA
    --dataset="$dataset"
    --data_dir="$data_dir"
    --save_dir="./checkpoints/$exp_name"
    --parents "${parents[@]}"
    --valid_frac="$valid_frac"
    --split_seed="$split_seed"
    --img_height="$img_height"
    --img_width="$img_width"
    --img_channels="$img_channels"

# TRAIN
    --resume=
    --exp_name="$exp_name"
    --seed=6
    --epochs="$epochs"
    --bs="$bs"
    --lr="$lr"
    --lr_warmup=2000
    --wd=0.0
    --betas 0.9 0.999
    --eps=1e-8
    --ema_rate=0.9999
    --eval_freq=5000
    --num_workers=8
    --prefetch_factor=4
    --dist

# FLOW
    --alpha=1.0
    --sigma=0.0
    --T=150
    --p_uncond="$p_uncond"
    --cond_embedder="$cond_embedder"

# MODEL
    unet
    --model_channels="$model_channels"
    --channel_mult 1 2 2 2
    --cond_embed_dim="$cond_embed_dim"
    --num_blocks=3
    --attn_resolutions 16x16 8x8
    --label_balance=0.5
    --concat_balance=0.5
    --resample_filter 1 1
    --channels_per_head=64
    --dropout=0.0
    --res_balance=0.3
    --attn_balance=0.3
    --clip_act=256
)

# ----------------------------
# Slurm / local launch
# ----------------------------

NPROC_PER_NODE=2

if [ "$partition" = "gpus48" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=./checkpoints/$exp_name/slurm.%j.log

source ~/.bashrc
cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen

nvidia-smi
export OMP_NUM_THREADS=${NPROC_PER_NODE}
export TQDM_MININTERVAL=300
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)
export NCCL_P2P_DISABLE=1

srun uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_id="\$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \
    -m src.training.train_flow ${ARGS[@]} | tee "./checkpoints/$exp_name/log.out"
EOF

elif [ "$partition" = "gpus24" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=./checkpoints/$exp_name/slurm.%j.log

source ~/.bashrc
cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen

nvidia-smi
export OMP_NUM_THREADS=${NPROC_PER_NODE}
export TQDM_MININTERVAL=300
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)

srun uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_id="\$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \
    -m src.training.train_flow ${ARGS[@]} | tee "./checkpoints/$exp_name/log.out"
EOF

else
    NPROC_PER_NODE=2
    RDZV_ID="${RDZV_ID:-$(date +%s)-$$}"
    export OMP_NUM_THREADS=1
    export TQDM_MININTERVAL=300
    export MASTER_ADDR=localhost
    export MASTER_PORT=$(shuf -i 10001-29500 -n 1)

    cd /vol/biomedic3/tx1215/mamo-flow
    uv sync --frozen

    uv run torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv_id="${RDZV_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m src.training.train_flow "${ARGS[@]}" | tee "./checkpoints/$exp_name/log.out"
fi