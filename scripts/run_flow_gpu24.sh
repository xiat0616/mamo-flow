#!/bin/bash

base_name="${1:-flow}"

# ----------------------------
# define variables first
# ----------------------------
img_height=256
img_width=192
cond_embedder="per_attr"
model_channels=32
p_uncond=0.2

# optional: other useful vars
img_channels=1
epochs=10000
bs=24
lr=1e-4

exp_name="${base_name}_flow_embed_${img_height}_${img_width}_condemb_${cond_embedder}_mchannel_${model_channels}_puncond_${p_uncond}"

mkdir -p ../checkpoints
mkdir -p "../checkpoints/$exp_name"   # must be unique

ARGS=(
# DATA
    --data_dir="/vol/biodata/data/Mammo/EMBED/pngs/1024x768"
    --csv_filepath="/vol/biomedic3/tx1215/mamo-flow/assets/EMBED_meta.csv"
    --save_dir="./checkpoints/$exp_name"
    --parents age view density scanner cview
    --exclude_cviews=1
    --hold_out_model_5=1
    --prop_train=1.0
    --valid_frac=0.075
    --test_frac=0.125
    --split_seed=33
    --img_height=$img_height
    --img_width=$img_width
    --img_channels=$img_channels

# TRAIN
    --resume=
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

# FLOW
    --alpha=1.0
    --sigma=0.0
    --T=150
    --p_uncond=$p_uncond
    --cond_embedder=$cond_embedder

# MODEL
    unet
    --model_channels=$model_channels
    --channel_mult 1 2 3 4 5
    --cond_embed_dim=160
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

# # TO RESUME
# ARGS=(
#     --resume="../checkpoints/flow_dit/best_checkpoint.pt"
#     --exp_name="$exp_name"
#     --save_dir="../checkpoints/$exp_name"
# )

# export NCCL_P2P_LEVEL=LOC  # comment for lora
#SBATCH --nodelist=lora,luna
#SBATCH --nodelist=monal04
#SBATCH --exclude=loki

NPROC_PER_NODE=3

if [ "$2" == "slurm" ]; then
    sbatch <<EOF
#!/bin/bash

#SBATCH --partition=gpus
#SBATCH --gres=gpu:3
#SBATCH --nodelist=monal04
#SBATCH --output=../checkpoints/$exp_name/slurm.%j.log

cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen

nvidia-smi
export OMP_NUM_THREADS=3
export TQDM_MININTERVAL=300
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)  # select random port in range

srun uv run torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_id="\$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \
    ./src/training/train_flow.py ${ARGS[@]} 2>&1 | tee "./checkpoints/$exp_name/log.out"
EOF
else
    NPROC_PER_NODE=8
    RDZV_ID="${RDZV_ID:-$(date +%s)-$$}"
    export OMP_NUM_THREADS=1
    export TQDM_MININTERVAL=300
    export MASTER_ADDR=localhost
    export MASTER_PORT=$(shuf -i 10001-29500 -n 1)
    # export WANDB_MODE="disabled"

    uv run torchrun \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        --rdzv_id="$RDZV_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        ../src/training/train_flow.py "${ARGS[@]}" 2>&1 | tee "../checkpoints/$exp_name/log.out"
fi