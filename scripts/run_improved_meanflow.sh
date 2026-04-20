#!/bin/bash

base_name="${1:-improved_meanflow}"

# ----------------------------
# Core experiment config
# ----------------------------
img_height=256
img_width=192
img_channels=1

cond_embedder="per_attr"
model_channels=64
cond_embed_dim=160
p_uncond=0.2

epochs=10000
bs=16
lr=1e-4

# ----------------------------
# iMF training hyperparams
# Note: these still use the mf_* flag names because
# train_improved_meanflow.py currently parses the shared
# MeanFlowConfig argument names.
# ----------------------------
mf_ratio_r_neq_t=0.25
mf_time_sampler="lognorm"
mf_lognorm_mu=-0.4
mf_lognorm_sigma=1.0
mf_adaptive_weight_p=1.0
mf_adaptive_weight_eps=1e-3

# Eval-time sampling intervals for plots only.
# This is NOT the training discretization.
sample_steps=1

exp_name="${base_name}_improved_meanflow_embed_${img_height}_${img_width}_condemb_${cond_embedder}_mchannel_${model_channels}_puncond_${p_uncond}_rneqt_${mf_ratio_r_neq_t}_${mf_time_sampler}"

mkdir -p ../checkpoints
mkdir -p "../checkpoints/$exp_name"

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
    --img_height="$img_height"
    --img_width="$img_width"
    --img_channels="$img_channels"
# TRAIN
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
# SAMPLING / CFG
    --sample_steps="$sample_steps"
    --p_uncond="$p_uncond"
    --cond_embedder="$cond_embedder"
# MEANFLOW / IMF CONFIG
    --mf_ratio_r_neq_t="$mf_ratio_r_neq_t"
    --mf_time_sampler="$mf_time_sampler"
    --mf_lognorm_mu="$mf_lognorm_mu"
    --mf_lognorm_sigma="$mf_lognorm_sigma"
    --mf_adaptive_weight_p="$mf_adaptive_weight_p"
    --mf_adaptive_weight_eps="$mf_adaptive_weight_eps"
# MODEL
    unet
    --model_channels="$model_channels"
    --channel_mult 1 2 3 4 5
    --cond_embed_dim="$cond_embed_dim"
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

NPROC_PER_NODE=2

#SBATCH --nodelist=mira05

if [ "$2" = "gpus48" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=../checkpoints/$exp_name/slurm.%j.log

cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen

source ~/.bashrc

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
    ./src/training/train_improved_meanflow.py ${ARGS[@]} | tee "./checkpoints/$exp_name/log.out"
EOF

elif [ "$2" = "gpus" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:${NPROC_PER_NODE}
#SBATCH --output=../checkpoints/$exp_name/slurm.%j.log

cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen
source ~/.bashrc

nvidia-smi
export OMP_NUM_THREADS=3
export TQDM_MININTERVAL=300
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)

srun uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_id="\$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \
    ./src/training/train_improved_meanflow.py ${ARGS[@]} | tee "./checkpoints/$exp_name/log.out"
EOF

else
    NPROC_PER_NODE=8
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
        ./src/training/train_improved_meanflow.py ${ARGS[@]} | tee "./checkpoints/$exp_name/log.out"
fi