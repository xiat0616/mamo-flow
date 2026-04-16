#!/bin/bash

cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen

sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:3
#SBATCH --nodelist=loki
#SBATCH --output=./test_loki/slurm.%j.log

cd /vol/biomedic3/tx1215/mamo-flow
uv sync --frozen

nvidia-smi

export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$(shuf -i 10001-29500 -n 1)

srun uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=3 \
    --rdzv_id="\$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \
    ./test_loki/test_ddp_init.py 2>&1 | tee "./test_loki/log.out"
EOF
