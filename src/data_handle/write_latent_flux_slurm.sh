#!/bin/bash
set -euo pipefail

# ============================================================
# Config
# ============================================================

project_root="/vol/biomedic3/tx1215/mamo-flow"

split_dir="${project_root}/assets/embed_splits_v1"
data_dir="/vol/biodata/data/Mammo/EMBED/pngs/1024x768"
out_dir="${project_root}/cache/flux2_vae_512x384_2"

img_height=512
img_width=384

batch_size=16
num_workers=8

# Slurm log directory
log_dir="${out_dir}/logs"
mkdir -p "${log_dir}"


# ============================================================
# Python arguments
# ============================================================

ARGS=(
    --split_dir "${split_dir}"
    --data_dir "${data_dir}"
    --out_dir "${out_dir}"
    --device cuda:0
    --batch_size "${batch_size}"
    --num_workers "${num_workers}"
    --img_height "${img_height}"
    --img_width "${img_width}"
)

# Uncomment if you intentionally want to overwrite existing caches:
# ARGS+=(--overwrite)


# ============================================================
# Build command
# ============================================================

cmd=$(printf '%q ' python write_latent_cache.py "${ARGS[@]}")
cmd="${cmd% }"


# ============================================================
# Submit Slurm job
# ============================================================

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=embed_flux2_cache
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --exclude=monal04,monal05
#SBATCH --output=${log_dir}/slurm.%j.out

set -euo pipefail

source "${project_root}/.venv/bin/activate"

echo "========================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Date: \$(date)"
echo "========================================"

nvidia-smi

echo
echo "Python:"
which python
python --version

echo
echo "Starting FLUX.2 latent cache generation..."
echo

${cmd} 2>&1 | tee "${log_dir}/write_latent_cache.\${SLURM_JOB_ID}.log"

echo
echo "========================================"
echo "Latent cache generation finished."
echo "Date: \$(date)"
echo "========================================"
EOF