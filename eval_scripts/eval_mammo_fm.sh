#!/bin/bash

# ============================================================
# Mammo-FM counterfactual evaluation
# ============================================================

project_root="/vol/biomedic3/tx1215/mamo-flow"

run_dir="${project_root}/sampling_results/embed_latent_flow_flux2_dit_b2_128_96_condemb_per_attr_cdim_160_puncond_0.2/last_checkpoint/ode-dopri5_atol-1e-5_rtol-1e-5/paired_view_cfs/MLO_to_CC/test"

batch_size=4
num_workers=4
bootstrap_samples=5000
seed=0
save_features=0


# ============================================================
# Check
# ============================================================

if [ ! -d "$run_dir" ]; then
    echo "Run directory not found:"
    echo "$run_dir"
    exit 1
fi

if [ ! -f "${run_dir}/samples.csv" ]; then
    echo "samples.csv not found:"
    echo "${run_dir}/samples.csv"
    exit 1
fi


# ============================================================
# Python arguments
# ============================================================

ARGS=(
    --run_dir "$run_dir"
    --batch_size "$batch_size"
    --num_workers "$num_workers"
    --bootstrap_samples "$bootstrap_samples"
    --seed "$seed"
    --save_features "$save_features"
)


# ============================================================
# Build command
# ============================================================

cmd=$(printf '%q ' \
    uv run python -m src.evaluation.eval_mammo_fm_cf \
    "${ARGS[@]}"
)

cmd="${cmd% }"


# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "Mammo-FM counterfactual evaluation"
echo "============================================================"
echo "Run directory      : $run_dir"
echo "Batch size         : $batch_size"
echo "Num workers        : $num_workers"
echo "Bootstrap samples  : $bootstrap_samples"
echo "Seed               : $seed"
echo "Save features      : $save_features"
echo "============================================================"


# ============================================================
# Submit SLURM
# ============================================================

sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=mammo_fm_eval
#SBATCH --partition=gpus48,gpus24
#SBATCH --gres=gpu:1
#SBATCH --exclude=monal04,monal05
#SBATCH --output=${run_dir}/eval_mammo_fm_slurm.%j.out

source ~/.bashrc

cd "${project_root}"

uv sync --frozen

export OMP_NUM_THREADS=1
export TQDM_MININTERVAL=300
export WANDB_MODE=disabled

echo "============================================================"
echo "Host: \$(hostname)"
echo "Job:  \$SLURM_JOB_ID"
echo "Run directory: ${run_dir}"
echo "============================================================"

nvidia-smi

echo
echo "Running:"
echo "${cmd}"
echo

${cmd} 2>&1 | tee "${run_dir}/eval_mammo_fm.log"

EOF