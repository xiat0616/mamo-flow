#!/bin/bash

# ============================================================
# Experiment
# ============================================================

# exp_name="embed_latent_flow_small_bs_flux2_dit_b2_64_48_condemb_per_attr_cdim_160_puncond_0.2"
# exp_name="embed_latent_flow_xl_flux2_dit_b4_128_96_condemb_per_attr_cdim_160_puncond_0.2"
exp_name="embed_latent_flow_flux2_dit_b4_128_96_condemb_per_attr_cdim_160_puncond_0.2"

ckpt_file="last_checkpoint.pt"

# Usage:
#
#   bash run_paired_view_cf_latent.sh CC
#   bash run_paired_view_cf_latent.sh MLO
#
# Optional split:
#
#   bash run_paired_view_cf_latent.sh CC test
#   bash run_paired_view_cf_latent.sh MLO valid

source_view="${1:?Please provide source view: CC or MLO}"
split="${2:-test}"


# ============================================================
# Direction
# ============================================================

if [ "$source_view" = "CC" ]; then
    target_view="MLO"
elif [ "$source_view" = "MLO" ]; then
    target_view="CC"
else
    echo "Unknown source_view: $source_view"
    echo "Please use: CC or MLO"
    exit 1
fi


# ============================================================
# Paths
# ============================================================

project_root="/vol/biomedic3/tx1215/mamo-flow"

ckpt="${project_root}/checkpoints/${exp_name}/${ckpt_file}"

save_root="${project_root}/sampling_results"

split_dir="${project_root}/assets/embed_splits_v1"

# IMPORTANT:
# Must be the SAME latent cache used during training.
# cache_dir="${project_root}/cache/flux2_vae_512x384"
cache_dir="${project_root}/cache/flux2_vae_1024x768"
pair_csv="${cache_dir}/pair_csv_latent.csv"


# ============================================================
# Check files
# ============================================================

if [ ! -f "$ckpt" ]; then
    echo "Checkpoint not found:"
    echo "$ckpt"
    exit 1
fi

if [ ! -d "$cache_dir" ]; then
    echo "Latent cache directory not found:"
    echo "$cache_dir"
    exit 1
fi

if [ ! -f "$pair_csv" ]; then
    echo "Latent pair CSV not found:"
    echo "$pair_csv"
    exit 1
fi

for split_name in train valid test; do
    latent_file="${cache_dir}/flux2encoding_float32_${split_name}.dat"
    manifest_file="${cache_dir}/${split_name}_manifest.csv"

    if [ ! -f "$latent_file" ]; then
        echo "Latent cache not found:"
        echo "$latent_file"
        exit 1
    fi

    if [ ! -f "$manifest_file" ]; then
        echo "Latent manifest not found:"
        echo "$manifest_file"
        exit 1
    fi
done


# ============================================================
# Sampling config
# ============================================================

num_samples=500
seed=0

# 1 = EMA
# 0 = raw model
use_ema=1

# 0 = first N pairs
# 1 = random N pairs
shuffle=0


# ============================================================
# ODE config
# ============================================================

# Adaptive solver
ode_method="dopri5"
ode_atol="1e-5"
ode_rtol="1e-5"
ode_steps=""

# Fixed-step alternative:
#
# ode_method="midpoint"
# ode_atol=""
# ode_rtol=""
# ode_steps="150"


# ============================================================
# Helper: match Python sampler float formatting
# ============================================================

format_float_tag() {
python3 - "$1" <<'PY'
import sys

x = float(sys.argv[1])
s = f"{x:.0e}"
s = s.replace("e-0", "e-").replace("e+0", "e+")

print(s)
PY
}


# ============================================================
# Sampling folder name
#
# Must match build_sampling_root() in Python.
# ============================================================

ckpt_tag="${ckpt_file%.*}"

if [ -n "$ode_steps" ]; then
    sampler_tag="ode-${ode_method}_steps-${ode_steps}"
else
    ode_atol_tag="$(format_float_tag "$ode_atol")"
    ode_rtol_tag="$(format_float_tag "$ode_rtol")"

    sampler_tag="ode-${ode_method}_atol-${ode_atol_tag}_rtol-${ode_rtol_tag}"
fi


# ============================================================
# Final sampling directory
#
# Python saves:
#
#   cfs/
#   inputs/
#   reconstructions/
#   ground_truth/
#   visuals/
#   samples.csv
#   sampling_args.json
#
# SLURM logs are also saved here.
# ============================================================

run_dir="${save_root}/${exp_name}/${ckpt_tag}/${sampler_tag}/paired_view_cfs/${source_view}_to_${target_view}/${split}"

mkdir -p "$run_dir"


# ============================================================
# Python arguments
# ============================================================

ARGS=(
    --ckpt "$ckpt"
    --save_dir "$save_root"
    --split_dir "$split_dir"
    --pair_csv "$pair_csv"
    --num_samples "$num_samples"
    --source_view "$source_view"
    --split "$split"
    --seed "$seed"
    --use_ema "$use_ema"
    --ode_method "$ode_method"
)


# ============================================================
# ODE arguments
# ============================================================

if [ -n "$ode_steps" ]; then
    ARGS+=(
        --ode_steps "$ode_steps"
    )
else
    ARGS+=(
        --ode_atol "$ode_atol"
        --ode_rtol "$ode_rtol"
    )
fi


# ============================================================
# Shuffle
# ============================================================

if [ "$shuffle" = "1" ]; then
    ARGS+=(--shuffle)
fi


# ============================================================
# Build command
# ============================================================

cmd=$(printf '%q ' \
    uv run python -m src.sampling.sample_paired_view_cf \
    "${ARGS[@]}"
)

cmd="${cmd% }"


# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "Latent-flow paired CC/MLO counterfactual sampling"
echo "============================================================"
echo "Experiment      : $exp_name"
echo "Direction       : ${source_view} -> ${target_view}"
echo "Split           : $split"
echo "Num samples     : $num_samples"
echo "Use EMA         : $use_ema"
echo "Shuffle         : $shuffle"
echo
echo "ODE method      : $ode_method"
echo "ODE steps       : ${ode_steps:-adaptive}"
echo "ODE atol        : ${ode_atol:-N/A}"
echo "ODE rtol        : ${ode_rtol:-N/A}"
echo
echo "Checkpoint      : $ckpt"
echo "Split dir       : $split_dir"
echo "Latent cache    : $cache_dir"
echo "Latent pair CSV : $pair_csv"
echo
echo "Output folder   : $run_dir"
echo "============================================================"


# ============================================================
# Submit SLURM
# ============================================================

sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=latent_cf_${source_view}
#SBATCH --partition=gpus48,gpus24
#SBATCH --gres=gpu:1
#SBATCH --exclude=monal04,monal05
#SBATCH --output=${run_dir}/slurm.%j.out

source ~/.bashrc

cd "${project_root}"

uv sync --frozen

export OMP_NUM_THREADS=1
export TQDM_MININTERVAL=300
export WANDB_MODE=disabled

echo "============================================================"
echo "Host: \$(hostname)"
echo "Job:  \$SLURM_JOB_ID"
echo "============================================================"

nvidia-smi

echo
echo "Running:"
echo "${cmd}"
echo

${cmd} 2>&1 | tee "${run_dir}/sample.log"

EOF