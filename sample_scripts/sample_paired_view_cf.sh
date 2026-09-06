#!/bin/bash

# ============================================================
# Experiment
# ============================================================

exp_name="embed_flow_debug_flip_density_128_96_condemb_per_attr_mchannel_32_puncond_0.2"
ckpt_file="last_checkpoint.pt"

# Usage:
#   bash run_paired_view_cf.sh CC
#   bash run_paired_view_cf.sh MLO
#
# Optional split:
#   bash run_paired_view_cf.sh CC test
#   bash run_paired_view_cf.sh MLO valid

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

save_root="${project_root}/sampling_results"

ckpt="${project_root}/checkpoints/${exp_name}/${ckpt_file}"

split_dir="${project_root}/assets/embed_splits_v1"

pair_csv="${split_dir}/EMBED_same_breast_CC_MLO_pairs.csv"

python_script="${project_root}/src/sampling/sample_paired_view_cf.py"


# ============================================================
# Check files
# ============================================================

if [ ! -f "$ckpt" ]; then
    echo "Checkpoint not found:"
    echo "$ckpt"
    exit 1
fi

if [ ! -f "$pair_csv" ]; then
    echo "Pair CSV not found:"
    echo "$pair_csv"
    exit 1
fi

if [ ! -f "$python_script" ]; then
    echo "Python script not found:"
    echo "$python_script"
    exit 1
fi


# ============================================================
# Sampling config
# ============================================================

num_samples=500
seed=0

# 1 = EMA
# 0 = raw model
use_ema=1

# 0 = first N pairs
# 1 = randomly sample N pairs
shuffle=0


# ============================================================
# ODE config
# ============================================================

# ----------------------------
# Adaptive solver
# ----------------------------

ode_method="dopri5"
ode_atol="1e-5"
ode_rtol="1e-5"
ode_steps=""


# ----------------------------
# Fixed-step solver
# ----------------------------

# ode_method="midpoint"
# ode_atol=""
# ode_rtol=""
# ode_steps="50"


# ============================================================
# Helper: same formatting as Python sampler
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
# Build sampler tag
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
# Output directory
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


# ODE arguments
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


# Randomly choose pairs
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
echo "Paired CC/MLO counterfactual sampling"
echo "============================================================"
echo "Experiment   : $exp_name"
echo "Direction    : ${source_view} -> ${target_view}"
echo "Split        : $split"
echo "Num samples  : $num_samples"
echo "Use EMA      : $use_ema"
echo "Shuffle      : $shuffle"
echo "ODE method   : $ode_method"
echo "ODE steps    : $ode_steps"
echo "Checkpoint   : $ckpt"
echo "Pair CSV     : $pair_csv"
echo "Output       : $run_dir"
echo "============================================================"


# ============================================================
# Submit SLURM
# ============================================================

sbatch <<EOF
#!/bin/bash

#SBATCH --partition=gpus48,gpus24
#SBATCH --gres=gpu:1
#SBATCH --exclude=monal04,monal05
#SBATCH --output=${run_dir}/slurm.%j.out

cd "${project_root}"

uv sync --frozen

source ~/.bashrc

nvidia-smi

echo "Running:"
echo "${cmd}"

${cmd} 2>&1 | tee "${run_dir}/sample.log"

EOF