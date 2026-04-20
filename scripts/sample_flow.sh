#!/bin/bash
exp_name="mammo256_loki_flow_embed_256_192_condemb_per_attr_mchannel_64_puncond_0.2"
ckpt_file="best_checkpoint.pt"
mode="${1:?Please provide mode: random or cf}"   # random or cf
do_key="${2:-none}"                              # for cf key: view, cview, density
do_mode="${3:-flip}"                             # for cf mode: flip, null, or random

project_root="/vol/biomedic3/tx1215/mamo-flow"
save_root="${project_root}/sampling_results"
ckpt="${project_root}/checkpoints/${exp_name}/${ckpt_file}"

if [ ! -f "$ckpt" ]; then
    echo "Checkpoint not found: $ckpt"
    exit 1
fi

# ----------------------------
# Sampling config
# ----------------------------
num_samples=16
batch_size=4
split="test"
use_ema=1
cond_source="dataset"

ode_method="dopri5"
ode_atol="1e-5"
ode_rtol="1e-5"
ode_steps=""   # e.g. 100 ; leave empty to match Python's None

# ----------------------------
# Helper: match Python _format_float_tag()
# ----------------------------
format_float_tag() {
    python - "$1" <<'PY'
import sys
x = float(sys.argv[1])
s = f"{x:.0e}"
s = s.replace("e-0", "e-").replace("e+0", "e+")
print(s)
PY
}

ode_atol_tag="$(format_float_tag "$ode_atol")"
ode_rtol_tag="$(format_float_tag "$ode_rtol")"

ckpt_tag="${ckpt_file%.*}"
sampler_tag="ode-${ode_method}_atol-${ode_atol_tag}_rtol-${ode_rtol_tag}"
if [ -n "$ode_steps" ]; then
    sampler_tag="${sampler_tag}_steps-${ode_steps}"
fi

if [ "$mode" = "random" ]; then
    cond_tag="cond_dataset"
    if [ "$cond_source" = "none" ]; then
        cond_tag="uncond"
    fi
    run_dir="${save_root}/${exp_name}/${ckpt_tag}/${sampler_tag}/randoms/${cond_tag}"
elif [ "$mode" = "cf" ]; then
    if [ "$do_mode" = "null" ]; then
        run_dir="${save_root}/${exp_name}/${ckpt_tag}/${sampler_tag}/reconstructions/null"
    else
        run_dir="${save_root}/${exp_name}/${ckpt_tag}/${sampler_tag}/cfs/${do_key}/${do_mode}"
    fi
else
    echo "Unknown mode: $mode"
    exit 1
fi

mkdir -p "$run_dir"

ARGS=(
    --ckpt="$ckpt"
    --save_dir="$save_root"
    --num_samples="$num_samples"
    --batch_size="$batch_size"
    --seed=0
    --split="$split"
    --ode_method="$ode_method"
    --ode_atol="$ode_atol"
    --ode_rtol="$ode_rtol"
    --mode="$mode"
    --cond_source="$cond_source"
)

if [ -n "$ode_steps" ]; then
    ARGS+=(--ode_steps="$ode_steps")
fi

if [ "$use_ema" = "1" ]; then
    ARGS+=(--use_ema)
fi

if [ "$mode" = "cf" ]; then
    ARGS+=(--do_mode="$do_mode")
    if [ "$do_mode" != "null" ] && [ "$do_key" != "none" ]; then
        ARGS+=(--do_key="$do_key")
    fi
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus48,gpus24,gpus
#SBATCH --gres=gpu:1
#SBATCH --output=${run_dir}/slurm.%j.out

cd "${project_root}"
uv sync --frozen
source ~/.bashrc

nvidia-smi

uv run python -m src.sampling.sample_flow ${ARGS[@]} | tee "${run_dir}/sample.log"
EOF