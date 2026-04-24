#!/bin/bash
exp_name="mammo128_gpus48_improved_meanflow_embed_128_96_condemb_per_attr_mchannel_64_puncond_0.2_rneqt_0.25_lognorm"
ckpt_file="best_checkpoint.pt"

mode="${1:-rs}"                              # rs or cf
do_key="${2:-none}"                          # for cf key: view, cview, density
do_mode="${3:-flip}"                         # for cf mode: flip, null, or random

project_root="/vol/biomedic3/tx1215/mamo-flow"
save_root="${project_root}/sampling_results"
ckpt="${project_root}/checkpoints/${exp_name}/${ckpt_file}"
split_dir="${project_root}/assets/embed_splits_v1"

if [ ! -f "$ckpt" ]; then
    echo "Checkpoint not found: $ckpt"
    exit 1
fi

# ----------------------------
# Sampling config
# ----------------------------
num_samples=60
batch_size=4
split="valid"
use_ema=1
cond_source="dataset"
seed=0

# ----------------------------
# Improved MeanFlow sampling config
# ----------------------------
sample_steps=5

# CFG for ImprovedMeanFlow.sample(...)
cfg_mode="none"         # none or cfg
cfg_scale="1.0"
null_keys=()            # e.g. ("density") or ("view" "density")

# ----------------------------
# Experimental CF inversion config
# Only used when mode=cf
# ----------------------------
# Adaptive solver example:
# ode_method="dopri5"
# ode_atol="1e-5"
# ode_rtol="1e-5"
# ode_steps=""

# Fixed-step solver example:
ode_method="midpoint"
ode_atol=""
ode_rtol=""
ode_steps="1000"

# -------------------------------
# Helper: match Python _format_float_tag()
# -------------------------------
format_float_tag() {
    python3 - "$1" <<'PY'
import sys
x = float(sys.argv[1])
s = f"{x:.0e}"
s = s.replace("e-0", "e-").replace("e+0", "e+")
print(s)
PY
}

ckpt_tag="${ckpt_file%.*}"

# -------------------------------
# Match sample_improved_meanflow.py tag logic
# -------------------------------
cfg_tag="no_cfg"
if [ "$cfg_mode" != "none" ]; then
    cfg_tag="${cfg_mode}_w-${cfg_scale}"
fi

if [ "${#null_keys[@]}" -gt 0 ]; then
    null_joined="$(printf '%s\n' "${null_keys[@]}" | sort | paste -sd '_' -)"
    cfg_tag="${cfg_tag}_null-${null_joined}"
fi

rs_sampler_tag="imf_steps-${sample_steps}_${cfg_tag}"

if [ -n "$ode_steps" ]; then
    inv_tag="invode-${ode_method}_steps-${ode_steps}"
else
    ode_atol_tag="$(format_float_tag "$ode_atol")"
    ode_rtol_tag="$(format_float_tag "$ode_rtol")"
    inv_tag="invode-${ode_method}_atol-${ode_atol_tag}_rtol-${ode_rtol_tag}"
fi

cf_sampler_tag="${inv_tag}_imf_steps-${sample_steps}_${cfg_tag}"

if [ "$mode" = "rs" ]; then
    sampler_tag="$rs_sampler_tag"

    cond_tag="cond_dataset"
    if [ "$cond_source" = "none" ]; then
        cond_tag="uncond"
    fi

    run_dir="${save_root}/${exp_name}/${ckpt_tag}/${sampler_tag}/randoms/${cond_tag}"

elif [ "$mode" = "cf" ]; then
    sampler_tag="$cf_sampler_tag"

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
    --ckpt "$ckpt"
    --save_dir "$save_root"
    --split_dir "$split_dir"
    --num_samples "$num_samples"
    --batch_size "$batch_size"
    --seed "$seed"
    --split "$split"
    --mode "$mode"
    --cond_source "$cond_source"
    --sample_steps "$sample_steps"
    --cfg_mode "$cfg_mode"
    --cfg_scale "$cfg_scale"
)

if [ "${#null_keys[@]}" -gt 0 ]; then
    ARGS+=(--null_keys "${null_keys[@]}")
fi

if [ "$use_ema" = "1" ]; then
    ARGS+=(--use_ema)
fi

if [ "$mode" = "cf" ]; then
    ARGS+=(--do_mode "$do_mode")
    ARGS+=(--ode_method "$ode_method")

    if [ -n "$ode_steps" ]; then
        ARGS+=(--ode_steps "$ode_steps")
    else
        ARGS+=(--ode_atol "$ode_atol")
        ARGS+=(--ode_rtol "$ode_rtol")
    fi

    if [ "$do_mode" != "null" ] && [ "$do_key" != "none" ]; then
        ARGS+=(--do_key "$do_key")
    fi
fi

cmd=$(printf '%q ' uv run python -m src.sampling.sample_improved_meanflow "${ARGS[@]}")
cmd="${cmd% }"

sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpus48,gpus24,gpus
#SBATCH --gres=gpu:1
#SBATCH --exclude=monal04,monal05
#SBATCH --output=${run_dir}/slurm.%j.out

cd "${project_root}"
uv sync --frozen
source ~/.bashrc

nvidia-smi

${cmd} | tee "${run_dir}/sample.log"
EOF