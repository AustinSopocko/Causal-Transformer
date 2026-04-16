#!/bin/bash
#SBATCH --job-name=crt-s4-seeds
#SBATCH --output=slurm-stage4-seeds-%A_%a.out
#SBATCH --error=slurm-stage4-seeds-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --account=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --array=0-2

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/RolloutTransformer}"
CONFIG_PATH="${CONFIG_PATH:-src/configs/stage4_h35_week46/03_huber_anchor02_nonneg.yaml}"
OXFORD_CSV="${OXFORD_CSV:-data/oxford/oxford_panel.csv}"
DEVICE="${DEVICE:-cuda}"
SEEDS_CSV="${SEEDS_CSV:-11,22,33}"
OUT_CKPT_ROOT="${OUT_CKPT_ROOT:-checkpoints/oxford_stage4_seed_stability}"
OUT_EXT_ROOT="${OUT_EXT_ROOT:-results/stage4_seed_stability}"

cd "$REPO_DIR"

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate crt311
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate crt311
fi

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]]; then
  echo "Array index $SLURM_ARRAY_TASK_ID out of range for seeds: $SEEDS_CSV"
  exit 1
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="c03_seed${SEED}"
CKPT_DIR="${OUT_CKPT_ROOT}/${EXP_NAME}"
EXT_DIR="${OUT_EXT_ROOT}/${EXP_NAME}"

echo "Running seed stability experiment: $EXP_NAME"
echo "Config: $CONFIG_PATH"
echo "Oxford CSV: $OXFORD_CSV"
echo "Device: $DEVICE"

python train_oxford.py \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --checkpoint_dir "$CKPT_DIR" \
  --model_type crt \
  --seed "$SEED" \
  --device "$DEVICE"

python evaluate_oxford_extended.py \
  --model "${EXP_NAME}=${CKPT_DIR}/best_crt.pt" \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --output_dir "$EXT_DIR" \
  --short_horizon_end 7 \
  --long_horizon_start 22 \
  --late_horizon_start 32 \
  --clip_nonnegative \
  --drop_negligible_countries \
  --negligible_outcome new_cases_smoothed_per_million \
  --negligible_history_len 21 \
  --negligible_threshold 5.0 \
  --device "$DEVICE"

echo "Done: $EXP_NAME"
