#!/bin/bash
#SBATCH --job-name=crt-s3f
#SBATCH --output=slurm-stage3f-%A_%a.out
#SBATCH --error=slurm-stage3f-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --account=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --array=0-4

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/RolloutTransformer}"
CONFIG_LIST="${CONFIG_LIST:-src/configs/stage3_filtered/experiments.txt}"
OXFORD_CSV="${OXFORD_CSV:-data/oxford/oxford_panel_drop_negligible_thr5.csv}"
DEVICE="${DEVICE:-cuda}"

cd "$REPO_DIR"

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate crt311
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate crt311
fi

N_CFG=$(wc -l < "$CONFIG_LIST")
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_CFG" ]]; then
  echo "Array index $SLURM_ARRAY_TASK_ID is out of range for $N_CFG configs"
  exit 1
fi

CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
EXP_NAME=$(basename "$CONFIG_PATH" .yaml)
CKPT_DIR="checkpoints/oxford_stage3_filtered/${EXP_NAME}"
EXT_RESULT_DIR="results/stage3_extended_filtered/${EXP_NAME}"

echo "Running Stage 3 filtered experiment: $EXP_NAME"
echo "Config: $CONFIG_PATH"
echo "Oxford CSV: $OXFORD_CSV"

python train_oxford.py \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --checkpoint_dir "$CKPT_DIR" \
  --model_type crt \
  --device "$DEVICE"

python evaluate_oxford_extended.py \
  --model "crt=${CKPT_DIR}/best_crt.pt" \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --output_dir "$EXT_RESULT_DIR" \
  --short_horizon_end 7 \
  --long_horizon_start 22 \
  --late_horizon_start 32 \
  --clip_nonnegative \
  --no_baselines \
  --drop_negligible_countries \
  --negligible_outcome new_cases_smoothed_per_million \
  --negligible_history_len 21 \
  --negligible_threshold 5.0 \
  --save_incidence_regime_metrics \
  --save_regime_breakdown \
  --device "$DEVICE"

echo "Done: $EXP_NAME"
