#!/bin/bash
#SBATCH --job-name=crt-s5-h2h
#SBATCH --output=slurm-stage5-h2h-%A_%a.out
#SBATCH --error=slurm-stage5-h2h-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --array=0-3

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/RolloutTransformer}"
CONFIG_LIST="${CONFIG_LIST:-src/configs/stage5_context/experiments.txt}"
OXFORD_CSV="${OXFORD_CSV:-data/oxford/oxford_panel_drop_negligible_thr5.csv}"
DEVICE="${DEVICE:-cpu}"

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
CKPT_PATH="checkpoints/oxford_stage5_context/${EXP_NAME}/best_crt.pt"
OUT_DIR="results/stage5_context_h2h_with_baselines/${EXP_NAME}"

echo "Running Stage 5 baseline eval: $EXP_NAME"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "Oxford CSV: $OXFORD_CSV"

python evaluate_oxford_extended.py \
  --model "crt=${CKPT_PATH}" \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --output_dir "$OUT_DIR" \
  --short_horizon_end 7 \
  --long_horizon_start 22 \
  --late_horizon_start 32 \
  --clip_nonnegative \
  --drop_negligible_countries \
  --negligible_outcome new_cases_smoothed_per_million \
  --negligible_history_len 21 \
  --negligible_threshold 5.0 \
  --save_trajectory_plots \
  --trajectory_focus_model crt \
  --trajectory_reference_model persistence \
  --trajectory_samples_per_group 20 \
  --save_regime_breakdown \
  --save_incidence_regime_metrics \
  --save_shape_metrics \
  --save_stability_metrics \
  --save_utility_metrics \
  --include_ml_baselines \
  --ml_skip_boosted \
  --ml_include_country_context \
  --include_lstm_baseline \
  --lstm_include_country_context \
  --device "$DEVICE"

echo "Done: $EXP_NAME"
