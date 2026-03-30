#!/bin/bash
#SBATCH --job-name=crt-s2
#SBATCH --output=slurm-stage2-%A_%a.out
#SBATCH --error=slurm-stage2-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-7

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/RolloutTransformer}"
CONFIG_LIST="${CONFIG_LIST:-src/configs/stage2/experiments.txt}"
OXFORD_CSV="${OXFORD_CSV:-data/oxford/oxford_panel.csv}"
DEVICE="${DEVICE:-cuda}"

cd "$REPO_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
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
CKPT_DIR="checkpoints/oxford_stage2/${EXP_NAME}"
RESULT_DIR="results/stage2/${EXP_NAME}"
EXT_RESULT_DIR="results/stage2_extended/${EXP_NAME}"

echo "Running Stage 2 experiment: $EXP_NAME"
echo "Config: $CONFIG_PATH"

python train_oxford.py \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --checkpoint_dir "$CKPT_DIR" \
  --model_type crt \
  --device "$DEVICE"

python evaluate_oxford.py \
  --checkpoint "${CKPT_DIR}/best_crt.pt" \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --output_dir "$RESULT_DIR" \
  --device "$DEVICE" \
  --plot

python evaluate_oxford_extended.py \
  --model "crt=${CKPT_DIR}/best_crt.pt" \
  --oxford_csv "$OXFORD_CSV" \
  --config "$CONFIG_PATH" \
  --output_dir "$EXT_RESULT_DIR" \
  --device "$DEVICE"

echo "Done: $EXP_NAME"
