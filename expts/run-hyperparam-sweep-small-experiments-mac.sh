#!/usr/bin/env bash
set -euo pipefail

# CPU/MPS-friendly version of the original NVIDIA 8-GPU sweep.
# Runs experiments sequentially on a single Mac (CPU or Apple Silicon MPS).
#
# Usage:
#   bash expts/run-hyperparam-sweep-small-experiments-mac.sh
#
# Notes:
# - Disables wandb by default.
# - Uses smaller batch size / workers for laptop-friendly runs.

BASE_PORT=29100

datasets=("rel-avito" "rel-f1" "rel-event" "rel-trial" "rel-f1" "rel-f1" "rel-avito" "rel-avito" "rel-event" "rel-event" "rel-trial" "rel-trial")
tasks=("ad-ctr" "driver-dnf" "user-repeat" "study-outcome" "driver-position" "driver-top3" "user-clicks" "user-visits" "user-attendance" "user-ignore" "study-adverse" "site-success")

# Dropout and num_layers combos
# (keep the same grid as upstream; adjust if it takes too long)
dropouts=(0.3 0.4 0.5)
num_layers=(1 4 8)

# Mac-friendly defaults
batch_size=64
max_steps_per_epoch=200
num_workers=0
warmup_steps=50
epochs_default=3
lr=0.0001

mkdir -p results

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

expt=0
for d in "${dropouts[@]}"; do
  for l in "${num_layers[@]}"; do
    for i in "${!datasets[@]}"; do
      dataset="${datasets[$i]}"
      task="${tasks[$i]}"
      dro="${d//./}" # 0.3 -> 03

      epochs="$epochs_default"
      if [[ "$dataset" == "rel-trial" && "$task" == "site-success" ]]; then
        epochs=2
      fi

      run_name="relgt-mac-l${l}-512-dropout${dro}-BS${batch_size}-MAX${max_steps_per_epoch}"
      out_dir="results/${run_name}/${dataset}__${task}"
      mkdir -p "$out_dir"

      log "(expt $expt) Running $dataset ($task) dropout=$d layers=$l"

      # We avoid torchrun/DDP on macOS. main_node_ddp.py was patched to init a 1-process group.
      WANDB_MODE=disabled \
      MASTER_PORT=$((BASE_PORT + expt)) \
      python main_node_ddp.py \
        --dataset "$dataset" \
        --task "$task" \
        --seed 0 \
        --batch_size "$batch_size" \
        --num_neighbors 300 \
        --num_layers "$l" \
        --channels 512 \
        --max_steps_per_epoch "$max_steps_per_epoch" \
        --num_workers "$num_workers" \
        --epochs "$epochs" \
        --lr "$lr" \
        --warmup_steps "$warmup_steps" \
        --ff_dropout "$d" \
        --attn_dropout "$d" \
        --run_name "$run_name" \
        --out_dir "$out_dir" \
        2>&1 | tee "$out_dir/run.log"

      expt=$((expt + 1))
    done
  done
done

log "Sweep complete. Logs are under results/"
