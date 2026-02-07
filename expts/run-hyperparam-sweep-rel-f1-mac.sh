#!/usr/bin/env bash
set -euo pipefail

# Minimal Mac sweep: ONLY rel-f1 (smallest dataset in RelBench core).
# Runs sequentially on CPU/MPS.

BASE_PORT=29200

dataset="rel-f1"
# A few representative tasks from rel-f1:
tasks=("driver-dnf" "driver-top3" "driver-position")

# Keep a small grid so it finishes on Mac.
dropouts=(0.3 0.5)
num_layers=(1 4)

batch_size=128
max_steps_per_epoch=300
num_workers=0
warmup_steps=50
epochs=5
lr=0.0001

mkdir -p results
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

expt=0
for d in "${dropouts[@]}"; do
  for l in "${num_layers[@]}"; do
    for task in "${tasks[@]}"; do
      dro="${d//./}" # 0.3 -> 03
      run_name="relgt-mac-f1-l${l}-512-dropout${dro}-BS${batch_size}-MAX${max_steps_per_epoch}"
      out_dir="results/${run_name}/${dataset}__${task}"
      mkdir -p "$out_dir"

      log "(expt $expt) Running $dataset ($task) dropout=$d layers=$l"

      WANDB_MODE=disabled \
      RELGT_USE_MPS=0 \
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

log "rel-f1 sweep complete. Logs are under results/"
