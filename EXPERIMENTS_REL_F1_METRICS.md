# RelGT – rel-f1 metrics (collected runs)

This file tracks *observed* timings + metrics from our Mac/WSL experiments.

## Environment notes
- Mac mini: CPU runs (MPS disabled)
- WSL Ubuntu (Windows GPU): NVIDIA GeForce GTX 1080 Ti, torch 2.4.1+cu121
- W&B disabled (WANDB_MODE=disabled)

---

## Timing sanity check (driver-dnf)
Settings: `epochs=1`, `batch_size=64`, `max_steps_per_epoch=50`, `num_workers=0`, `dropout=0.3`.

### Mac (CPU)
- real: **351.86s**
- Best Val: AP **0.9261514139**, acc **0.7915194346**, f1 **0.8763102725**, roc_auc **0.7780680272**
- Best Test: AP **0.8461403810**, acc **0.6866096866**, f1 **0.8126064736**, roc_auc **0.6987361538**

### WSL (CUDA, GTX 1080 Ti)
- real: **226.41s**
- Best Val: AP **0.8926309240**, acc **0.5600706714**, f1 **0.6278026906**, roc_auc **0.7248616780**
- Best Test: AP **0.8115777083**, acc **0.4957264957**, f1 **0.4747774481**, roc_auc **0.6175376958**

---

## Large-base-style rel-f1 sweep (WSL CUDA)
Goal: match `expts/run-large-base-experiments.sh` settings as closely as possible, but only rel-f1 tasks.

Shared settings (intended):
- `dataset=rel-f1`
- tasks: `driver-dnf`, `driver-top3`, `driver-position`
- `--precompute`, `--seed 0`, `--num_neighbors 300`
- `--num_layers 4`, `--channels 512`, `--gt_conv_type full`, `--ablate none`
- `--epochs 10`, `--max_steps_per_epoch 500`, `--lr 1e-4`, `--warmup_steps 10`
- `--ff_dropout 0.3`, `--attn_dropout 0.3`, `--num_workers 8`

### driver-top3
- BS=1024: OOM
- BS=512: OOM
- **BS=256: completed**
  - Best Val: AP **0.4418107950**, acc **0.7380952381**, f1 **0.4689655172**, roc_auc **0.7490458870**
  - Best Test: AP **0.3991067687**, acc **0.7451790634**, f1 **0.4699140401**, roc_auc **0.7830528846**
  - real: **1717.95s**

### driver-dnf
- BS=1024: OOM
- BS=512: *failed* with CUDA OOM during pin_memory / dataloader (real **889.10s**)
- BS=256: ran but only timing captured (real **587.83s**); metrics not recorded here
- **BS=128: completed**
  - Best Val: AP **0.9209112365**, acc **0.5459363958**, f1 **0.6100151745**, roc_auc **0.7668752834**
  - Best Test: AP **0.8306531574**, acc **0.6339031339**, f1 **0.6862026862**, roc_auc **0.6876104036**
  - real: **1755.27s**

### driver-position
- BS=1024: OOM
- **BS=512: completed**
  - Best Val: r2 **0.2922098644**, mae **3.1276646954**, rmse **3.9003406779**
  - Best Test: r2 **0.1797089060**, mae **3.9104570101**, rmse **4.7190421238**
  - real: **1235.16s**

