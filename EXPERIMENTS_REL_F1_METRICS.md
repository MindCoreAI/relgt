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

---

## Strategy 1 – Gradient accumulation (NO AMP) on WSL CUDA
Goal: keep effective batch size similar to large-base settings while avoiding CUDA OOM.

Settings: same as large-base-style rel-f1 (layers=4, channels=512, dropout=0.3, epochs=10, max_steps=500, neighbors=300, seed=0, precompute). GPU: GTX 1080 Ti.

### Eff batch size ≈ 512

#### driver-dnf (accum-only)
- micro-batch size: **128**, grad_accum_steps: **4** (effective ≈ 512)
- Best Val: AP **0.9267476479**, acc **0.7915194346**, f1 **0.8691796009**, roc_auc **0.7787210884**
- Best Test: AP **0.8019643342**, acc **0.6866096866**, f1 **0.8119658120**, roc_auc **0.6050651442**
- real: **1697.27s**

#### driver-top3 (accum-only)
- micro-batch size: **256**, grad_accum_steps: **2** (effective ≈ 512)
- Best Val: AP **0.4507588510**, acc **0.7551020408**, f1 **0.4666666667**, roc_auc **0.7291662934**
- Best Test: AP **0.3342315649**, acc **0.6294765840**, f1 **0.2864721485**, roc_auc **0.6890546614**
- real: **1680.54s**

### Eff batch size = 1024

#### driver-dnf (accum-only)
- micro-batch size: **128**, grad_accum_steps: **8** (effective = 1024)
- Best Val: AP **0.9174346028**, acc **0.6307420495**, f1 **0.7133058985**, roc_auc **0.7553741497**
- Best Test: AP **0.7619947303**, acc **0.4629629630**, f1 **0.4414814815**, roc_auc **0.5044063827**
- real: **1704.69s**

#### driver-top3 (accum-only)
- micro-batch size: **256**, grad_accum_steps: **4** (effective = 1024)
- Best Val: AP **0.4063318048**, acc **0.4829931973**, f1 **0.4307116105**, roc_auc **0.7426851338**
- Best Test: AP **0.2301230747**, acc **0.5316804408**, f1 **0.3884892086**, roc_auc **0.6404029055**
- real: **1672.14s**

### AMP status
Attempting `--amp` currently triggers NaNs during backward on this stack (e.g. AddmmBackward0/BCEWithLogitsBackward0 returned NaNs). Proceeding without AMP for now.

