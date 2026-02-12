# Self-supervised tabular pretraining (DAE/MFM) — initial prototype

Branch: `alter-embed-selfsup`

This adds a first-pass **DAE / Masked Feature Modeling (MFM)** pretraining script over RelBench/TorchFrame tables.

## What it does
- Builds the RelBench hetero graph features (same pipeline as training) to obtain TorchFrame `TensorFrame`s per node type.
- For each node type/table with **numerical features**, trains a TorchFrame backbone to reconstruct masked numerical values.
- Saves one checkpoint per table (node type).

**Current scope (intentionally minimal):**
- Numerical features only (categorical reconstruction not yet implemented).
- Loss: MSE on masked positions.

## Script
- `expts/pretrain_mfm_relbench.py`

## Example command
From repo root:

```bash
cd ~/projects/relgt

~/micromamba/bin/micromamba run -n relgt \
python expts/pretrain_mfm_relbench.py \
  --dataset rel-f1 \
  --device cpu \
  --backbone mlp \
  --steps 200 \
  --batch_size 512 \
  --mask_prob 0.3 \
  --lr 1e-3 \
  --out_dir results/selfsup_mfm
```

## Smoke run
A short smoke run completed end-to-end:

```bash
~/micromamba/bin/micromamba run -n relgt \
python expts/pretrain_mfm_relbench.py \
  --dataset rel-f1 \
  --steps 2 \
  --batch_size 128 \
  --mask_prob 0.3 \
  --backbone mlp \
  --out_dir results/selfsup_mfm_smoke
```

Outputs were written to:
- `results/selfsup_mfm_smoke/rel-f1/mfm_mlp/*.pt`

## Known issues / next steps
- Some tables have no numerical features → currently skipped.
- Loss scale can be huge if numerical features are unnormalized (expected; next step is normalization or robust loss).
- Next: add categorical reconstruction and/or a SCARF-style contrastive objective.
- Next: add a clean way to **load pretrained backbone weights** into RelGT training.
