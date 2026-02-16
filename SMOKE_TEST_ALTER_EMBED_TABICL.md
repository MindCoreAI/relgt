# alter-embed-tabicl: rel-f1 smoke test (TabICL v2 proba embedding)

This branch adds an option to use **TabICL v2** (via the `tabicl` Python package) as an experimental tabular backbone.

## Integration (Option 1)
- We use `TabICLClassifier.fit(X, y)` once per node-type encoder instance, with **synthetic binary labels**.
- For inference, we compute `predict_proba(X)` and treat the probability vector as a small feature embedding.
- A learned linear projection maps that vector to `out_channels` (RelGT channels).

**Caveat:** TabICL is a supervised in-context learner; using it inside RelGT’s forward pass is expected to be slow.

## Code
- New file: `tabicl_backbone.py`
- `model.py`: selects `TabICLBackbone` when `--tf_model tabicl`
- `main_node_ddp.py`: adds `--tf_model {resnet,tabicl}` and allows CPU/local (non-torchrun) smoke tests.

## Cache location
TabICL downloads weights from Hugging Face Hub. On this machine, it cached under:
- `~/.cache/huggingface/hub/models--jingang--TabICL/`

## Smoke command
```bash
cd ~/projects/relgt

WANDB_MODE=disabled \
~/micromamba/bin/micromamba run -n relgt \
python main_node_ddp.py \
  --dataset rel-f1 \
  --task driver-dnf \
  --epochs 1 \
  --batch_size 16 \
  --max_steps_per_epoch 1 \
  --num_workers 0 \
  --tf_model tabicl \
  --run_name smoke_tabicl \
  --out_dir results/smoke_tabicl \
  --ff_dropout 0.3 \
  --attn_dropout 0.3
```

## Observed runtime + result
- Train (1 step): ~2 seconds before moving to validation.
- Validation: ~3.0s/it (36 iters ≈ ~1m46s). Completed.
- Printed after epoch 1:
  - Train loss: **0.6489**
  - Val metrics: AP **0.8576**, acc **0.7792**, f1 **0.8759**, roc_auc **0.6343**

After the epoch loop, the script runs an additional final eval pass (`Val` again, then `Test`).
- Second Val pass started (~3.0s/it).
- Test pass started (~3.0s/it) and reached **26/44**.

The run was terminated by a **5-minute cap** (SIGKILL) before completion, so we did not capture final “Best Test metrics” in that timed run.

### Estimate to finish
Roughly:
- 1st Val: ~106s
- 2nd Val: ~106s
- Test: ~132s
Total eval ≈ ~344s plus overhead → **~6 minutes**.
So with a longer time budget, the run should complete.
