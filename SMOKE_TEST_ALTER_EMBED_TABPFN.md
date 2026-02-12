# alter-embed-tabpfn: rel-f1 smoke test (TabPFN embedding backbone)

This branch adds an option to use **TabPFN** as a (very experimental) table-feature backbone.

Important caveat up front: TabPFN is fundamentally a *supervised* tabular predictor (requires `fit(X, y)`), and using it as a label-free embedding extractor inside RelGT is not a clean match. This branch is primarily to validate plumbing and quantify performance/cost.

## What was implemented
- New CLI flag in `main_node_ddp.py`:
  - `--tf_model {resnet,tabpfn}`
- New file: `tabpfn_backbone.py`
  - Implements `TabPFNBackbone`, a TorchFrame-model-like wrapper.
  - Converts a TorchFrame `TensorFrame` to a dense numpy matrix.
  - Fits a `TabPFNClassifier` (using **synthetic labels**) once per node-type encoder instance.
  - Uses `predict_proba(X)` as a small feature vector and applies a learned linear projection to `out_channels`.

## Environment
- Machine: Mac mini (CPU)
- Python env: micromamba env `relgt`
- TabPFN: installed in the env (`tabpfn`)

## Attempts + outcomes

### Attempt 1 (10 steps/epoch)
Command:

```bash
cd ~/projects/relgt

WANDB_MODE=disabled \
~/micromamba/bin/micromamba run -n relgt \
python main_node_ddp.py \
  --dataset rel-f1 \
  --task driver-dnf \
  --epochs 1 \
  --batch_size 16 \
  --max_steps_per_epoch 10 \
  --num_workers 0 \
  --tf_model tabpfn \
  --run_name smoke_tabpfn \
  --out_dir results/smoke_tabpfn \
  --ff_dropout 0.3 \
  --attn_dropout 0.3
```

Failure:
- TabPFN raised:
  - `tabpfn.errors.TabPFNValidationError: All features are constant and would have been removed! Unable to predict using TabPFN.`

Fix applied:
- In `TabPFNBackbone._ensure_fitted()`: if all columns are constant, add tiny Gaussian noise to `X_fit`.

### Attempt 2 (10 steps/epoch, after constant-feature fix)
Result:
- No longer hit the constant-feature validation error.
- But training was **extremely slow** (minutes per step) due to TabPFN fitting/preprocessing overhead inside the forward path.
- The run was eventually killed (SIGKILL), likely due to runtime/resource limits.

### Attempt 3 (1 step/epoch, reduced TabPFN fit cost)
Additional changes:
- `TabPFNClassifier(... n_estimators=1, fit_mode="low_memory", n_preprocessing_jobs=1)`
- Reduced `max_fit_rows` to 200 (smoke-test only)

Command:

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
  --tf_model tabpfn \
  --run_name smoke_tabpfn_steps1 \
  --out_dir results/smoke_tabpfn \
  --ff_dropout 0.3 \
  --attn_dropout 0.3
```

Outcome:
- Training still very slow (e.g., ~15s for the single train step; validation iterations ~19s/it observed early on).
- Not practical to run full end-to-end eval in this configuration.

## Conclusion
- TabPFN-as-backbone **works at the code/plumbing level**, but is too slow when used in the inner training loop.
- To make this viable, we likely need a **precompute/caching** approach:
  - Precompute TabPFN-derived row embeddings per table/node type once,
  - Cache to disk,
  - During RelGT training, only index cached embeddings.
