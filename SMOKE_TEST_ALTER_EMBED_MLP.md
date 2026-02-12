# alter-embed-mlp: rel-f1 smoke test (TorchFrame MLP)

This branch adds an option to swap the TorchFrame tabular encoder backbone from **ResNet (default)** to **MLP** via `--tf_model`.

## Environment
- Machine: Mac mini (CPU)
- Python env: micromamba env `relgt`
- Notes:
  - Run with `WANDB_MODE=disabled` to avoid W&B init/key prompts.
  - This smoke test is intended to verify **end-to-end execution** (data load → train → eval → save/load best).

## Command
From the repo root:

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
  --tf_model mlp \
  --run_name smoke_mlp \
  --out_dir results/smoke_mlp \
  --ff_dropout 0.3 \
  --attn_dropout 0.3
```

## Results
Run output summary:
- Runtime: **real 74.55s** (user 450.28s, sys 235.56s)
- Best Val metrics:
  - average_precision: **0.8146**
  - accuracy: **0.2261**
  - f1: **0.0135**
  - roc_auc: **0.5572**
- Best Test metrics:
  - average_precision: **0.8171**
  - accuracy: **0.4487**
  - f1: **0.4000**
  - roc_auc: **0.6406**

## Where artifacts go
- Output dir: `results/smoke_mlp/rel-f1/driver-dnf/`
- Best checkpoint (when saved): `finetuned.pt`
