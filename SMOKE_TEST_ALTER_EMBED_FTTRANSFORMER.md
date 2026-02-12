# alter-embed-fttransformer: rel-f1 smoke test (TorchFrame FTTransformer)

This branch adds an option to swap the TorchFrame tabular encoder backbone from **ResNet (default)** to **FTTransformer** via `--tf_model`.

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
  --tf_model fttransformer \
  --run_name smoke_fttransformer \
  --out_dir results/smoke_fttransformer \
  --ff_dropout 0.3 \
  --attn_dropout 0.3
```

## Results
Run output summary:
- Runtime: **real 133.99s** (user 815.96s, sys 453.86s)
- Epoch 1 Val metrics (printed after training loop):
  - average_precision: **0.8390**
  - accuracy: **0.7792**
  - f1: **0.8759**
  - roc_auc: **0.6160**
- Best Val metrics (after selecting best checkpoint):
  - average_precision: **0.8256**
  - accuracy: **0.7792**
  - f1: **0.8759**
  - roc_auc: **0.5837**
- Best Test metrics:
  - average_precision: **0.6367**
  - accuracy: **0.7051**
  - f1: **0.8271**
  - roc_auc: **0.4212**

## Where artifacts go
- Output dir: `results/smoke_fttransformer/rel-f1/driver-dnf/`
- Best checkpoint (when saved): `finetuned.pt`
