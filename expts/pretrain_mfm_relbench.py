"""Masked Feature Modeling (MFM) / Denoising Autoencoder (DAE) pretraining for RelBench TorchFrame tables.

This is a pragmatic first step toward self-supervised tabular pretraining.
It trains a TorchFrame backbone (ResNet/MLP/FTTransformer) to reconstruct masked
numerical features.

Notes:
- We start with numerical reconstruction only (fast, simple). Categorical
  reconstruction can be added later.
- We pretrain per node type (table) and write checkpoints to an output dir.

Example:
  ~/micromamba/bin/micromamba run -n relgt \
    python expts/pretrain_mfm_relbench.py \
      --dataset rel-f1 --steps 200 --batch_size 512 --mask_prob 0.3 \
      --backbone mlp --out_dir results/selfsup_mfm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Allow importing project-local modules when running from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_frame
from relbench.datasets import get_dataset


def get_numeric_matrix(tf: "torch_frame.data.TensorFrame") -> torch.Tensor:
    feat_dict = tf.feat_dict
    if torch_frame.numerical not in feat_dict:
        raise ValueError("TensorFrame has no numerical features; MFM numerical pretrain cannot run.")
    x = feat_dict[torch_frame.numerical]
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def mask_numeric(x: torch.Tensor, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (x_corrupt, mask) where mask=True indicates positions to reconstruct."""
    mask = torch.rand_like(x).lt(mask_prob)
    x_corrupt = x.clone()
    x_corrupt[mask] = 0.0
    return x_corrupt, mask


class NumericReconHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def make_backbone(kind: str, out_channels: int, col_stats, col_names_dict) -> nn.Module:
    if kind == "resnet":
        cls = torch_frame.nn.models.ResNet
        kwargs = dict(channels=128, num_layers=4)
    elif kind == "mlp":
        cls = torch_frame.nn.models.MLP
        kwargs = dict(channels=128, num_layers=4, dropout_prob=0.2)
    elif kind == "fttransformer":
        cls = torch_frame.nn.models.FTTransformer
        kwargs = dict(channels=128, num_layers=4)
    else:
        raise ValueError(f"Unknown backbone: {kind}")

    # Use default stype encoders for a standard setup.
    stype_encoder_dict = {
        torch_frame.categorical: torch_frame.nn.EmbeddingEncoder(),
        torch_frame.numerical: torch_frame.nn.LinearEncoder(),
        torch_frame.multicategorical: torch_frame.nn.MultiCategoricalEmbeddingEncoder(),
        torch_frame.embedding: torch_frame.nn.LinearEmbeddingEncoder(),
        torch_frame.timestamp: torch_frame.nn.TimestampEncoder(),
    }

    model = cls(
        **kwargs,
        out_channels=out_channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
        stype_encoder_dict={k: v for k, v in stype_encoder_dict.items() if k in col_names_dict},
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="rel-f1")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--backbone", type=str, default="mlp", choices=["resnet", "mlp", "fttransformer"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mask_prob", type=float, default=0.3)
    p.add_argument("--out_dir", type=str, default="results/selfsup_mfm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ds = get_dataset(args.dataset, download=True)
    db = ds.get_db()

    out_root = Path(args.out_dir) / args.dataset / f"mfm_{args.backbone}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Build the heterogeneous graph + TorchFrame features once (same as training).
    from relbench.modeling.utils import get_stype_proposal
    from relbench.modeling.graph import make_pkey_fkey_graph
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from utils import GloveTextEmbedding

    col_to_stype_dict = get_stype_proposal(db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=256,
        ),
        cache_dir=str(out_root / "materialized"),
    )

    # Iterate node types (tables) present in the graph.
    for table_name in data.node_types:
        tf = data[table_name].tf

        if torch_frame.numerical not in tf.feat_dict:
            print(f"[skip] {table_name}: no numerical features")
            continue

        x_num = get_numeric_matrix(tf)
        n, d = x_num.shape
        if n < 2:
            print(f"[skip] {table_name}: too few rows ({n})")
            continue

        # For the first MFM version, pretrain on numerical features only.
        num_col_names = tf.col_names_dict[torch_frame.numerical]
        col_stats_num = {k: v for k, v in col_stats_dict[table_name].items() if k in num_col_names}
        col_names_dict_num = {torch_frame.numerical: num_col_names}

        # Create a TorchFrame model that produces row embeddings.
        backbone = make_backbone(
            kind=args.backbone,
            out_channels=256,
            col_stats=col_stats_num,
            col_names_dict=col_names_dict_num,
        ).to(device)

        # Decoder to reconstruct numerical vector.
        head = NumericReconHead(in_dim=256, out_dim=d).to(device)

        optim = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=args.lr)

        # Training loop: sample random rows.
        backbone.train()
        head.train()

        print(f"[pretrain] table={table_name} rows={n} num_dim={d} steps={args.steps}")
        for step in range(1, args.steps + 1):
            idx = torch.randint(0, n, (min(args.batch_size, n),))
            x = x_num[idx].to(device)

            x_corrupt, mask = mask_numeric(x, args.mask_prob)

            # Replace the numerical features in a copy of TensorFrame.
            tf_batch = torch_frame.data.TensorFrame(
                feat_dict={torch_frame.numerical: x_corrupt},
                col_names_dict=col_names_dict_num,
                num_rows=x_corrupt.size(0),
            )

            z = backbone(tf_batch)  # [B, 256]
            x_hat = head(z)

            # Reconstruction loss only on masked positions.
            if mask.any():
                loss = F.mse_loss(x_hat[mask], x[mask])
            else:
                loss = F.mse_loss(x_hat, x)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step == 1 or step % 50 == 0:
                print(f"  step={step:04d} loss={loss.item():.6f}")

        # Save checkpoint per table.
        ckpt = {
            "table": table_name,
            "backbone": args.backbone,
            "backbone_state": backbone.state_dict(),
            "head_state": head.state_dict(),
            "num_dim": d,
            "mask_prob": args.mask_prob,
            "steps": args.steps,
        }
        out_path = out_root / f"{table_name}.pt"
        torch.save(ckpt, out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
