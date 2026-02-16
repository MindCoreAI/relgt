"""TabICL backbone wrapper.

Option 1 (pragmatic): use TabICLClassifier predicted class probabilities as a
fixed-size feature vector, then apply a learned projection to RelGT channels.

Caveats:
- TabICL is a supervised in-context learner and requires `fit(X, y)`.
- This wrapper uses *synthetic* labels purely to satisfy the API.
- Running fit/inference inside the training loop is expected to be slow; this
  is mainly to validate integration plumbing and measure cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import torch_frame


def _tensorframe_to_numpy(tf: "torch_frame.data.TensorFrame") -> np.ndarray:
    """Convert a TorchFrame TensorFrame into a dense numpy matrix."""
    parts = []
    feat_dict = tf.feat_dict

    if torch_frame.numerical in feat_dict:
        parts.append(feat_dict[torch_frame.numerical].float())

    if torch_frame.categorical in feat_dict:
        parts.append(feat_dict[torch_frame.categorical].float())

    if torch_frame.multicategorical in feat_dict:
        x = feat_dict[torch_frame.multicategorical]
        parts.append(x.reshape(x.size(0), -1).float())

    if len(parts) == 0:
        n = getattr(tf, "_num_rows", None) or next(iter(feat_dict.values())).size(0)
        device = next(iter(feat_dict.values())).device
        parts = [torch.zeros((n, 1), dtype=torch.float32, device=device)]

    X = torch.cat(parts, dim=1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.detach().cpu().numpy()


@dataclass
class TabICLBackboneConfig:
    # Smoke-test settings: aggressively cap data used for fit.
    max_fit_rows: int = 50
    max_fit_cols: int = 32
    random_state: int = 0


class TabICLBackbone(nn.Module):
    """TorchFrame-model-like wrapper that returns row embeddings via TabICL."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: Dict[str, Dict[Any, Any]],
        col_names_dict: Dict[torch_frame.stype, list[str]],
        stype_encoder_dict: Optional[Dict[torch_frame.stype, nn.Module]] = None,
        config: Optional[TabICLBackboneConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.stype_encoder_dict = stype_encoder_dict
        self.config = config or TabICLBackboneConfig()

        self._clf = None
        self._proj: Optional[nn.Linear] = None
        self._col_idx = None

    def reset_parameters(self) -> None:
        if self._proj is not None:
            self._proj.reset_parameters()

    def _ensure_fitted(self, X: np.ndarray) -> None:
        if self._clf is not None:
            return

        try:
            from tabicl import TabICLClassifier
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "tabicl is not installed. Install it to use --tf_model tabicl."
            ) from e

        n, d = X.shape
        rng = np.random.default_rng(self.config.random_state)

        n_fit = min(n, self.config.max_fit_rows)
        ridx = rng.choice(n, size=n_fit, replace=False) if n_fit < n else np.arange(n)
        X_fit = X[ridx]

        if d > self.config.max_fit_cols:
            cidx = rng.choice(d, size=self.config.max_fit_cols, replace=False)
            cidx.sort()
            X_fit = X_fit[:, cidx]
            self._col_idx = cidx
        else:
            self._col_idx = None

        # Avoid all-constant feature failures.
        if X_fit.shape[0] < 2:
            X_fit = np.repeat(X_fit, repeats=2, axis=0)
        if np.all(X_fit.std(axis=0) == 0):
            X_fit = X_fit + rng.normal(0.0, 1e-3, size=X_fit.shape)

        # Synthetic binary labels.
        y_fit = rng.integers(0, 2, size=X_fit.shape[0], dtype=np.int64)

        clf = TabICLClassifier(
            n_estimators=1,
            batch_size=4,
            device="cpu",
            allow_auto_download=True,
            random_state=self.config.random_state,
            verbose=False,
        )
        clf.fit(X_fit, y_fit)
        self._clf = clf

    def forward(self, tf: "torch_frame.data.TensorFrame") -> torch.Tensor:
        X = _tensorframe_to_numpy(tf)
        self._ensure_fitted(X)

        X_in = X
        if self._col_idx is not None:
            X_in = X[:, self._col_idx]

        proba = self._clf.predict_proba(X_in)
        proba_t = torch.from_numpy(proba).to(dtype=torch.float32)

        if self._proj is None:
            self._proj = nn.Linear(proba_t.size(1), self.out_channels)
            self._proj.to(proba_t.device)

        return self._proj(proba_t)
