"""TabPFN backbone wrapper.

This is a pragmatic integration intended for experimentation/smoke-testing:
- Fits a TabPFNClassifier once (per node type encoder instance) on a subset of rows
  with *synthetic* labels, then uses predicted class probabilities as a fixed
  embedding, followed by a learned projection.

Caveats:
- This is NOT a principled label-free tabular representation learning method.
- TabPFN is primarily designed for supervised prediction; this wrapper uses it
  as a feature extractor to test plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import torch_frame


def _tensorframe_to_numpy(tf: "torch_frame.data.TensorFrame") -> np.ndarray:
    """Convert a TorchFrame TensorFrame into a dense numpy matrix.

    We concatenate available stypes in a simple way:
    - numerical: float features
    - categorical: integer codes cast to float
    - multicategorical: flattened integer codes cast to float

    This is intentionally simple; better encodings are possible.
    """

    parts = []

    feat_dict = getattr(tf, "feat_dict")

    if torch_frame.numerical in feat_dict:
        x = feat_dict[torch_frame.numerical]
        parts.append(x.float())

    if torch_frame.categorical in feat_dict:
        x = feat_dict[torch_frame.categorical]
        parts.append(x.float())

    if torch_frame.multicategorical in feat_dict:
        x = feat_dict[torch_frame.multicategorical]
        # [N, C, K] or [N, K]? flatten all but batch.
        parts.append(x.reshape(x.size(0), -1).float())

    if len(parts) == 0:
        # Fall back to a single zero feature to keep shapes valid.
        n = getattr(tf, "_num_rows", None) or next(iter(feat_dict.values())).size(0)
        parts = [torch.zeros((n, 1), dtype=torch.float32, device=next(iter(feat_dict.values())).device)]

    X = torch.cat(parts, dim=1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.detach().cpu().numpy()


@dataclass
class TabPFNBackboneConfig:
    max_fit_rows: int = 1000
    random_state: int = 0


class TabPFNBackbone(nn.Module):
    """A TorchFrame-like model that returns row embeddings via TabPFN.

    Signature matches torch_frame model constructors used in NeighborTfsEncoder.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: Dict[str, Dict[Any, Any]],
        col_names_dict: Dict[torch_frame.stype, list[str]],
        stype_encoder_dict: Optional[Dict[torch_frame.stype, nn.Module]] = None,
        config: Optional[TabPFNBackboneConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.stype_encoder_dict = stype_encoder_dict
        self.config = config or TabPFNBackboneConfig()

        self._clf = None
        self._proj: Optional[nn.Linear] = None

    def reset_parameters(self) -> None:
        if self._proj is not None:
            self._proj.reset_parameters()
        # Keep fitted TabPFN model; reset_parameters is used for torch modules.

    def _ensure_fitted(self, X: np.ndarray) -> None:
        if self._clf is not None:
            return

        try:
            from tabpfn import TabPFNClassifier
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TabPFN is not installed. Install it in your environment to use --tf_model tabpfn."
            ) from e

        n = X.shape[0]
        n_fit = min(n, self.config.max_fit_rows)
        rng = np.random.default_rng(self.config.random_state)
        idx = rng.choice(n, size=n_fit, replace=False) if n_fit < n else np.arange(n)
        X_fit = X[idx]
        # Synthetic binary labels (balanced) purely to enable fit().
        y_fit = rng.integers(0, 2, size=n_fit, dtype=np.int64)

        clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, random_state=self.config.random_state)
        clf.fit(X_fit, y_fit)
        self._clf = clf

    def forward(self, tf: "torch_frame.data.TensorFrame") -> torch.Tensor:
        X = _tensorframe_to_numpy(tf)
        self._ensure_fitted(X)

        proba = self._clf.predict_proba(X)  # [N, C]
        proba_t = torch.from_numpy(proba).to(dtype=torch.float32)

        if self._proj is None:
            self._proj = nn.Linear(proba_t.size(1), self.out_channels)
            self._proj.to(proba_t.device)

        out = self._proj(proba_t)
        return out
