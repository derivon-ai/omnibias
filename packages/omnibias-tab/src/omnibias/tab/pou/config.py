# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Configuration for TabPOU (theory 05-04).

Terminology: the split gate ``sigmoid(beta (x[f] - t))`` hardens as ``beta -> inf``
-- temperature collapse (feasibility sense), distinct from the founding bias
collapse (``delta -> 0`` to ``sigma^(K-1)``).
"""

from __future__ import annotations

from dataclasses import dataclass

_TASKS = ("binary", "multiclass", "regression")
_SPLIT_KINDS = ("axis", "oblique", "sparse")
_ROLES = ("band", "integral")


@dataclass(frozen=True)
class TabPOUConfig:
    r"""Knobs for :func:`~omnibias.tab.pou.train.fit_tabpou`.

    Defaults match the flagship ``TabConfig`` budget (60 stages, shrinkage 0.3)
    with axis splits and a band+raw numerical front-end. Residual TabM is off
    until G3 is measured; turn ``use_residual=True`` for G4.
    """

    n_features: int
    n_outputs: int = 1
    task: str = "regression"
    depth: int = 3
    n_stages: int = 60
    trees_per_stage: int = 1
    learning_rate: float = 0.3
    beta_final: float = 8.0
    leaf_l2: float = 1e-4
    n_quantiles: int = 16
    colsample: float = 1.0
    subsample: float = 1.0
    n_bins: int = 16
    use_embed: bool = True
    concat_raw: bool = True
    embed_role: str = "band"
    use_residual: bool = False
    residual_k: int = 8
    residual_hidden: int = 64
    residual_steps: int = 40
    residual_lr: float = 0.05
    split_kind: str = "axis"
    seed: int = 0
    robust_scale: bool = True
    clip: float = 5.0
    onehot_max_card: int = 16
    patience: int | None = 12

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {self.n_outputs}")
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {self.task!r}")
        if self.task == "binary" and self.n_outputs != 1:
            raise ValueError("binary task requires n_outputs == 1")
        if self.task == "multiclass" and self.n_outputs < 2:
            raise ValueError("multiclass task requires n_outputs >= 2")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.n_stages < 1:
            raise ValueError(f"n_stages must be >= 1, got {self.n_stages}")
        if self.trees_per_stage < 1:
            raise ValueError("trees_per_stage must be >= 1")
        if self.split_kind not in _SPLIT_KINDS:
            raise ValueError(f"split_kind must be one of {_SPLIT_KINDS}, got {self.split_kind!r}")
        if self.embed_role not in _ROLES:
            raise ValueError(f"embed_role must be one of {_ROLES}, got {self.embed_role!r}")
        if not (0.0 < self.colsample <= 1.0):
            raise ValueError("colsample must be in (0, 1]")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if self.residual_k < 1:
            raise ValueError("residual_k must be >= 1")
        if self.beta_final <= 0.0:
            raise ValueError("beta_final must be positive")
        if self.leaf_l2 < 0.0:
            raise ValueError("leaf_l2 must be >= 0")


__all__ = ["TabPOUConfig"]
