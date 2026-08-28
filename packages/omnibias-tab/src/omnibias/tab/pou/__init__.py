# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""TabPOU: axis-aligned POU boosting with closed-form Newton leaves (theory 05-04 / 05-05).

A from-scratch tabular model: axis-aligned oblivious trees, exact ridge leaves,
optional band/integral numerical embeddings, optional TabM residual. The gate
``sigmoid(beta (x[f] - t))`` hardens as ``beta -> inf`` -- temperature collapse
(feasibility sense), not the founding ``delta -> 0`` bias collapse.

Theory 05-05 adds ``fit_tabpou_joint`` (threshold polish, trained embed, POU
residual) without changing v0 ``fit_tabpou`` defaults.
"""

from __future__ import annotations

from omnibias.tab.pou.config import TabPOUConfig
from omnibias.tab.pou.grow import fit_axis_tree, token_feature_groups
from omnibias.tab.pou.preprocess import TabPreprocessor

__all__ = [
    "TabPOUConfig",
    "TabPreprocessor",
    "fit_axis_tree",
    "token_feature_groups",
]

try:
    from omnibias.tab.pou.ensemble import LinearBatchEnsemble, TabMResidual
    from omnibias.tab.pou.joint import (
        TabPOUJointConfig,
        TabPOUJointResult,
        fit_tabpou_joint,
        polish_thresholds,
    )
    from omnibias.tab.pou.model import TabPOU
    from omnibias.tab.pou.train import TabPOUResult, fit_tabpou

    __all__ = [
        "LinearBatchEnsemble",
        "TabMResidual",
        "TabPOU",
        "TabPOUConfig",
        "TabPOUJointConfig",
        "TabPOUJointResult",
        "TabPOUResult",
        "TabPreprocessor",
        "fit_axis_tree",
        "fit_tabpou",
        "fit_tabpou_joint",
        "polish_thresholds",
        "token_feature_groups",
    ]
except ImportError:  # pragma: no cover - torch extra missing
    pass
