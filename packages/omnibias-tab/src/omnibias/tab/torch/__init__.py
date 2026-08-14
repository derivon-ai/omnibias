# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""PyTorch backend for omnibias-tab: the trainable soft-tree module + trainers."""

from __future__ import annotations

from omnibias.tab.torch.arrangement import (
    ArrangementBoosted,
    ArrangementClassifier,
    BoostedFitResult,
    FitResult,
    fit_arrangement,
    fit_arrangement_boosted,
)
from omnibias.tab.torch.boosting import BoostResult, fit_boosted
from omnibias.tab.torch.jet import (
    TreeJet,
    extract_arrangement_jet,
    extract_tree_jet,
    extract_tree_jet_directional,
    sequential_mlp_jet,
)
from omnibias.tab.torch.model import SoftTreeEnsemble
from omnibias.tab.torch.plugin import TabHead, as_head
from omnibias.tab.torch.train import TrainResult, fit_first_order, fit_joint, fit_second_order

__all__ = [
    "ArrangementBoosted",
    "ArrangementClassifier",
    "BoostedFitResult",
    "BoostResult",
    "FitResult",
    "SoftTreeEnsemble",
    "TabHead",
    "TrainResult",
    "TreeJet",
    "as_head",
    "extract_arrangement_jet",
    "extract_tree_jet",
    "extract_tree_jet_directional",
    "fit_arrangement",
    "fit_arrangement_boosted",
    "fit_boosted",
    "fit_first_order",
    "fit_joint",
    "fit_second_order",
    "sequential_mlp_jet",
]
