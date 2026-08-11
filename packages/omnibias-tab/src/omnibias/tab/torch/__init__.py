# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""PyTorch backend for omnibias-tab: the trainable soft-tree module + trainers."""

from __future__ import annotations

from omnibias.tab.torch.arrangement import ArrangementClassifier, FitResult, fit_arrangement
from omnibias.tab.torch.boosting import BoostResult, fit_boosted
from omnibias.tab.torch.model import SoftTreeEnsemble
from omnibias.tab.torch.train import TrainResult, fit_first_order, fit_second_order

__all__ = [
    "ArrangementClassifier",
    "BoostResult",
    "FitResult",
    "SoftTreeEnsemble",
    "TrainResult",
    "fit_arrangement",
    "fit_boosted",
    "fit_first_order",
    "fit_second_order",
]
