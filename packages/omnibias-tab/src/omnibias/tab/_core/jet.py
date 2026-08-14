# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic tree-jet container (FieldJet layout, no symbolic import)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TreeJet:
    r"""Samples of a scalar soft-tree output and mixed partials up to ``order``.

    Layout matches :class:`omnibias.symbolic.field_discovery.FieldJet` so a
    caller can wrap it without ``tab`` importing ``symbolic``.
    """

    X: np.ndarray
    order: int
    partials: dict[tuple[int, ...], np.ndarray]
    var_names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def value(self) -> np.ndarray:
        return self.partials[(0,) * self.dim]

    def partial(self, alpha: tuple[int, ...]) -> np.ndarray:
        return self.partials[tuple(alpha)]


__all__ = ["TreeJet"]
