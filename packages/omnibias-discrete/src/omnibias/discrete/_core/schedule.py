# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-neutral temperature schedule for differentiable annealing.

The feasibility limit beta -> inf hardens soft assignments. Founding bias
collapse is the distinct delta -> 0 limit that extracts activation derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnealSchedule:
    r"""Homotopy schedule for the differentiable annealed (temperature-collapse) relaxation.

    The relaxation parametrises ``x = sigmoid(beta theta) in (0, 1)^n`` and descends the
    closed-form energy gradient by unrolled gradient descent along a geometric ``beta``
    homotopy; as ``beta`` grows the soft assignment collapses onto a binary vertex.
    Defaults are eval-quality; :meth:`fast` is enough to train *through* the relaxation.

    Attributes
    ----------
    beta0, beta_growth:
        Initial inverse-temperature and geometric growth factor per stage.
    stages:
        Number of homotopy stages (each warm-starts the next).
    steps:
        Gradient-descent steps per stage.
    step_safety:
        Fraction of the (closed-form Lipschitz) step to take (``0 < step_safety <= 1``).
    """

    beta0: float = 0.5
    beta_growth: float = 1.5
    stages: int = 12
    steps: int = 60
    step_safety: float = 0.9

    def __post_init__(self) -> None:
        if self.beta0 <= 0.0:
            raise ValueError("beta0 must be > 0")
        if self.beta_growth < 1.0:
            raise ValueError("beta_growth must be >= 1")
        if self.stages < 1 or self.steps < 1:
            raise ValueError("stages and steps must be >= 1")
        if not 0.0 < self.step_safety <= 1.0:
            raise ValueError("step_safety must be in (0, 1]")

    def betas(self) -> list[float]:
        """The inverse-temperature ``beta`` at each homotopy stage."""
        out, beta = [], self.beta0
        for _ in range(self.stages):
            out.append(beta)
            beta *= self.beta_growth
        return out

    @classmethod
    def fast(cls) -> AnnealSchedule:
        """A lighter schedule for training *through* the relaxation."""
        return cls(beta0=0.5, beta_growth=1.7, stages=8, steps=30)


__all__ = ["AnnealSchedule"]
