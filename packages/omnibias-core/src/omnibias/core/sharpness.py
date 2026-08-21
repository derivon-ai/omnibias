# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sharpness-scheduled cubic / learning-rate algebra (theory 08-06).

A few exact Hessian-vector products give a Ritz estimate ``ell_k`` of
``lambda_max`` of the loss Hessian. The cubic penalty or a gradient
step length is then ``c * max(ell_k, ell_min)`` or ``c / max(ell_k,
ell_min)``. Sharpness is a **step-size** signal, not a generalization
certificate and not CCF stretch.

``ell_k`` is a lower bound on ``lambda_max`` (Ritz). Underestimating
sharpness can still diverge; the gate is finite finishes, not an
optimal ``c``. Bias collapse (``delta -> 0``) makes HVPs exact.
No temperature collapse. Hutchinson is not the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SharpnessTarget = Literal["cubic_sigma", "lr"]


@dataclass(frozen=True)
class SharpnessSchedule:
    """Named map from a Ritz ``lambda_max`` to a cubic ``sigma`` or a learning rate."""

    n_lanczos: int = 4
    c: float = 1.0
    ell_min: float = 1e-6
    target: SharpnessTarget = "cubic_sigma"

    def __post_init__(self) -> None:
        if int(self.n_lanczos) < 1:
            raise ValueError(f"n_lanczos must be >= 1, got {self.n_lanczos}")
        if float(self.c) <= 0.0:
            raise ValueError(f"c must be > 0, got {self.c}")
        if float(self.ell_min) < 0.0:
            raise ValueError(f"ell_min must be >= 0, got {self.ell_min}")
        if self.target not in ("cubic_sigma", "lr"):
            raise ValueError(f"target must be 'cubic_sigma' or 'lr', got {self.target!r}")


@dataclass(frozen=True)
class SharpnessReport:
    """Per-step sharpness readout written into the smoke artifact."""

    ell_k: float
    scheduled: float
    target: SharpnessTarget
    c: float
    n_lanczos: int
    ell_min: float


def scheduled_value(ell_k: float, schedule: SharpnessSchedule) -> float:
    """Map a Ritz value to cubic ``sigma`` or a gradient learning rate."""
    sharpness = max(float(ell_k), float(schedule.ell_min))
    if schedule.target == "cubic_sigma":
        return float(schedule.c) * sharpness
    if sharpness == 0.0:
        return float("inf")
    return float(schedule.c) / sharpness


def make_report(ell_k: float, schedule: SharpnessSchedule) -> SharpnessReport:
    """Bundle the named schedule with the measured ``ell_k``."""
    return SharpnessReport(
        ell_k=float(ell_k),
        scheduled=scheduled_value(ell_k, schedule),
        target=schedule.target,
        c=float(schedule.c),
        n_lanczos=int(schedule.n_lanczos),
        ell_min=float(schedule.ell_min),
    )


__all__ = [
    "SharpnessReport",
    "SharpnessSchedule",
    "SharpnessTarget",
    "make_report",
    "scheduled_value",
]
