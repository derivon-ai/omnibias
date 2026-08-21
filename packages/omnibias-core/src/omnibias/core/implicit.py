# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Implicit / DEQ algebra (theory 08-08).

An equilibrium ``u = sigma(W u + x)`` is a root of
``F(u) = u - sigma(W u + x)``. The implicit-function theorem differentiates
the root with exact ``sigma'`` from the founding bias collapse
(``delta -> 0``): one linear solve against
``I - diag(sigma'(z)) W`` replaces unrolled backprop through a
fixed-point iteration.

This module is backend-free: config, the contraction raise, and the
infinity-norm spectral-radius bound. Tensor Newton / Banach iteration
and the IFT VJP live in ``omnibias.{torch,jax}.implicit``.

No temperature collapse. Not a global min, not CCF stretch, and not a
continuum PDE existence claim. IFT is the chain rule at a fixed point,
not an absence of the chain rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DEQSolver = Literal["newton", "iterate", "anderson"]

HONESTY: dict[str, bool] = {
    "navier_stokes_proof_claim": False,
    "ccf_stretch_cleared": False,
    "global_min_claim": False,
    "unrolled_bptt_claim": False,
    "greedy_only_claimed_optimal": False,
}


class DEQNotContractive(ValueError):
    """Raised when ``require_contraction=True`` and the bound is ``>= 1``."""


class DEQSolverUnknown(ValueError):
    """Raised when the solver name is not implemented."""


@dataclass(frozen=True)
class DEQConfig:
    """Fixed-point solver budget and contraction policy.

    ``newton`` is the gated method. ``iterate`` is Banach iteration.
    ``anderson`` is recorded extra and raises
    :class:`DEQSolverUnknown` until it earns a gate.
    """

    solver: DEQSolver = "newton"
    max_iter: int = 50
    tol: float = 1e-10
    require_contraction: bool = False

    def __post_init__(self) -> None:
        if self.solver not in ("newton", "iterate", "anderson"):
            raise ValueError(
                f"solver must be 'newton', 'iterate', or 'anderson', "
                f"got {self.solver!r}"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol <= 0.0 or not math.isfinite(self.tol):
            raise ValueError(f"tol must be a finite number > 0, got {self.tol}")


def spectral_radius_inf_bound(
    abs_sigma_prime_times_row_l1: float,
) -> float:
    """Infinity-norm bound on ``diag(sigma') W``.

    ``max_i |sigma'_i| ||W_{i,:}||_1`` is enough for the gate. A value
    ``< 1`` is a sufficient Banach contraction test, not a necessary one.
    """
    bound = float(abs_sigma_prime_times_row_l1)
    if bound < 0.0 or not math.isfinite(bound):
        raise ValueError(
            f"spectral-radius bound must be a finite number >= 0, got {bound}"
        )
    return bound


def reject_deq_contraction(bound: float, *, require: bool) -> None:
    """G11: raise rather than silently unroll when the bound is ``>= 1``."""
    radius = spectral_radius_inf_bound(bound)
    if require and radius >= 1.0:
        raise DEQNotContractive(
            f"spectral_radius_bound={radius} >= 1; refuse to unroll. "
            "Pass require_contraction=False to attempt Newton without "
            "a Banach certificate"
        )


def reject_anderson(solver: str) -> None:
    """Anderson is extra; the gate uses Newton."""
    if solver == "anderson":
        raise DEQSolverUnknown(
            "anderson acceleration is extra and is not the gated solver; "
            "use solver='newton' or solver='iterate'"
        )


def honesty_payload() -> dict[str, bool]:
    """Sealed artifact keys. Values are copies."""
    return dict(HONESTY)


__all__ = [
    "DEQConfig",
    "DEQNotContractive",
    "DEQSolver",
    "DEQSolverUnknown",
    "HONESTY",
    "honesty_payload",
    "reject_anderson",
    "reject_deq_contraction",
    "spectral_radius_inf_bound",
]
