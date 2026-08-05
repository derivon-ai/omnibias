# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic containers for the omnibias-control safety filter.

The numeric filter / builders live in :mod:`omnibias.control.jax` and
:mod:`omnibias.control.torch`; these containers only hold the data so the two
backends present an identical surface. Arrays are stored untyped (``Any``) so the
same :class:`SafeAction` works for ``jax.Array`` and ``torch.Tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class FilterSchedule:
    r"""Homotopy schedule for the differentiable CBF-QP projection.

    The filter solves the per-sample projection
    ``a* = argmin_a 1/2||a - a_nom||^2 s.t. G a <= h`` with an *exterior* hard-hinge
    penalty ``mu/2 sum relu(G a - h)^2`` minimised by accelerated (Nesterov) gradient
    descent over ``stages`` geometric increases of the penalty weight ``mu``. The
    hinge gradient is the temperature-collapse unit ``mu G^T relu(G a - h)`` and the step is
    the closed-form Lipschitz value ``safety / (1 + mu ||G||_F^2)`` (no ``beta`` term,
    unlike the tempered softplus penalty -- this is the well-conditioned limit).

    Defaults are **eval-quality** (near-exact projection). For training *through* the
    filter, a lighter schedule is enough (only the gradient direction is needed);
    use :meth:`fast`.

    Attributes
    ----------
    mu0, mu_growth:
        Initial penalty weight and its geometric growth factor per stage.
    stages:
        Number of homotopy stages (each warm-starts the next).
    steps:
        Nesterov steps per stage.
    safety:
        Fraction of the Lipschitz step to take (``0 < safety <= 1``).
    """

    mu0: float = 2.0
    mu_growth: float = 2.0
    stages: int = 9
    steps: int = 60
    safety: float = 0.9

    def __post_init__(self) -> None:
        if self.mu0 <= 0.0:
            raise ValueError("mu0 must be > 0")
        if self.mu_growth < 1.0:
            raise ValueError("mu_growth must be >= 1")
        if self.stages < 1 or self.steps < 1:
            raise ValueError("stages and steps must be >= 1")
        if not 0.0 < self.safety <= 1.0:
            raise ValueError("safety must be in (0, 1]")

    @classmethod
    def fast(cls) -> FilterSchedule:
        """A lighter schedule for training *through* the filter."""
        return cls(mu0=2.0, mu_growth=2.2, stages=4, steps=20)


@dataclass(frozen=True)
class CBFSpec:
    r"""Control-barrier-function design constants.

    A relative-degree-``d`` barrier ``h(x)`` yields the (exponential-CBF) row
    ``L_f^d h + L_g L_f^{d-1} h a + sum_k gains[k] * (dh/dt-chain) >= 0``. With one
    class-K gain (``relative_degree == 1``): ``L_f h + L_g h a + gains[0] h >= 0``.
    With two (``relative_degree == 2``): ``L_f^2 h + L_g L_f h a + (g0+g1) L_f h +
    g0 g1 h >= 0`` (poles at ``-g0, -g1``). Rearranged to ``G a <= h_vec``.

    Attributes
    ----------
    gains:
        Class-K linear gains ``(alpha_1, ...)``; ``len(gains)`` is the relative degree
        (only 1 and 2 are supported).
    a_max:
        Symmetric actuator box ``||a||_inf <= a_max`` appended as ``2 * dim`` rows.
        ``None`` disables the box (unbounded actuation).
    """

    gains: tuple[float, ...]
    a_max: float | None = None

    def __post_init__(self) -> None:
        if len(self.gains) not in (1, 2):
            raise ValueError(
                f"only relative degree 1 or 2 supported, got {len(self.gains)} gains"
            )
        if any(g <= 0.0 for g in self.gains):
            raise ValueError("class-K gains must be > 0")
        if self.a_max is not None and self.a_max <= 0.0:
            raise ValueError("a_max must be > 0 (or None)")

    @property
    def relative_degree(self) -> int:
        return len(self.gains)


@dataclass(frozen=True)
class SafeAction(Generic[ArrayT]):
    """Diagnostics wrapper around a filtered action.

    Attributes
    ----------
    action:
        The filtered (safe) action ``a*``, shape ``(B, d)``.
    nominal:
        The task-driven input ``a_nom`` before filtering, shape ``(B, d)``.
    residual:
        Per-sample worst constraint residual ``max_i (G a* - h)_i``, shape ``(B,)``
        (``<= 0`` means the returned action is feasible).
    """

    action: ArrayT
    nominal: ArrayT
    residual: ArrayT


@dataclass(frozen=True)
class RecoverableCertificate:
    r"""Result of a rigorous recoverable-set enclosure.

    ``certify_recoverable`` runs interval branch-and-bound on the recoverability
    margin ``phi(s)`` (the CBF-QP is feasible iff ``phi(s) >= 0``). A rigorous
    ``f_lower >= 0`` proves the safety filter is feasible over the *entire* state box
    (the recoverable set); ``f_upper < 0`` is a rigorous witness that some state in the
    box is not recoverable.

    Attributes
    ----------
    f_lower, f_upper:
        Rigorous lower / upper bounds on ``min_box phi``.
    boxes_explored:
        Number of interval boxes the branch-and-bound explored.
    converged:
        Whether the ``f_upper - f_lower`` gap closed below ``tol``.
    """

    f_lower: float
    f_upper: float
    boxes_explored: int
    converged: bool

    @property
    def certified(self) -> bool:
        """``True`` iff the whole box is provably recoverable (``f_lower >= 0``)."""
        return self.f_lower >= 0.0


__all__ = [
    "CBFSpec",
    "FilterSchedule",
    "RecoverableCertificate",
    "SafeAction",
]
