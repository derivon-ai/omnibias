# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic containers for omnibias-routing.

The differentiable relaxation layers live in :mod:`omnibias.routing.jax` and
:mod:`omnibias.routing.torch`; the numpy decoder / exact oracle live in
:mod:`omnibias.routing._core.decode`; the certificate lives in
:mod:`omnibias.routing.certify`. These containers only hold data so the two
backends present an identical surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

ArrayT = TypeVar("ArrayT")

_TINY = 1e-12


@dataclass(frozen=True)
class RoutingProblem:
    r"""A (directed / asymmetric) TSP instance: an ``n x n`` arc-cost matrix.

    ``cost[i, j]`` is the cost of the directed arc ``i -> j`` (the diagonal is
    ignored). A tour is a cyclic permutation visiting every city once; its cost is
    the sum of its ``n`` traversed arcs. Symmetric (Euclidean) instances are the
    special case ``cost == cost.T``.

    Attributes
    ----------
    cost:
        ``(n, n)`` numpy arc-cost matrix (float).
    coords:
        Optional ``(n, 2)`` city coordinates (Euclidean instances), for plotting /
        provenance only; the cost matrix is authoritative.
    name:
        Optional label.
    """

    cost: np.ndarray
    coords: np.ndarray | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        c = np.asarray(self.cost, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError(f"cost must be a square (n, n) matrix, got shape {c.shape}")
        if c.shape[0] < 3:
            raise ValueError("a tour needs at least 3 cities")
        object.__setattr__(self, "cost", c)

    @property
    def n(self) -> int:
        return int(self.cost.shape[0])

    @property
    def symmetric(self) -> bool:
        return bool(np.allclose(self.cost, self.cost.T))

    @classmethod
    def from_coords(cls, coords: Any, *, name: str | None = None) -> RoutingProblem:
        """Build a symmetric Euclidean instance from ``(n, 2)`` coordinates."""
        pts = np.asarray(coords, dtype=float)
        diff = pts[:, None, :] - pts[None, :, :]
        cost = np.sqrt(np.sum(diff * diff, axis=-1))
        return cls(cost=cost, coords=pts, name=name)


@dataclass(frozen=True)
class RelaxationSchedule:
    r"""Homotopy schedule for the differentiable (unrolled) temperature-collapse relaxation.

    The relaxation is a convex LP over a poly-size TSP polytope (assignment / flow /
    Held-Karp) solved by the omnibias temperature-collapse penalty, *unrolled* for
    differentiability (the proven ``route.py`` pattern): a small quadratic
    regulariser ``reg`` makes it strongly convex (smooth cost-sensitive gradients),
    each inequality contributes the hard-hinge exterior penalty ``mu/2 relu(.)^2``
    and each equality the quadratic penalty ``mu/2 (.)^2`` -- both with a closed-form
    gradient -- minimised by accelerated (Nesterov) gradient descent along a
    geometric ``mu`` homotopy with the closed-form Lipschitz step. This is the
    well-conditioned hard-hinge limit (no ``beta`` blow-up), so it stays ``jit`` /
    ``grad`` friendly. Defaults are eval-quality; :meth:`fast` is enough to train
    *through* the relaxation.

    Terminology: "temperature-collapse penalty" here is the feasibility sense of
    "collapse" (a hard-hinge constraint force), distinct from the
    **founding bias collapse** (multi-bias ``delta -> 0`` limit to
    ``sigma^(K-1)``, a derivative; see ``docs/theory.md``).

    Attributes
    ----------
    reg:
        Quadratic regulariser weight (strong convexity for stable gradients).
    mu0, mu_growth:
        Initial penalty weight and geometric growth factor per stage.
    stages:
        Number of homotopy stages (each warm-starts the next).
    steps:
        Nesterov steps per stage.
    step_safety:
        Fraction of the Lipschitz step to take (``0 < step_safety <= 1``).
    """

    reg: float = 0.10
    mu0: float = 1.0
    mu_growth: float = 1.8
    stages: int = 9
    steps: int = 150
    step_safety: float = 0.9

    def __post_init__(self) -> None:
        if self.reg <= 0.0:
            raise ValueError("reg must be > 0 (strong convexity keeps the layer differentiable)")
        if self.mu0 <= 0.0:
            raise ValueError("mu0 must be > 0")
        if self.mu_growth < 1.0:
            raise ValueError("mu_growth must be >= 1")
        if self.stages < 1 or self.steps < 1:
            raise ValueError("stages and steps must be >= 1")
        if not 0.0 < self.step_safety <= 1.0:
            raise ValueError("step_safety must be in (0, 1]")

    def mus(self) -> list[float]:
        """The penalty weight ``mu`` at each homotopy stage."""
        out, mu = [], self.mu0
        for _ in range(self.stages):
            out.append(mu)
            mu *= self.mu_growth
        return out

    @classmethod
    def fast(cls) -> RelaxationSchedule:
        """A lighter schedule for training *through* the relaxation."""
        return cls(mu0=1.0, mu_growth=2.0, stages=5, steps=60)


@dataclass(frozen=True)
class TourSolution(Generic[ArrayT]):
    r"""A decoded tour and (optionally) the fractional relaxation it was rounded from.

    Attributes
    ----------
    tour:
        City visiting order as a permutation of ``range(n)`` starting at city 0; the
        implied cycle returns ``tour[-1] -> tour[0]``.
    cost:
        The tour cost under the problem's cost matrix (an *upper* bound on the
        optimum).
    relaxed:
        Optional fractional arc-use vector / matrix produced by the differentiable
        relaxation (the heatmap the tour was decoded from), backend array.
    """

    tour: tuple[int, ...]
    cost: float
    relaxed: ArrayT | None = None

    @property
    def n(self) -> int:
        return len(self.tour)


@dataclass(frozen=True)
class HeldKarpCertificate:
    r"""A rigorous optimality-gap certificate for a decoded tour.

    Combines a rigorous **lower** bound on the optimal tour cost (an LP relaxation
    dual bound, enclosed by outward-rounded interval arithmetic via the
    Neumaier-Shcherbina certificate in :func:`omnibias.convex.lp_dual_lower_bound`,
    which is built on :mod:`omnibias.core.verified`) with the decoded tour's cost as
    the **upper** bound. The true optimum is provably sandwiched
    ``lower_bound <= optimum <= tour_cost``; the gap certifies how close to optimal
    the tour is -- **without** any exact-optimality (P = NP) claim.

    Attributes
    ----------
    lower_bound:
        Rigorous lower bound on the optimal tour cost for this cost matrix.
    tour_cost:
        Decoded tour cost (the certified upper bound).
    relaxation:
        Which relaxation produced the lower bound (``"held_karp"`` / ``"flow"`` /
        ``"assignment"``).
    certified:
        ``True`` iff ``lower_bound`` came from the rigorous interval enclosure
        (outward-rounded Neumaier-Shcherbina bound); ``False`` if it is a plain
        float LP value (still a valid bound, just not interval-sealed).
    """

    lower_bound: float
    tour_cost: float
    relaxation: str
    certified: bool

    @property
    def absolute_gap(self) -> float:
        """Certified absolute optimality gap ``tour_cost - lower_bound`` (``>= 0``)."""
        return self.tour_cost - self.lower_bound

    @property
    def relative_gap(self) -> float:
        """Certified relative gap ``(tour_cost - lower_bound) / |lower_bound|``."""
        return self.absolute_gap / max(abs(self.lower_bound), _TINY)

    @property
    def is_sound(self) -> bool:
        """Whether the sandwich holds (``lower_bound <= tour_cost`` within rounding)."""
        return self.lower_bound <= self.tour_cost + 1e-9


__all__ = [
    "HeldKarpCertificate",
    "RelaxationSchedule",
    "RoutingProblem",
    "TourSolution",
]
