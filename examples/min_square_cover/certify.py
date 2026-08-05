# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified lower bounds, exact feasibility, and a robustness margin for a discrete cover.

This is the *second register*: alongside the continuous solve we bound the problem rigorously.

* :func:`area_lower_bound` -- an always-available, certified-by-construction lower bound
  (a square covers at most ``side^2`` pixels, so at least ``ceil(n_ones / side^2)`` squares
  are needed).
* :func:`lp_lower_bound` -- a *tighter* certified lower bound from the LP relaxation of the
  set-cover ILP, solved with :func:`omnibias.convex.torch.solve_lp` (verified interior-point).
  Optional: returns ``None`` if ``omnibias-convex`` is not installed.
* :func:`verify_cover` -- exact check that every 1-pixel is covered (no floating point).
* :func:`robustness_margin` -- the largest ``d`` such that shrinking every placed square by
  ``d`` pixels per side still covers the image (equivalently, tolerance to a ``d``-pixel
  inward placement error).

Together they turn a heuristic count ``K`` into a certified statement:
``ceil(lower_bound) <= optimum <= K``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor

from examples.min_square_cover.data import is_feasible


def area_lower_bound(image: Tensor, side: int) -> int:
    """Certified lower bound ``ceil(n_ones / side_eff^2)`` on the number of squares."""
    m, n = int(image.shape[0]), int(image.shape[1])
    cover_max = min(side, m) * min(side, n)
    n_ones = int(image.sum())
    if cover_max <= 0 or n_ones == 0:
        return 0
    return math.ceil(n_ones / cover_max)


def _candidate_positions(shape: tuple[int, int], side: int) -> list[tuple[int, int]]:
    m, n = shape
    side = max(1, min(side, m, n))
    return [(r, c) for r in range(m - side + 1) for c in range(n - side + 1)]


def verify_cover(image: Tensor, squares: list[tuple[int, int]], side: int) -> bool:
    """Exact (integer) check that ``squares`` cover every 1-pixel of ``image``."""
    return is_feasible(image, squares, side)


def robustness_margin(
    image: Tensor, squares: list[tuple[int, int]], side: int, *, max_delta: int = 3
) -> int:
    """Largest ``d`` such that shrinking each square by ``d`` px/side still covers the image.

    ``d == 0`` means the cover is exactly tight somewhere; larger ``d`` means every 1-pixel has
    at least ``d`` pixels of overlap slack (tolerance to inward placement error).
    """
    if not verify_cover(image, squares, side):
        return -1
    for d in range(1, max_delta + 1):
        s2 = side - 2 * d
        if s2 <= 0:
            return d - 1
        shrunk = [(r + d, c + d) for r, c in squares]
        if not is_feasible(image, shrunk, s2):
            return d - 1
    return max_delta


@dataclass
class LPCover:
    """The LP-relaxation fractional set cover: candidate positions and their weights."""

    positions: list[tuple[int, int]]  # candidate square top-lefts
    weights: list[float]  # fractional selection x_i in [0, 1], aligned with positions
    objective: float  # certified lower bound on the LP optimum (obj - gap), <= ILP optimum


def lp_fractional_cover(image: Tensor, side: int, *, max_positions: int = 2000) -> LPCover | None:
    """Solve the fractional set-cover LP relaxation and return the whole solution.

    Builds ``min sum_i x_i  s.t.  (cover) x >= 1, 0 <= x <= 1`` over every valid square position
    and solves it with :func:`omnibias.convex.torch.solve_lp` (verified interior point). The
    returned :class:`LPCover` exposes the fractional weights ``x_i`` -- which both lower-bound the
    integer optimum (via ``objective``) and warm-start the continuous solve (top-weighted
    positions become initial centers). Returns ``None`` if ``omnibias-convex`` is unavailable or
    the instance exceeds ``max_positions`` candidate squares.
    """
    try:
        import torch
        from omnibias.convex.torch import solve_lp
    except ImportError:
        return None

    positions = _candidate_positions(image.shape, side)
    if len(positions) > max_positions:
        return None
    side_eff = max(1, min(side, image.shape[0], image.shape[1]))
    ones = image.to(torch.bool).nonzero(as_tuple=False).tolist()
    if not ones:
        return LPCover(positions=positions, weights=[0.0] * len(positions), objective=0.0)
    n_pos = len(positions)
    cover = torch.zeros(len(ones), n_pos, dtype=torch.float64)
    for i, (r, c) in enumerate(positions):
        for j, (py, px) in enumerate(ones):
            if r <= py < r + side_eff and c <= px < c + side_eff:
                cover[j, i] = 1.0
    eye = torch.eye(n_pos, dtype=torch.float64)
    a = torch.cat([-cover, -eye, eye], dim=0)
    b = torch.cat([
        -torch.ones(len(ones), dtype=torch.float64),
        torch.zeros(n_pos, dtype=torch.float64),
        torch.ones(n_pos, dtype=torch.float64),
    ])
    c_obj = torch.ones(n_pos, dtype=torch.float64)
    sol = solve_lp(c_obj, a, b)
    weights = sol.x.clamp(0.0, 1.0).tolist()
    # ``obj - gap`` is a rigorous lower bound on the LP optimum (the surrogate duality gap
    # ``m / t`` bounds ``primal - dual`` at the centered iterate), hence on the ILP optimum --
    # robust even when a degenerate LP stops early with ``converged=False``.
    objective = float(sol.obj) - float(sol.gap)
    return LPCover(positions=positions, weights=weights, objective=objective)


def lp_lower_bound(image: Tensor, side: int, *, max_positions: int = 2000) -> float | None:
    """LP-relaxation lower bound on the ILP optimum (the LP optimum ``sum_i x_i``).

    Thin wrapper over :func:`lp_fractional_cover`; returns ``None`` when the LP is unavailable.
    """
    lp = lp_fractional_cover(image, side, max_positions=max_positions)
    return None if lp is None else lp.objective


def lp_rounded_cover(
    image: Tensor, side: int, *, weight_threshold: float = 0.5, max_positions: int = 2000
) -> list[tuple[int, int]] | None:
    """LP-register *upper* bound: round the fractional LP to a feasible discrete cover.

    Selects every candidate square whose LP weight is ``>= weight_threshold``, then greedily fills
    residual holes and prunes redundant squares (the same finaliser the soft-cover path uses), so
    the result is a *feasible, irredundant* cover. Pairs with :func:`lp_lower_bound`: the LP
    register then yields both a certified lower bound and a concrete cover. Returns ``None`` when
    ``omnibias-convex`` is unavailable (or the instance exceeds ``max_positions`` candidates).
    """
    lp = lp_fractional_cover(image, side, max_positions=max_positions)
    if lp is None:
        return None
    from examples.min_square_cover.coverage import complete_and_prune

    placements = [
        pos for pos, w in zip(lp.positions, lp.weights, strict=False) if w >= weight_threshold
    ]
    squares, _, _ = complete_and_prune(image, side, placements)
    return squares


@dataclass
class Certificate:
    """A certified report on one discrete cover."""

    feasible: bool
    n_used: int
    area_lower_bound: int
    lp_lower_bound: float | None
    robustness_margin: int
    optimality_ratio: float  # n_used / ceil(best available lower bound)


def certify_cover(
    image: Tensor,
    squares: list[tuple[int, int]],
    side: int,
    *,
    with_lp: bool = False,
    max_delta: int = 3,
) -> Certificate:
    """Assemble a :class:`Certificate` for ``squares`` (optionally including the LP bound)."""
    feasible = verify_cover(image, squares, side)
    area_lb = area_lower_bound(image, side)
    lp_lb = lp_lower_bound(image, side) if with_lp else None
    best_lb = area_lb
    if lp_lb is not None:
        best_lb = max(best_lb, math.ceil(lp_lb - 1e-6))
    n_used = len(squares)
    ratio = n_used / max(best_lb, 1)
    return Certificate(
        feasible=feasible,
        n_used=n_used,
        area_lower_bound=area_lb,
        lp_lower_bound=lp_lb,
        robustness_margin=robustness_margin(image, squares, side, max_delta=max_delta),
        optimality_ratio=ratio,
    )


__all__ = [
    "Certificate",
    "LPCover",
    "area_lower_bound",
    "certify_cover",
    "lp_fractional_cover",
    "lp_lower_bound",
    "lp_rounded_cover",
    "robustness_margin",
    "verify_cover",
]
