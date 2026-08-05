# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The differentiable coverage energy and the round-then-complete finaliser.

Thin wrappers over ``omnibias.shape.torch.ops`` that (a) turn ``(centers, gate_logits)``
into the scalar coverage energy or its Gauss-Newton residual vector for the optimiser, and
(b) convert the optimised soft cover into a *feasible* discrete cover (rounding the active
squares, then greedily completing any residual holes so the reported cover is always valid).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from omnibias.shape.torch import ops as shape
from torch import Tensor

from examples.min_square_cover.data import Instance, greedy_cover, is_feasible, squares_to_mask
from examples.min_square_cover.shapes import round_to_squares

SHAPE_KINDS = ("square", "disk")


def _occupancy(
    axes: Sequence[Tensor], centers: Tensor, side: float, beta: float, shape_kind: str
) -> Tensor:
    """Soft occupancy for the chosen surrogate shape (``"square"`` box or ``"disk"``).

    The disk uses the square's *inscribed* radius ``side / 2`` so its coverage is a conservative
    proxy for the square's; the discrete cover is always axis-aligned squares regardless, so the
    shape only changes the optimisation landscape / curvature (the P4 variant study).
    """
    if shape_kind == "square":
        return shape.soft_box(axes, centers, side, beta)
    if shape_kind == "disk":
        return shape.soft_disk(axes, centers, 0.5 * side, beta)
    raise ValueError(f"shape_kind must be one of {SHAPE_KINDS}, got {shape_kind!r}")


def scalar_energy(
    axes: Sequence[Tensor],
    centers: Tensor,
    gate_logits: Tensor,
    image: Tensor,
    side: float,
    beta: float,
    *,
    loss: str = "softplus",
    kappa: float = 4.0,
    lam: float = 0.0,
    shape_kind: str = "square",
) -> Tensor:
    """Scalar coverage energy for a scalar-closure optimiser (Adam / CubicNewton / TR-NCG)."""
    occ = _occupancy(axes, centers, side, beta, shape_kind)
    gates = torch.sigmoid(gate_logits)
    return shape.coverage_energy(occ, gates, image, loss=loss, kappa=kappa, lam=lam)


def residual_vector(
    axes: Sequence[Tensor],
    centers: Tensor,
    gate_logits: Tensor,
    image: Tensor,
    side: float,
    beta: float,
    *,
    lam: float = 0.0,
    shape_kind: str = "square",
) -> Tensor:
    """Under-coverage residual (+ optional count residual) for a Gauss-Newton optimiser.

    ``0.5 * mean(r^2)`` is the ``sq_hinge`` coverage energy; appending ``sqrt(lam) * gates``
    adds an L2 count pressure that the gates can shrink.
    """
    occ = _occupancy(axes, centers, side, beta, shape_kind)
    gates = torch.sigmoid(gate_logits)
    res = shape.coverage_residual(occ, gates, image)
    if lam:
        res = torch.cat([res, (lam**0.5) * gates])
    return res


def closed_form_newton_step(
    axes: Sequence[Tensor],
    centers: Tensor,
    gate_logits: Tensor,
    image: Tensor,
    side: float,
    beta: float,
    *,
    loss: str = "softplus",
    kappa: float = 4.0,
    lam: float = 0.0,
    eig_floor: float = 1e-3,
    max_backtracks: int = 25,
) -> float:
    r"""One **closed-form dense-Hessian** Newton step over ``[centers, gate_logits]``, in place.

    Assembles the exact gradient and full ``(K*D + K)^2`` Hessian of the coverage energy in
    closed form (``coverage_energy_grad`` / ``coverage_energy_hessian`` with ``wrt="all"``),
    then takes a **saddle-free** Newton direction ``d = -(|H| + eig_floor I)^{-1} g`` (Dauphin
    et al.: flip negative curvature via the symmetric eigendecomposition, so saddles become
    descent) with an Armijo backtrack on the same closed-form energy. This is the direct-dense
    counterpart to the matrix-free ``CubicNewton`` / ``TrustRegionNewtonCG`` arms; it exists to
    exercise the closed-form Hessian end to end. Returns the accepted energy.
    """
    with torch.no_grad():
        gates = torch.sigmoid(gate_logits)
        g = shape.coverage_energy_grad(
            axes, centers, side, beta, gates, image, loss=loss, kappa=kappa, lam=lam, wrt="all"
        )
        h = shape.coverage_energy_hessian(
            axes, centers, side, beta, gates, image, loss=loss, kappa=kappa, lam=lam, wrt="all"
        )
        h_sym = 0.5 * (h + h.T)
        evals, evecs = torch.linalg.eigh(h_sym)
        inv_abs = evecs @ torch.diag(1.0 / (evals.abs() + eig_floor)) @ evecs.T
        delta = -(inv_abs @ g)
        n_c = centers.numel()
        d_centers = delta[:n_c].reshape(centers.shape)
        d_gate = delta[n_c:]
        e0 = float(
            scalar_energy(axes, centers, gate_logits, image, side, beta, loss=loss, kappa=kappa, lam=lam)
        )
        slope = float(g @ delta)  # < 0 by construction (inv_abs is PD)
        alpha = 1.0
        for _ in range(max_backtracks):
            trial_c = centers + alpha * d_centers
            trial_g = gate_logits + alpha * d_gate
            e1 = float(
                scalar_energy(axes, trial_c, trial_g, image, side, beta, loss=loss, kappa=kappa, lam=lam)
            )
            if e1 <= e0 + 1e-4 * alpha * slope:
                centers.copy_(trial_c)
                gate_logits.copy_(trial_g)
                return e1
            alpha *= 0.5
        # No Armijo-acceptable step: fall back to a small gradient step (still monotone-safe).
        centers.copy_(centers - 1e-3 * g[:n_c].reshape(centers.shape))
        gate_logits.copy_(gate_logits - 1e-3 * g[n_c:])
        return e0


@dataclass
class DiscreteCover:
    """The rounded, guaranteed-feasible, irredundant discrete cover from a soft solution."""

    squares: list[tuple[int, int]]
    n_active: int  # squares the optimiser kept (gate >= threshold), before completion
    n_final: int  # total feasible squares (after hole-filling + redundancy pruning)
    n_completion: int  # squares greedy had to add to reach feasibility
    feasible_before_completion: bool


def _remove_redundant(
    image: Tensor, squares: list[tuple[int, int]], side: int
) -> list[tuple[int, int]]:
    """Drop any square whose 1-pixels are all covered by the others (irredundant cover).

    Squares covering the fewest 1-pixels are considered first, which removes the most squares.
    A cheap, exact post-process that makes the reported count independent of gate tuning.
    """
    ones = image.to(torch.bool)
    order = sorted(
        squares,
        key=lambda sq: int((ones & squares_to_mask(image.shape, [sq], side)).sum()),
    )
    kept = list(squares)
    for sq in order:
        trial = [s for s in kept if s != sq]
        if is_feasible(image, trial, side):
            kept = trial
    return kept


def complete_and_prune(
    image: Tensor, side: int, placements: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, bool]:
    """Greedily fill any holes left by ``placements``, then prune redundant squares.

    Returns ``(feasible_irredundant_squares, n_completion, feasible_before_completion)``. Shared by
    the soft-cover finaliser and the LP-register rounding so both produce a *feasible, irredundant*
    discrete cover by the same rule.
    """
    feasible_before = is_feasible(image, placements, side)
    extra: list[tuple[int, int]] = []
    if not feasible_before:
        covered = torch.zeros(tuple(image.shape), dtype=torch.bool)
        for r, c in placements:
            covered[r : r + side, c : c + side] = True
        residual_img = (image.to(torch.bool) & ~covered).to(image.dtype)
        extra = greedy_cover(residual_img, side)
        for sq in extra:
            if sq not in placements:
                placements.append(sq)
    kept = _remove_redundant(image, placements, side)
    return kept, len(extra), feasible_before


def finalize_cover(
    centers: Tensor, gate_logits: Tensor, instance: Instance, *, threshold: float = 0.5
) -> DiscreteCover:
    """Round active soft squares, greedily fill holes, then prune redundant squares.

    The result is always a *feasible, irredundant* discrete cover: rounding + hole-filling
    guarantee feasibility, and redundancy pruning guarantees no square can be removed while
    still covering every 1-pixel.
    """
    gates = torch.sigmoid(gate_logits.detach())
    placements = round_to_squares(centers.detach(), gates, instance, threshold=threshold)
    n_active = len(placements)
    squares, n_completion, feasible_before = complete_and_prune(
        instance.image, instance.side, placements
    )
    return DiscreteCover(
        squares=squares,
        n_active=n_active,
        n_final=len(squares),
        n_completion=n_completion,
        feasible_before_completion=feasible_before,
    )


__all__ = [
    "DiscreteCover",
    "SHAPE_KINDS",
    "closed_form_newton_step",
    "complete_and_prune",
    "finalize_cover",
    "residual_vector",
    "scalar_energy",
]
