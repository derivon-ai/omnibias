# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic collocation-point sampling (returns numpy arrays).

The solver drivers convert the numpy point sets to backend tensors with the
requested dtype. Two strategies are supported in v1: a tensor-product ``grid``
and uniform ``random`` sampling.

:class:`RefinementSpec` adds *residual-adaptive refinement* (RAR): the driver
scores fresh candidate points by residual magnitude and the selection rules here
decide which of them join the interior set. Scoring needs a backend (it is a
forward pass), so it stays in the driver; candidate generation and selection are
pure numpy and live here, which keeps ``_core`` backend-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from omnibias.pinn.solver._core.conditions import BoundaryCondition
from omnibias.pinn.solver._core.domain import Domain


@dataclass(frozen=True)
class CollocationSpec:
    """How to sample interior / boundary / initial points for a solve.

    ``n_interior`` is a *per-axis* count for the ``"grid"`` method (the total
    is ``n_interior ** ndim``) and an absolute count for ``"random"``.
    """

    n_interior: int = 16
    n_boundary: int = 16
    method: str = "grid"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in ("grid", "random"):
            raise ValueError(f"method must be 'grid' or 'random', got {self.method!r}")
        if self.n_interior < 1:
            raise ValueError("n_interior must be >= 1")


@dataclass(frozen=True)
class RefinementSpec:
    """Residual-adaptive refinement (RAR) of the interior collocation set.

    A uniform grid spends points where the solution is already easy. Every
    ``every`` iterations the driver draws ``n_candidates`` fresh uniform points,
    scores them by residual magnitude, and adds the ones that carry the error --
    so a shock or a sharp front gets resolved without globally refining.

    Parameters
    ----------
    every
        Refine every ``every`` optimiser iterations (never at iteration 0).
    n_candidates
        Candidates drawn per refinement round.
    n_add
        Points kept per round, subject to ``max_points``.
    strategy
        ``"proportional"`` (the default) samples without replacement with
        probability proportional to ``score ** power`` (RAD, Wu et al. 2023).
        ``"greedy"`` keeps the ``n_add`` largest scores (RAR, Lu et al. 2021).

        The default is measured, not inherited: on the benchmark in
        ``docs/benchmarks.md`` the greedy rule over-concentrates when the residual
        is diffuse and *lost* to uniform sampling on the mildest problem
        (viscosity 0.01 Burgers, median max-norm ratio 0.96, 3/8 seeds), while
        proportional never lost on the median anywhere. Greedy is the stronger
        rule once the feature is genuinely sharp (4.07x vs 3.42x on the front
        problem), so it stays available.
    power
        Exponent on the score for ``"proportional"``; ignored by ``"greedy"``.
        The default ``2.0`` is the value the benchmark was run at.
    max_points
        Hard cap on the total interior point count, so a long run cannot grow the
        residual batch without bound.
    seed
        Base seed. Round ``k`` draws with ``seed + k`` so rounds never repeat.
    """

    every: int = 50
    n_candidates: int = 512
    n_add: int = 32
    strategy: str = "proportional"
    power: float = 2.0
    max_points: int = 4096
    seed: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in ("greedy", "proportional"):
            raise ValueError(
                f"strategy must be 'greedy' or 'proportional', got {self.strategy!r}"
            )
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")
        if self.n_add < 1:
            raise ValueError(f"n_add must be >= 1, got {self.n_add}")
        if self.power <= 0.0:
            raise ValueError(f"power must be > 0, got {self.power}")
        if self.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {self.max_points}")


def _axis_grid(lo: float, hi: float, n: int, *, periodic: bool) -> np.ndarray:
    if periodic:
        # exclude the duplicated right endpoint on periodic axes
        return np.linspace(lo, hi, n, endpoint=False)
    # interior nodes only (avoid landing exactly on the boundary)
    step = (hi - lo) / (n + 1)
    return lo + step * np.arange(1, n + 1)


def interior_points(domain: Domain, spec: CollocationSpec) -> np.ndarray:
    """Interior collocation points, shape ``(N, ndim)``."""
    cs = domain.coordinate_spec
    if spec.method == "grid":
        per_axis = [
            _axis_grid(lo, hi, spec.n_interior, periodic=cs.periodicity[i])
            for i, (lo, hi) in enumerate(domain.bounds)
        ]
        mesh = [g.ravel() for g in np.meshgrid(*per_axis, indexing="ij")]
        return np.stack([m.ravel() for m in mesh], axis=-1)
    rng = np.random.default_rng(spec.seed)
    cols = [
        rng.uniform(lo, hi, size=spec.n_interior) for (lo, hi) in domain.bounds
    ]
    return np.stack(cols, axis=-1)


def candidate_points(
    domain: Domain,
    spec: CollocationSpec,
    refinement: RefinementSpec,
    *,
    round_index: int,
) -> np.ndarray:
    """Fresh uniform-random refinement candidates for round ``round_index``."""
    draw = replace(
        spec,
        method="random",
        n_interior=refinement.n_candidates,
        seed=refinement.seed + round_index,
    )
    return interior_points(domain, draw)


def select_refinement_points(
    candidates: np.ndarray,
    scores: np.ndarray,
    refinement: RefinementSpec,
    *,
    n_existing: int,
    round_index: int = 0,
) -> np.ndarray:
    """Choose which scored ``candidates`` join an interior set of ``n_existing``.

    ``scores`` are non-negative residual magnitudes, one per candidate. Returns a
    ``(k, ndim)`` array with ``k <= refinement.n_add`` (``k == 0`` once
    ``refinement.max_points`` is reached).
    """
    if candidates.shape[0] != scores.shape[0]:
        raise ValueError(
            f"got {candidates.shape[0]} candidates but {scores.shape[0]} scores"
        )
    budget = refinement.max_points - int(n_existing)
    n_add = min(refinement.n_add, budget, int(candidates.shape[0]))
    if n_add <= 0:
        return np.empty((0, candidates.shape[1]), dtype=candidates.dtype)
    finite = np.where(np.isfinite(scores), np.abs(scores), 0.0)
    if refinement.strategy == "greedy":
        keep = np.argsort(-finite, kind="stable")[:n_add]
        return np.asarray(candidates[keep])
    weights = finite**refinement.power
    total = float(weights.sum())
    if total <= 0.0:  # a converged residual carries no signal -> sample uniformly
        weights = np.ones_like(weights)
        total = float(weights.sum())
    rng = np.random.default_rng(refinement.seed + round_index)
    keep = rng.choice(
        candidates.shape[0], size=n_add, replace=False, p=weights / total
    )
    return np.asarray(candidates[keep])


def boundary_points(
    domain: Domain,
    spec: CollocationSpec,
    *,
    axis: str,
    side: str,
) -> np.ndarray:
    """Points on one face (``axis`` fixed to its ``lo``/``hi`` bound)."""
    cs = domain.coordinate_spec
    ai = cs.axis_index(axis)
    lo, hi = domain.bounds[ai]
    fixed = lo if side == "lo" else hi
    other_axes = [i for i in range(domain.ndim) if i != ai]
    if not other_axes:
        return np.array([[fixed]], dtype=float)
    per_axis = []
    for i in other_axes:
        blo, bhi = domain.bounds[i]
        per_axis.append(_axis_grid(blo, bhi, spec.n_boundary, periodic=cs.periodicity[i]))
    grids = np.meshgrid(*per_axis, indexing="ij")
    cols: list[np.ndarray] = []
    k = 0
    for i in range(domain.ndim):
        if i == ai:
            cols.append(None)  # type: ignore[arg-type]
        else:
            cols.append(grids[k].ravel())
            k += 1
    n_pts = cols[other_axes[0]].shape[0]
    out = np.empty((n_pts, domain.ndim), dtype=float)
    for i in range(domain.ndim):
        out[:, i] = fixed if i == ai else cols[i]
    return out


def spatial_boundary_points(domain: Domain, spec: CollocationSpec) -> np.ndarray:
    """All non-periodic spatial-boundary faces, concatenated."""
    cs = domain.coordinate_spec
    faces: list[np.ndarray] = []
    for axis in domain.spatial_axes:
        if cs.is_periodic(axis):
            continue
        for side in ("lo", "hi"):
            faces.append(boundary_points(domain, spec, axis=axis, side=side))
    if not faces:
        return np.empty((0, domain.ndim), dtype=float)
    return np.concatenate(faces, axis=0)


def bc_faces(domain: Domain, bc: BoundaryCondition) -> list[tuple[str, str]]:
    """The ``(axis, side)`` faces a boundary condition applies to.

    ``axis=None`` (the Dirichlet default) means every non-periodic spatial face.
    """
    cs = domain.coordinate_spec
    if bc.axis is None:
        return [
            (ax, side)
            for ax in domain.spatial_axes
            if not cs.is_periodic(ax)
            for side in ("lo", "hi")
        ]
    sides = (bc.side,) if bc.side is not None else ("lo", "hi")
    return [(bc.axis, s) for s in sides]


def periodic_axes(domain: Domain, bc: BoundaryCondition) -> tuple[str, ...]:
    """Axes a periodic boundary condition ties together across the seam.

    An explicit ``bc.axis`` is used as-is; otherwise every spatial axis the
    domain declares periodic. Shared by the hard planner and both assemblers so
    the three paths cannot disagree about which seams exist.
    """
    if bc.axis is not None:
        return (bc.axis,)
    cs = domain.coordinate_spec
    return tuple(a for a in domain.spatial_axes if cs.is_periodic(a))


def initial_slice_points(
    domain: Domain, spec: CollocationSpec, *, t0: float | None = None
) -> np.ndarray:
    """Spatial grid at the initial time (time axis fixed to ``t0``)."""
    if domain.time_axis is None:
        raise ValueError("initial_slice_points requires a time axis")
    t_lo, _ = domain.time_bounds()
    t_val = t_lo if t0 is None else t0
    pts = boundary_points(domain, spec, axis=domain.time_axis, side="lo")
    ai = domain.coordinate_spec.axis_index(domain.time_axis)
    pts[:, ai] = t_val
    return pts


__all__ = [
    "CollocationSpec",
    "RefinementSpec",
    "bc_faces",
    "boundary_points",
    "candidate_points",
    "initial_slice_points",
    "interior_points",
    "periodic_axes",
    "select_refinement_points",
    "spatial_boundary_points",
]
