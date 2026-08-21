# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Arrangement LP with learned facets (theory 03-02).

An inequality LP ``min c·x  s.t.  A x <= b`` is one cell of the
arrangement ``{a_i · x = b_i}``. This module is a learned-constraint
front end plus that dictionary -- **not** a new LP algorithm. It wraps
``solve_lp`` / ``lp_layer`` and the Neumaier-Shcherbina
``lp_dual_lower_bound``. There is no uncorrected float dual path.

Soft cell membership ``prod_i sigma(beta (b_i - a_i · x))`` is
**temperature collapse** (``beta -> inf``, feasibility). It is **not**
the founding bias collapse (``delta -> 0`` to ``sigma^(K-1)``). Do not conflate
the two. Vertex enumeration is exponential and is a
verification tool below ``VERTEX_ENUM_MAX_*`` only. The bound is a
sound enclosure, not a complexity claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from omnibias.convex.certify import lp_dual_lower_bound

FloatArray = NDArray[np.float64]

VERTEX_ENUM_MAX_D = 4
VERTEX_ENUM_MAX_N = 16
_VERT_ATOL = 1e-9
_ACTIVE_ATOL = 1e-6
_SIGMOID_CLIP = 60.0


class DiffMode(str, Enum):
    """Which differentiable route produced the gradient."""

    KKT = "kkt"
    SOFT = "soft"


@dataclass(frozen=True)
class LearnedPolytope:
    """Inequality polytope ``A x <= b``. Facets are arrangement hyperplanes."""

    normals: FloatArray
    offsets: FloatArray

    def __post_init__(self) -> None:
        a = np.asarray(self.normals, dtype=np.float64)
        b = np.asarray(self.offsets, dtype=np.float64).reshape(-1)
        if a.ndim != 2:
            raise ValueError("normals must have shape (n, D)")
        if b.shape[0] != a.shape[0]:
            raise ValueError("offsets must have length n")
        object.__setattr__(self, "normals", a)
        object.__setattr__(self, "offsets", b)

    @property
    def n(self) -> int:
        return int(self.normals.shape[0])

    @property
    def dim(self) -> int:
        return int(self.normals.shape[1])

    def as_arrangement(self) -> object:
        """The same hyperplanes as an :class:`~omnibias.partition.arrangement.Arrangement`."""
        from omnibias.partition.arrangement import Arrangement

        return Arrangement(self.normals, self.offsets)

    def residual(self, x: object) -> FloatArray:
        xv = np.asarray(x, dtype=np.float64)
        if xv.ndim == 1:
            return self.normals @ xv - self.offsets
        return xv @ self.normals.T - self.offsets[None, :]


@dataclass(frozen=True)
class LPOutput:
    """Primal point, sound lower bound, and the flags the gradient needs."""

    x: FloatArray
    value: float
    lower_bound: float
    active_set: tuple[int, ...]
    degenerate: bool
    mode: DiffMode
    feasible: bool
    n_vertices: int = 0


def honesty_payload() -> dict[str, bool]:
    return {
        "new_lp_algorithm": False,
        "uncorrected_float_dual": False,
        "p_equals_np_claim": False,
        "vertex_enum_is_exponential": True,
        "selection_is_temperature_collapse": True,
        "founding_bias_collapse_in_selection": False,
    }


def _sigmoid(z: object) -> FloatArray:
    zv = np.clip(np.asarray(z, dtype=np.float64), -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-zv))


def soft_membership(poly: LearnedPolytope, x: object, *, beta: float) -> FloatArray:
    """Soft feasible-cell weight ``prod_i sigma(beta (b_i - a_i · x))``."""
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0 (temperature collapse axis)")
    slack = -poly.residual(x)
    w = _sigmoid(float(beta) * slack)
    if w.ndim == 1:
        return np.asarray(np.prod(w), dtype=np.float64)
    return np.prod(w, axis=-1).astype(np.float64, copy=False)


def soft_cell_gap_bound(*, n_facets: int, beta: float) -> float:
    """``log(n)/beta``, the same Gibbs scale ``certify_partition_gap`` reports."""
    if int(n_facets) < 1:
        raise ValueError("n_facets must be >= 1")
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    return math.log(int(n_facets)) / float(beta)


def named_pentagon() -> tuple[LearnedPolytope, FloatArray, float]:
    """Spec §5: ``min -x-y`` over the pentagon, optimum value ``-3``."""
    a = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float64,
    )
    b = np.array([2.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float64)
    c = np.array([-1.0, -1.0], dtype=np.float64)
    return LearnedPolytope(a, b), c, -3.0


def known_dual_pentagon() -> FloatArray:
    """``y = (0,0,1,0,0)``: ``A^T y = -c`` and ``-b·y = -3``."""
    return np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def enumerate_vertices(poly: LearnedPolytope, *, atol: float = _VERT_ATOL) -> FloatArray:
    """Vertices of ``A x <= b``. Exponential; raises above the published cutoff."""
    d = poly.dim
    n = poly.n
    if d > VERTEX_ENUM_MAX_D or n > VERTEX_ENUM_MAX_N:
        raise ValueError(
            f"vertex enumeration is a small-instance tool only "
            f"(D<={VERTEX_ENUM_MAX_D}, n<={VERTEX_ENUM_MAX_N}); got D={d}, n={n}"
        )
    if d < 1 or n < d:
        return np.zeros((0, d), dtype=np.float64)
    verts: list[FloatArray] = []
    seen: list[FloatArray] = []
    a = poly.normals
    b = poly.offsets
    for idx in combinations(range(n), d):
        ai = a[list(idx)]
        if abs(float(np.linalg.det(ai))) < 1e-12:
            continue
        x = np.linalg.solve(ai, b[list(idx)])
        if not np.all(a @ x <= b + atol):
            continue
        if any(np.allclose(x, s, atol=1e-8) for s in seen):
            continue
        seen.append(x)
        verts.append(x)
    if not verts:
        return np.zeros((0, d), dtype=np.float64)
    return np.stack(verts, axis=0)


def vertex_optimum(poly: LearnedPolytope, c: object) -> tuple[FloatArray, float, FloatArray]:
    """Exact min of ``c·x`` over enumerated vertices. Empty => infeasible/unbounded."""
    verts = enumerate_vertices(poly)
    cv = np.asarray(c, dtype=np.float64).reshape(-1)
    if verts.shape[0] == 0:
        return np.zeros(poly.dim, dtype=np.float64), float("inf"), verts
    vals = verts @ cv
    k = int(np.argmin(vals))
    return verts[k].copy(), float(vals[k]), verts


def is_degenerate_optimum(verts: FloatArray, c: object, value: float, *, atol: float = 1e-8) -> bool:
    """True when two or more vertices share the optimal value (an optimal edge/face)."""
    if verts.shape[0] < 2:
        return False
    cv = np.asarray(c, dtype=np.float64).reshape(-1)
    return int(np.sum(np.abs(verts @ cv - value) <= atol)) >= 2


def active_set(poly: LearnedPolytope, x: object, *, atol: float = _ACTIVE_ATOL) -> tuple[int, ...]:
    slack = -poly.residual(x)
    return tuple(int(i) for i in np.nonzero(slack <= atol)[0])


def infer_box(poly: LearnedPolytope, *, fallback: float = 10.0) -> tuple[FloatArray, FloatArray]:
    """Axis-aligned bounds implied by singleton rows, else ``[-fallback, fallback]``."""
    lo = np.full(poly.dim, -np.inf, dtype=np.float64)
    hi = np.full(poly.dim, np.inf, dtype=np.float64)
    for i in range(poly.n):
        row = poly.normals[i]
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        if nz.size != 1:
            continue
        j = int(nz[0])
        a = float(row[j])
        rhs = float(poly.offsets[i]) / a
        if a > 0.0:
            hi[j] = min(hi[j], rhs)
        else:
            lo[j] = max(lo[j], rhs)
    lo = np.where(np.isfinite(lo), lo, -float(fallback))
    hi = np.where(np.isfinite(hi), hi, float(fallback))
    return lo.astype(np.float64), hi.astype(np.float64)


def sound_lower_bound(
    poly: LearnedPolytope,
    c: object,
    dual: object,
    *,
    x_lower: object | None = None,
    x_upper: object | None = None,
) -> float:
    """Neumaier-Shcherbina ``lo`` only. No uncorrected float dual is offered."""
    if x_lower is None or x_upper is None:
        x_lower, x_upper = infer_box(poly)
    iv = lp_dual_lower_bound(
        np.asarray(c, dtype=float),
        poly.normals,
        poly.offsets,
        dual,
        x_lower=x_lower,
        x_upper=x_upper,
    )
    return float(iv.lo)


def recover_vertex(poly: LearnedPolytope, x: object) -> FloatArray:
    """Active-set crossover: solve the ``D`` tightest residuals for a vertex."""
    d = poly.dim
    res = np.abs(poly.residual(x))
    order = np.argsort(res)
    for extra in range(0, min(4, poly.n - d + 1)):
        idx = order[: d + extra]
        for combo in combinations(idx.tolist(), d):
            ai = poly.normals[list(combo)]
            if abs(float(np.linalg.det(ai))) < 1e-10:
                continue
            xv = np.linalg.solve(ai, poly.offsets[list(combo)])
            if np.all(poly.normals @ xv <= poly.offsets + 1e-6):
                return xv.astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def softmax_neg(values: FloatArray, *, beta: float) -> FloatArray:
    z = -float(beta) * np.asarray(values, dtype=np.float64)
    z = z - float(np.max(z))
    w = np.exp(z)
    return (w / float(np.sum(w))).astype(np.float64, copy=False)


def soft_vertex_solution(
    verts: FloatArray, c: object, *, beta: float
) -> tuple[FloatArray, FloatArray, float]:
    """Softmin over vertices. Finite on a degenerate optimal face."""
    cv = np.asarray(c, dtype=np.float64).reshape(-1)
    vals = verts @ cv
    w = softmax_neg(vals, beta=beta)
    x = (w @ verts).astype(np.float64, copy=False)
    return x, w, float(w @ vals)


def soft_vertex_dx_dc(verts: FloatArray, c: object, *, beta: float) -> FloatArray:
    """Jacobian ``dx/dc`` of the softmax vertex mixture. Finite when KKT is not."""
    cv = np.asarray(c, dtype=np.float64).reshape(-1)
    vals = verts @ cv
    w = softmax_neg(vals, beta=beta)
    x = w @ verts
    # d w_i / d c = -beta w_i (v_i - x)
    # dx/dc = sum_i v_i (dw_i/dc)^T = -beta sum_i w_i (v_i - x) v_i^T
    centered = verts - x[None, :]
    jac = -float(beta) * ((w[:, None] * centered).T @ verts)
    return jac.astype(np.float64, copy=False)


def kkt_matrix(poly: LearnedPolytope, x: object, dual: object) -> FloatArray:
    """The LP KKT Jacobian at ``(x, lambda)``. Singular on a degenerate face."""
    a = poly.normals
    slack = -poly.residual(x)
    lam = np.asarray(dual, dtype=np.float64).reshape(-1)
    d = poly.dim
    q = np.zeros((d, d), dtype=np.float64)
    top = np.concatenate([q, a.T], axis=1)
    bottom = np.concatenate([lam[:, None] * a, np.diag(-slack)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def kkt_gradient_usable(kkt: FloatArray) -> bool:
    try:
        cond = float(np.linalg.cond(kkt))
    except np.linalg.LinAlgError:
        return False
    return bool(np.isfinite(cond) and cond < 1e10)


def random_feasible_lp(
    dim: int,
    n_extra: int,
    rng: Generator,
    *,
    unique_c: bool = True,
) -> tuple[LearnedPolytope, FloatArray]:
    """Bounded feasible LP: random halfspaces plus a box. ``0`` is strictly feasible."""
    extra = max(int(n_extra), 0)
    g = rng.normal(size=(extra, dim))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    g = g / np.maximum(norms, 1e-12)
    b_extra = rng.uniform(0.8, 2.5, size=extra)
    box_a = np.vstack([np.eye(dim), -np.eye(dim)])
    box_b = np.concatenate([np.full(dim, 2.0), np.zeros(dim)])
    if extra:
        a = np.vstack([g, box_a])
        b = np.concatenate([b_extra, box_b])
    else:
        a = box_a
        b = box_b
    c = rng.normal(size=dim)
    if unique_c:
        c = c + 0.15 * np.arange(1, dim + 1, dtype=np.float64)
    return LearnedPolytope(a.astype(np.float64), b.astype(np.float64)), c.astype(np.float64)


def solve_arrangement_lp(
    poly: LearnedPolytope,
    c: object,
    *,
    mode: DiffMode = DiffMode.KKT,
    beta: float = 8.0,
    dual: object | None = None,
) -> LPOutput:
    """Exact vertex solve (small instances) plus a sound NS lower bound."""
    cv = np.asarray(c, dtype=np.float64).reshape(-1)
    x_star, value, verts = vertex_optimum(poly, cv)
    feasible = verts.shape[0] > 0 and math.isfinite(value)
    if not feasible:
        lo, hi = infer_box(poly)
        if np.any(lo > hi):
            return LPOutput(
                x=np.full(poly.dim, np.nan),
                value=float("inf"),
                lower_bound=float("-inf"),
                active_set=(),
                degenerate=False,
                mode=mode,
                feasible=False,
                n_vertices=0,
            )
        return LPOutput(
            x=np.full(poly.dim, np.nan),
            value=float("inf"),
            lower_bound=sound_lower_bound(poly, cv, np.zeros(poly.n), x_lower=lo, x_upper=hi),
            active_set=(),
            degenerate=False,
            mode=mode,
            feasible=False,
            n_vertices=0,
        )
    degenerate = is_degenerate_optimum(verts, cv, value)
    if mode is DiffMode.SOFT:
        x_out, _w, value_soft = soft_vertex_solution(verts, cv, beta=beta)
        value_rep = value_soft
        x_rep = x_out
    else:
        x_rep = x_star
        value_rep = value
    y = np.asarray(dual, dtype=np.float64).reshape(-1) if dual is not None else _vertex_dual(poly, x_star, cv)
    lower = sound_lower_bound(poly, cv, y)
    return LPOutput(
        x=x_rep,
        value=float(value_rep),
        lower_bound=lower,
        active_set=active_set(poly, x_star),
        degenerate=degenerate,
        mode=mode,
        feasible=True,
        n_vertices=int(verts.shape[0]),
    )


def _vertex_dual(poly: LearnedPolytope, x: FloatArray, c: FloatArray) -> FloatArray:
    """A complementary dual at a vertex (least-squares on the active rows)."""
    act = list(active_set(poly, x, atol=1e-7))
    y = np.zeros(poly.n, dtype=np.float64)
    if not act:
        return y
    ai = poly.normals[act]
    # A_I^T y_I = -c, y_I >= 0.
    y_act, *_ = np.linalg.lstsq(ai.T, -c, rcond=None)
    y_act = np.maximum(y_act, 0.0)
    y[act] = y_act
    return y


def duality_holds(poly: LearnedPolytope, c: object, y: object, *, atol: float = 1e-9) -> bool:
    """Stationarity ``A^T y + c = 0`` and ``y >= 0`` (the G5 convention)."""
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    cv = np.asarray(c, dtype=np.float64).reshape(-1)
    if np.any(yv < -atol):
        return False
    return bool(np.allclose(poly.normals.T @ yv + cv, 0.0, atol=atol))


def dual_objective(poly: LearnedPolytope, y: object) -> float:
    """``-b · y`` for ``min c·x`` / ``A x <= b``. Matches primal at a tight pair."""
    return float(-poly.offsets @ np.asarray(y, dtype=np.float64).reshape(-1))


def predict_then_optimize_regret(
    true_c: FloatArray,
    x_hat: FloatArray,
    x_star: FloatArray,
) -> float:
    """Decision regret ``true_c · x_hat - true_c · x_star`` (minimization)."""
    tc = np.asarray(true_c, dtype=np.float64).reshape(-1)
    return float(tc @ np.asarray(x_hat, dtype=np.float64) - tc @ np.asarray(x_star, dtype=np.float64))


def knapsack_simplex() -> LearnedPolytope:
    """``x1 + x2 <= 1``, ``x >= 0`` -- the G4 decision polytope."""
    a = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return LearnedPolytope(a, b)


def two_stage_constant_predict(train_c: FloatArray) -> FloatArray:
    """Feature-blind two-stage baseline: predict the training-mean cost."""
    return np.mean(np.asarray(train_c, dtype=np.float64), axis=0)


__all__ = [
    "DiffMode",
    "LPOutput",
    "LearnedPolytope",
    "VERTEX_ENUM_MAX_D",
    "VERTEX_ENUM_MAX_N",
    "active_set",
    "dual_objective",
    "duality_holds",
    "enumerate_vertices",
    "honesty_payload",
    "infer_box",
    "is_degenerate_optimum",
    "kkt_gradient_usable",
    "kkt_matrix",
    "knapsack_simplex",
    "known_dual_pentagon",
    "named_pentagon",
    "predict_then_optimize_regret",
    "random_feasible_lp",
    "recover_vertex",
    "soft_cell_gap_bound",
    "soft_membership",
    "soft_vertex_dx_dc",
    "soft_vertex_solution",
    "softmax_neg",
    "solve_arrangement_lp",
    "sound_lower_bound",
    "two_stage_constant_predict",
    "vertex_optimum",
]
