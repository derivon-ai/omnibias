# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tropical-log homotopy (theory 01-08).

``logsumexp_beta`` interpolates the log semiring and max-plus. The gap is
``log(n)/beta`` from :func:`logsumexp_gap_bound` -- reuse, do not fork.

``beta -> inf`` is **temperature collapse**, not founding ``delta -> 0``.
The gap is sound, not P vs NP. Sampling the tie locus is a lower bound.
Large ``n, D`` are refused (subdivision is exponential in ``D``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.gap import logsumexp_gap_bound

FloatArray = NDArray[np.float64]

_MAX_N = 10
_MAX_D = 3


def _check_size(n: int, dim: int) -> None:
    if n > _MAX_N or dim > _MAX_D:
        raise ValueError(
            f"tropical combinatorics refuse n>{_MAX_N} or D>{_MAX_D} "
            f"(got n={n}, D={dim}); subdivision is exponential in D"
        )


@dataclass(frozen=True)
class TropicalLinear:
    """Tropical polynomial ``max_i (a_i + m_i . x)``.

    Terminology: ``beta -> inf`` hardens softmax into argmax (temperature /
    feasibility collapse). Do not conflate with founding bias collapse
    (``delta -> 0``).
    """

    coeffs: FloatArray
    exponents: FloatArray

    def __post_init__(self) -> None:
        a = np.asarray(self.coeffs, dtype=np.float64).reshape(-1)
        m = np.asarray(self.exponents, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != a.shape[0]:
            raise ValueError("exponents must have shape (n, D)")
        _check_size(int(a.shape[0]), int(m.shape[1]))
        object.__setattr__(self, "coeffs", a)
        object.__setattr__(self, "exponents", m)

    @property
    def n(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def dim(self) -> int:
        return int(self.exponents.shape[1])


def scores(poly: TropicalLinear, x: FloatArray) -> FloatArray:
    """``a_i + m_i . x``, shape ``(..., n)``."""
    xv = np.asarray(x, dtype=np.float64)
    if xv.ndim == 1:
        xv = xv.reshape(1, -1)
    return poly.coeffs[None, :] + xv @ poly.exponents.T


def tropical_value(poly: TropicalLinear, x: FloatArray) -> FloatArray:
    return np.max(scores(poly, x), axis=-1)


def relaxed_value(poly: TropicalLinear, x: FloatArray, *, beta: float) -> FloatArray:
    """``(1/beta) log sum_i exp(beta (a_i + m_i x))``."""
    if beta <= 0.0:
        raise ValueError("beta must be > 0 (temperature collapse axis)")
    s = scores(poly, x)
    scaled = beta * s
    m = np.max(scaled, axis=-1, keepdims=True)
    return (m.squeeze(-1) + np.log(np.exp(scaled - m).sum(axis=-1))) / beta


def relaxed_weights(poly: TropicalLinear, x: FloatArray, *, beta: float) -> FloatArray:
    if beta <= 0.0:
        raise ValueError("beta must be > 0 (temperature collapse axis)")
    s = scores(poly, x)
    scaled = beta * s
    m = np.max(scaled, axis=-1, keepdims=True)
    e = np.exp(scaled - m)
    return e / e.sum(axis=-1, keepdims=True)


def homotopy_gap_bound(poly: TropicalLinear, *, beta: float) -> float:
    """``log(n)/beta``, reusing :func:`logsumexp_gap_bound`."""
    return logsumexp_gap_bound(poly.n, beta)


def relaxed_grad(poly: TropicalLinear, x: FloatArray, *, beta: float) -> FloatArray:
    """Closed-form ``d/dx`` of ``relaxed_value``: ``sum_i p_i m_i``."""
    p = relaxed_weights(poly, x, beta=beta)
    return p @ poly.exponents


def relaxed_hess(poly: TropicalLinear, x: FloatArray, *, beta: float) -> FloatArray:
    """Closed-form Hessian: ``beta (sum p_i m_i m_i^T - mu mu^T)``."""
    p = relaxed_weights(poly, x, beta=beta)
    xv = np.asarray(x, dtype=np.float64)
    single = xv.ndim == 1
    if single:
        p = p.reshape(1, -1)
    batch = p.shape[0]
    d = poly.dim
    out = np.zeros((batch, d, d))
    m = poly.exponents
    for b in range(batch):
        mu = p[b] @ m
        acc = np.zeros((d, d))
        for i in range(poly.n):
            acc += p[b, i] * np.outer(m[i], m[i])
        out[b] = beta * (acc - np.outer(mu, mu))
    return out[0] if single else out


def newton_polytope(poly: TropicalLinear) -> tuple[tuple[float, ...], ...]:
    """Vertices of the convex hull of the exponent vectors."""
    pts = [tuple(float(v) for v in row) for row in poly.exponents]
    unique = list(dict.fromkeys(pts))
    if len(unique) <= 1 or poly.dim == 1:
        return tuple(unique)
    if poly.dim == 2:
        return _convex_hull_2d(unique)
    return _extreme_points(unique)


def _convex_hull_2d(pts: list[tuple[float, ...]]) -> tuple[tuple[float, ...], ...]:
    pts = sorted(pts)
    def cross(o: tuple[float, ...], a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, ...]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, ...]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


def _extreme_points(pts: list[tuple[float, ...]]) -> tuple[tuple[float, ...], ...]:
    arr = np.asarray(pts, dtype=np.float64)
    extreme: list[tuple[float, ...]] = []
    rng = np.random.default_rng(0)
    for _ in range(64 * len(pts)):
        d = rng.normal(size=arr.shape[1])
        i = int(np.argmax(arr @ d))
        extreme.append(tuple(float(v) for v in arr[i]))
    return tuple(dict.fromkeys(extreme))


def dual_subdivision(
    poly: TropicalLinear, *, n_samples: int = 4000, seed: int = 0
) -> tuple[tuple[int, ...], ...]:
    """Full-dimensional cells (unique argmax index) found by sampling.

    A lower bound on the dual subdivision, never a completeness claim.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, poly.dim))
    s = scores(poly, x)
    winners = np.argmax(s, axis=-1)
    top2 = np.partition(s, -2, axis=-1)[:, -2:]
    unique = top2[:, 1] - top2[:, 0] > 1e-10
    cells = tuple(sorted({(int(w),) for w, ok in zip(winners, unique, strict=True) if ok}))
    return cells


def tie_locus_samples(
    poly: TropicalLinear, box: float = 2.0, *, n: int = 32, seed: int = 0
) -> FloatArray:
    """Points where the argmax is (numerically) attained twice. Samples, not the whole locus."""
    rng = np.random.default_rng(seed)
    out: list[list[float]] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        x = rng.uniform(-box, box, size=poly.dim)
        s = scores(poly, x).reshape(-1)
        order = np.argsort(s)
        if abs(s[order[-1]] - s[order[-2]]) < 1e-3:
            # Newton on the two-term tie: (m_i - m_j)·x = a_j - a_i
            i, j = int(order[-1]), int(order[-2])
            diff = poly.exponents[i] - poly.exponents[j]
            rhs = poly.coeffs[j] - poly.coeffs[i]
            nrm = float(np.dot(diff, diff))
            if nrm < 1e-18:
                continue
            x = x - diff * ((np.dot(diff, x) - rhs) / nrm)
            out.append([float(v) for v in x])
    if not out:
        return np.zeros((0, poly.dim), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class TropicalGapCertificate:
    beta: float
    bound: float
    measured: float

    @property
    def is_sound(self) -> bool:
        return self.bound + 1e-12 >= self.measured


def certify_tropical_gap(
    poly: TropicalLinear, x: FloatArray, *, beta: float
) -> TropicalGapCertificate:
    """Sound ``log(n)/beta`` bound on ``relaxed - tropical``."""
    rel = relaxed_value(poly, x, beta=beta)
    hard = tropical_value(poly, x)
    measured = float(np.max(rel - hard))
    bound = homotopy_gap_bound(poly, beta=beta)
    return TropicalGapCertificate(beta=beta, bound=bound, measured=measured)


@dataclass(frozen=True)
class TropicalSchedule:
    """Temperature homotopy; duck-compatible with ``AnnealSchedule``.

    ``omnibias-struct`` cannot import ``omnibias-discrete`` (discrete already
    depends on struct). Pass an ``AnnealSchedule`` anyway: ``from_anneal``
    and :func:`as_tropical_schedule` read the same field names and
    ``betas()``.
    """

    beta0: float = 0.5
    beta_growth: float = 1.6
    stages: int = 8
    steps: int = 24
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
        out, beta = [], self.beta0
        for _ in range(self.stages):
            out.append(beta)
            beta *= self.beta_growth
        return out

    @classmethod
    def from_anneal(cls, schedule: Any) -> TropicalSchedule:
        """Wire ``AnnealSchedule`` (or any duck-typed schedule) into this driver."""
        return cls(
            beta0=float(schedule.beta0),
            beta_growth=float(schedule.beta_growth),
            stages=int(schedule.stages),
            steps=int(schedule.steps),
            step_safety=float(schedule.step_safety),
        )


def as_tropical_schedule(schedule: Any | None = None) -> TropicalSchedule:
    """Accept ``TropicalSchedule``, ``AnnealSchedule``, or ``None`` (defaults)."""
    if schedule is None:
        return TropicalSchedule()
    if isinstance(schedule, TropicalSchedule):
        return schedule
    return TropicalSchedule.from_anneal(schedule)


@dataclass(frozen=True)
class TropicalPathResult:
    """Decoded tropical objective after a temperature homotopy."""

    x: FloatArray
    decoded: float
    winner: int
    n_evals: int
    gap: TropicalGapCertificate
    method: str


def surrounding_tropical(n: int = 6, dim: int = 2, *, seed: int = 0) -> TropicalLinear:
    """PWL family whose exponent hull contains 0 (unique unconstrained min).

    Directions surround the origin so ``tropical_value`` has a unique
    unconstrained minimizer; that is the G4 comparison family.
    """
    if n < dim + 1:
        raise ValueError("need n >= dim+1 so conv(exponents) can contain 0")
    rng = np.random.default_rng(seed)
    if dim == 1:
        signs = np.ones((n, 1), dtype=np.float64)
        signs[1::2, 0] = -1.0
        exponents = signs
    elif dim == 2:
        jitter = rng.uniform(-0.1, 0.1, size=n)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + jitter
        exponents = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    else:
        raw = rng.normal(size=(n, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        exponents = raw / np.maximum(norms, 1e-12)
    coeffs = rng.normal(size=n) * 0.25
    return TropicalLinear(coeffs, exponents)


def _clip_box(x: FloatArray, box: float) -> FloatArray:
    if box <= 0.0:
        raise ValueError("box must be > 0")
    return np.clip(np.asarray(x, dtype=np.float64), -float(box), float(box))


def _lipschitz(poly: TropicalLinear) -> float:
    return float(np.max(np.linalg.norm(poly.exponents, axis=1))) + 1e-12


def _decode(
    poly: TropicalLinear,
    x: FloatArray,
    *,
    beta: float,
    n_evals: int,
    method: str,
) -> TropicalPathResult:
    xv = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    decoded = float(np.asarray(tropical_value(poly, xv)).reshape(-1)[0])
    winner = int(np.argmax(scores(poly, xv).reshape(-1)))
    gap = certify_tropical_gap(poly, xv, beta=beta)
    return TropicalPathResult(
        x=xv,
        decoded=decoded,
        winner=winner,
        n_evals=int(n_evals),
        gap=gap,
        method=method,
    )


def tropical_anneal_descent(
    poly: TropicalLinear,
    x0: FloatArray,
    *,
    box: float = 2.0,
    schedule: Any | None = None,
) -> TropicalPathResult:
    """First-order temperature homotopy; mirrors ``anneal_descent``.

    Uses :class:`TropicalSchedule` or a duck-typed ``AnnealSchedule``.
    Each gradient step counts as one objective evaluation.
    """
    sched = as_tropical_schedule(schedule)
    x = _clip_box(np.asarray(x0, dtype=np.float64).reshape(-1), box)
    if int(x.shape[0]) != poly.dim:
        raise ValueError(f"x0 dim {int(x.shape[0])} != poly.dim {poly.dim}")
    lip = _lipschitz(poly)
    n_evals = 0
    betas = sched.betas()
    for beta in betas:
        eta = sched.step_safety / (beta * lip + 1e-12)
        for _ in range(sched.steps):
            grad = np.asarray(relaxed_grad(poly, x, beta=beta)).reshape(-1)
            x = _clip_box(x - eta * grad, box)
            n_evals += 1
    return _decode(poly, x, beta=betas[-1], n_evals=n_evals, method="anneal")


def _newton_stage(
    poly: TropicalLinear,
    x: FloatArray,
    beta: float,
    *,
    box: float,
    lip: float,
    safety: float,
    steps: int,
) -> tuple[FloatArray, int]:
    n_evals = 0
    dim = poly.dim
    eye = np.eye(dim)
    for _ in range(steps):
        grad = np.asarray(relaxed_grad(poly, x, beta=beta)).reshape(-1)
        hess = np.asarray(relaxed_hess(poly, x, beta=beta)).reshape(dim, dim)
        n_evals += 1
        eta_g = safety / (beta * lip + 1e-12)
        scale = float(np.linalg.norm(hess, ord="fro"))
        if scale < 1e-12:
            x = _clip_box(x - eta_g * grad, box)
            continue
        try:
            step = np.linalg.solve(hess + 1e-8 * eye, grad)
        except np.linalg.LinAlgError:
            step = eta_g * grad
        cand = _clip_box(x - safety * step, box)
        f0 = float(np.asarray(relaxed_value(poly, x, beta=beta)).reshape(-1)[0])
        f1 = float(np.asarray(relaxed_value(poly, cand, beta=beta)).reshape(-1)[0])
        n_evals += 1
        x = cand if f1 <= f0 else _clip_box(x - eta_g * grad, box)
    return x, n_evals


def path_follow(
    poly: TropicalLinear,
    x0: FloatArray,
    *,
    box: float = 2.0,
    schedule: Any | None = None,
    newton_steps: int = 2,
    polish_steps: int = 4,
) -> TropicalPathResult:
    """Second-order path-follow along the same ``AnnealSchedule`` homotopy.

    Each Newton step is one closed-form Hessian/grad pass plus one
    accept/reject value eval. A short polish at the final ``beta``
    matches the first-order decode.
    """
    if newton_steps < 1:
        raise ValueError("newton_steps must be >= 1")
    if polish_steps < 0:
        raise ValueError("polish_steps must be >= 0")
    sched = as_tropical_schedule(schedule)
    x = _clip_box(np.asarray(x0, dtype=np.float64).reshape(-1), box)
    if int(x.shape[0]) != poly.dim:
        raise ValueError(f"x0 dim {int(x.shape[0])} != poly.dim {poly.dim}")
    lip = _lipschitz(poly)
    n_evals = 0
    betas = sched.betas()
    for beta in betas:
        x, stage_evals = _newton_stage(
            poly,
            x,
            beta,
            box=box,
            lip=lip,
            safety=sched.step_safety,
            steps=newton_steps,
        )
        n_evals += stage_evals
    if polish_steps:
        x, polish_evals = _newton_stage(
            poly,
            x,
            betas[-1],
            box=box,
            lip=lip,
            safety=sched.step_safety,
            steps=polish_steps,
        )
        n_evals += polish_evals
    return _decode(poly, x, beta=betas[-1], n_evals=n_evals, method="path_follow")


__all__ = [
    "TropicalGapCertificate",
    "TropicalLinear",
    "TropicalPathResult",
    "TropicalSchedule",
    "as_tropical_schedule",
    "certify_tropical_gap",
    "dual_subdivision",
    "homotopy_gap_bound",
    "newton_polytope",
    "path_follow",
    "relaxed_grad",
    "relaxed_hess",
    "relaxed_value",
    "relaxed_weights",
    "scores",
    "surrounding_tropical",
    "tie_locus_samples",
    "tropical_anneal_descent",
    "tropical_value",
]
