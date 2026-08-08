# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Signed-distance primitives and R-function CSG (pure numpy).

An SDF returns negative values in the interior, zero on the boundary, and
positive outside (the graphics convention). R-function compositions
(Rvachev) keep the zero set algebraic and :math:`C^k`-smooth, unlike
``min`` / ``max`` CSG which introduce kinks.

References
----------
Rvachev, *Theory of R-functions and Some Applications* (1982).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

FloatArray = np.ndarray
SDFFn = Callable[[FloatArray], FloatArray]


class SDF(Protocol):
    """Callable signed-distance (or approximate-distance) function."""

    def __call__(self, X: FloatArray) -> FloatArray: ...

    @property
    def ndim(self) -> int: ...


@dataclass(frozen=True)
class Halfspace:
    """``n · (x - p)`` -- positive on the ``n`` side of the plane through ``p``."""

    normal: tuple[float, ...]
    point: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.normal) != len(self.point):
            raise ValueError("normal and point must have the same length")
        n = np.asarray(self.normal, dtype=float)
        if float(np.linalg.norm(n)) <= 0.0:
            raise ValueError("normal must be non-zero")

    @property
    def ndim(self) -> int:
        return len(self.normal)

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = np.asarray(self.normal, dtype=float)
        n = n / np.linalg.norm(n)
        p = np.asarray(self.point, dtype=float)
        return cast(FloatArray, (X - p) @ n)


@dataclass(frozen=True)
class Sphere:
    """``||x - c|| - r`` -- negative inside the ball."""

    center: tuple[float, ...]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {self.radius}")

    @property
    def ndim(self) -> int:
        return len(self.center)

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        c = np.asarray(self.center, dtype=float)
        return cast(
            FloatArray, np.linalg.norm(X - c, axis=-1) - float(self.radius)
        )


@dataclass(frozen=True)
class Box:
    """Axis-aligned box SDF (Inigo Quilez form); negative inside."""

    lo: tuple[float, ...]
    hi: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lo) != len(self.hi):
            raise ValueError("lo and hi must have the same length")
        for a, b in zip(self.lo, self.hi, strict=True):
            if not b > a:
                raise ValueError(f"need hi > lo on every axis; got {(a, b)}")

    @property
    def ndim(self) -> int:
        return len(self.lo)

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        q = np.abs(X - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return cast(FloatArray, outside + inside)


@dataclass(frozen=True)
class Cylinder:
    """Infinite cylinder along ``axis`` (0/1/2); negative inside."""

    center: tuple[float, float]
    radius: float
    axis: int = 2
    ambient_dim: int = 3

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        if not 0 <= self.axis < self.ambient_dim:
            raise ValueError(f"axis {self.axis} out of range for dim {self.ambient_dim}")
        if len(self.center) != self.ambient_dim - 1:
            raise ValueError(
                f"center must have length ambient_dim-1={self.ambient_dim - 1}"
            )

    @property
    def ndim(self) -> int:
        return self.ambient_dim

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        axes = [i for i in range(self.ambient_dim) if i != self.axis]
        c = np.asarray(self.center, dtype=float)
        return cast(
            FloatArray,
            np.linalg.norm(X[:, axes] - c, axis=-1) - float(self.radius),
        )


@dataclass(frozen=True)
class Polygon:
    """2-D polygon SDF via winding + edge distance; negative inside."""

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("polygon needs at least 3 vertices")

    @property
    def ndim(self) -> int:
        return 2

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[-1] != 2:
            raise ValueError(f"Polygon SDF expects 2-D points; got shape {X.shape}")
        V = np.asarray(self.vertices, dtype=float)
        n = V.shape[0]
        # Edge distance (unsigned) + winding sign.
        d = np.full(X.shape[0], np.inf)
        winding = np.zeros(X.shape[0], dtype=float)
        for i in range(n):
            a = V[i]
            b = V[(i + 1) % n]
            pa = X - a
            ba = b - a
            h = np.clip((pa @ ba) / (ba @ ba + 1e-30), 0.0, 1.0)
            dist = np.linalg.norm(pa - h[:, None] * ba, axis=-1)
            d = np.minimum(d, dist)
            # Winding contribution (Hormann-Agathos style sign flip on crossings).
            cond = ((a[1] > X[:, 1]) != (b[1] > X[:, 1])) & (
                X[:, 0]
                < (b[0] - a[0]) * (X[:, 1] - a[1]) / (b[1] - a[1] + 1e-30) + a[0]
            )
            winding = np.where(cond, 1.0 - winding, winding)
        sign = np.where(winding > 0.5, -1.0, 1.0)
        return cast(FloatArray, sign * d)


def r_conjunction(a: FloatArray, b: FloatArray, *, alpha: float = 0.0) -> FloatArray:
    r"""R-function conjunction (intersection): zero when either factor is zero.

    .. math::

        a \wedge_\alpha b = \frac{1}{1+\alpha}\Bigl(
            a + b - \sqrt{a^2 + b^2 - 2\alpha a b}\Bigr)

    With ``alpha=0`` this is the classical R0 conjunction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not -1.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (-1, 1], got {alpha}")
    return cast(
        FloatArray,
        (a + b - np.sqrt(np.maximum(a * a + b * b - 2.0 * alpha * a * b, 0.0)))
        / (1.0 + alpha),
    )


def r_disjunction(a: FloatArray, b: FloatArray, *, alpha: float = 0.0) -> FloatArray:
    r"""R-function disjunction (union)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not -1.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (-1, 1], got {alpha}")
    return cast(
        FloatArray,
        (a + b + np.sqrt(np.maximum(a * a + b * b - 2.0 * alpha * a * b, 0.0)))
        / (1.0 + alpha),
    )


def r_negation(a: FloatArray) -> FloatArray:
    """R-function complement: ``-a``."""
    return np.asarray(-np.asarray(a, dtype=float), dtype=float)


@dataclass(frozen=True)
class RCompose:
    """Binary R-function composition of two SDFs."""

    left: SDF
    right: SDF
    op: str = "and"  # "and" | "or"
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.op not in ("and", "or"):
            raise ValueError(f"op must be 'and' or 'or', got {self.op!r}")
        if self.left.ndim != self.right.ndim:
            raise ValueError(
                f"SDF ndim mismatch: {self.left.ndim} vs {self.right.ndim}"
            )

    @property
    def ndim(self) -> int:
        return self.left.ndim

    def __call__(self, X: FloatArray) -> FloatArray:
        a = self.left(X)
        b = self.right(X)
        if self.op == "and":
            return r_conjunction(a, b, alpha=self.alpha)
        return r_disjunction(a, b, alpha=self.alpha)


@dataclass(frozen=True)
class Negate:
    """Complement of an SDF."""

    child: SDF

    @property
    def ndim(self) -> int:
        return self.child.ndim

    def __call__(self, X: FloatArray) -> FloatArray:
        return r_negation(self.child(X))


def intersect(left: SDF, right: SDF, *, alpha: float = 0.0) -> RCompose:
    return RCompose(left, right, op="and", alpha=alpha)


def union(left: SDF, right: SDF, *, alpha: float = 0.0) -> RCompose:
    return RCompose(left, right, op="or", alpha=alpha)


def complement(child: SDF) -> Negate:
    return Negate(child)


def evaluate_sdf(sdf: SDF, X: FloatArray) -> FloatArray:
    """Evaluate any SDF at ``X`` of shape ``(n, d)`` or ``(d,)``."""
    return np.asarray(sdf(X), dtype=float).reshape(-1)


__all__ = [
    "Box",
    "Cylinder",
    "Halfspace",
    "Negate",
    "Polygon",
    "RCompose",
    "SDF",
    "Sphere",
    "complement",
    "evaluate_sdf",
    "intersect",
    "r_conjunction",
    "r_disjunction",
    "r_negation",
    "union",
]
