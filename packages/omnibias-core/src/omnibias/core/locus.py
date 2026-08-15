# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equality-locus geometry (theory 01-09).

Forcing collapsed units to agree, ``f_i(x) = f_j(x)``, cuts out a
codimension-``m`` constraint manifold whose Jacobian and Hessian the
derivative tower supplies in closed form. Newton projection needs no
autodiff; existence on a box is the Krawczyk test already in
:mod:`omnibias.core.verified.kantorovich`.

The units are founding bias collapse (``delta -> 0``). There is no
temperature collapse here. An equality locus is a **constraint manifold**,
not a PDE solution -- that step is spec 02-12.

Pure Python: no tensor imports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.polynomials import (
    hermite_coeffs,
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import KrawczykCertificate, krawczyk_certificate
from omnibias.core.verified.sigma import sigma_tower_interval

_SUPPORTED_BASES: frozenset[str] = frozenset({"tanh", "sigmoid", "gaussian"})


def _horner(coeffs: Sequence[float], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def sigma_n(base: str, z: float, n: int) -> float:
    """Closed-form ``sigma^(n)(z)`` for the locus bases."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    name = str(base).lower().strip()
    if name == "tanh":
        return _horner(tanh_polynomial_coeffs(n), math.tanh(z))
    if name == "sigmoid":
        return _horner(sigmoid_polynomial_coeffs(n), _sigmoid(z))
    if name == "gaussian":
        g = math.exp(-0.5 * z * z)
        he = _horner(hermite_coeffs(n), z)
        sign = 1.0 if n % 2 == 0 else -1.0
        return sign * he * g
    raise ValueError(
        f"unsupported locus base {base!r}; expected one of {sorted(_SUPPORTED_BASES)}"
    )


def _sigma_n_iv(base: str, z: Interval, n: int) -> Interval:
    tower = sigma_tower_interval(base, z, n)
    return tower[n]


@dataclass(frozen=True)
class UnitTerm:
    """One collapsed unit ``f(x) = c * sigma^(n)(w . x + b)``."""

    order: int
    weight: float
    normal: tuple[float, ...]
    bias: float = 0.0

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if not self.normal:
            raise ValueError("normal must be non-empty")
        if not all(math.isfinite(v) for v in self.normal):
            raise ValueError("normal must be finite")
        if not math.isfinite(self.weight) or not math.isfinite(self.bias):
            raise ValueError("weight and bias must be finite")


@dataclass(frozen=True)
class EqualitySystem:
    """``m + 1`` units yield the residual map ``F : R^D -> R^m``."""

    terms: tuple[UnitTerm, ...]
    base: str = "tanh"

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise ValueError("EqualitySystem needs at least two units")
        dim = len(self.terms[0].normal)
        if any(len(t.normal) != dim for t in self.terms):
            raise ValueError("all unit normals must share a dimension")
        name = str(self.base).lower().strip()
        if name not in _SUPPORTED_BASES:
            raise ValueError(
                f"unsupported locus base {self.base!r}; "
                f"expected one of {sorted(_SUPPORTED_BASES)}"
            )
        object.__setattr__(self, "base", name)

    @property
    def dim(self) -> int:
        return len(self.terms[0].normal)

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def codimension(self) -> int:
        return len(self.terms) - 1


@dataclass(frozen=True)
class AffineSet:
    """Hyperplane ``normal . x + offset = 0`` on an order-matched branch."""

    normal: tuple[float, ...]
    offset: float
    branch: int  # +1 identity z_i = z_j; -1 mirror z_i = -z_j


@dataclass(frozen=True)
class NewtonResult:
    """Gauss-Newton projection onto the locus."""

    point: tuple[float, ...]
    residual_norm: float
    iterations: int
    converged: bool
    condition: float
    branch: tuple[int, ...]
    transversal: bool


def _z(term: UnitTerm, x: Sequence[float]) -> float:
    return sum(w * xi for w, xi in zip(term.normal, x, strict=True)) + term.bias


def _z_iv(term: UnitTerm, xs: Sequence[Interval]) -> Interval:
    acc = Interval.point(term.bias)
    for w, xi in zip(term.normal, xs, strict=True):
        acc = acc + Interval.point(w) * xi
    return acc


def _eval_term(term: UnitTerm, x: Sequence[float], base: str) -> float:
    return term.weight * sigma_n(base, _z(term, x), term.order)


def _grad_term(term: UnitTerm, x: Sequence[float], base: str) -> tuple[float, ...]:
    scale = term.weight * sigma_n(base, _z(term, x), term.order + 1)
    return tuple(scale * w for w in term.normal)


def _hess_term(term: UnitTerm, x: Sequence[float], base: str) -> tuple[tuple[float, ...], ...]:
    scale = term.weight * sigma_n(base, _z(term, x), term.order + 2)
    return tuple(tuple(scale * wi * wj for wj in term.normal) for wi in term.normal)


def residual(sys: EqualitySystem, x: Sequence[float]) -> tuple[float, ...]:
    """``F_k = f_k - f_{k+1}``."""
    if len(x) != sys.dim:
        raise ValueError(f"x has dim {len(x)}, system has dim {sys.dim}")
    vals = [_eval_term(t, x, sys.base) for t in sys.terms]
    return tuple(vals[i] - vals[i + 1] for i in range(sys.codimension))


def jacobian(sys: EqualitySystem, x: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """Closed-form ``DF``, an ``m x D`` matrix. No autodiff."""
    if len(x) != sys.dim:
        raise ValueError(f"x has dim {len(x)}, system has dim {sys.dim}")
    grads = [_grad_term(t, x, sys.base) for t in sys.terms]
    return tuple(
        tuple(grads[i][d] - grads[i + 1][d] for d in range(sys.dim))
        for i in range(sys.codimension)
    )


def hessian_blocks(
    sys: EqualitySystem, x: Sequence[float]
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """``Hess F_k = Hess f_k - Hess f_{k+1}``, one ``D x D`` block per equation."""
    if len(x) != sys.dim:
        raise ValueError(f"x has dim {len(x)}, system has dim {sys.dim}")
    blocks = [_hess_term(t, x, sys.base) for t in sys.terms]
    out: list[tuple[tuple[float, ...], ...]] = []
    for i in range(sys.codimension):
        h0, h1 = blocks[i], blocks[i + 1]
        out.append(
            tuple(
                tuple(h0[r][c] - h1[r][c] for c in range(sys.dim)) for r in range(sys.dim)
            )
        )
    return tuple(out)


def _residual_norm(f: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in f))


def is_transversal(sys: EqualitySystem, x: Sequence[float], *, tol: float = 1e-10) -> bool:
    """Full row-rank of ``DF`` (regular-value test)."""
    return smallest_singular(jacobian(sys, x)) > tol


def smallest_singular(df: Sequence[Sequence[float]]) -> float:
    """Smallest singular value of an ``m x D`` matrix via the Gram eigenvalues."""
    m = len(df)
    if m == 0:
        return 0.0
    gram = [[sum(df[i][k] * df[j][k] for k in range(len(df[0]))) for j in range(m)] for i in range(m)]
    return math.sqrt(max(_smallest_eig_sym(gram), 0.0))


def _smallest_eig_sym(gram: list[list[float]]) -> float:
    n = len(gram)
    if n == 1:
        return gram[0][0]
    if n == 2:
        a, b, c = gram[0][0], gram[0][1], gram[1][1]
        disc = math.sqrt(max((a - c) * (a - c) + 4.0 * b * b, 0.0))
        return 0.5 * (a + c - disc)
    # Power iteration on the inverse is overkill; fall back to Gershgorin lower bound.
    # For Newton we only need a conservative condition number.
    mins: list[float] = []
    for i, row in enumerate(gram):
        radius = sum(abs(row[j]) for j in range(n) if j != i)
        mins.append(row[i] - radius)
    return min(mins)


def branch_signature(sys: EqualitySystem, x: Sequence[float]) -> tuple[int, ...]:
    """Which even/odd branch each consecutive pair sits on.

    For odd order (even ``sigma^(n)``) ``+1`` is the identity ``z_i = z_j``
    and ``-1`` is the mirror ``z_i = -z_j``.
    """
    signs: list[int] = []
    for a, b in zip(sys.terms, sys.terms[1:], strict=False):
        z1, z2 = _z(a, x), _z(b, x)
        if a.order == b.order and a.order % 2 == 1:
            signs.append(1 if abs(z1 - z2) <= abs(z1 + z2) else -1)
        else:
            signs.append(1 if z1 >= z2 else -1)
    return tuple(signs)


def affine_locus(sys: EqualitySystem) -> tuple[AffineSet, ...] | None:
    """Exact hyperplanes when every consecutive pair is order-matched with equal ``c``.

    Returns ``None`` when any pair is order-mismatched (the curved case).
    """
    planes: list[AffineSet] = []
    for a, b in zip(sys.terms, sys.terms[1:], strict=False):
        if a.order != b.order or a.weight != b.weight:
            return None
        ident = AffineSet(
            tuple(wi - wj for wi, wj in zip(a.normal, b.normal, strict=True)),
            a.bias - b.bias,
            1,
        )
        planes.append(ident)
        if a.order % 2 == 1:
            # even sigma^(n): also the mirror z_i = -z_j
            planes.append(
                AffineSet(
                    tuple(wi + wj for wi, wj in zip(a.normal, b.normal, strict=True)),
                    a.bias + b.bias,
                    -1,
                )
            )
    return tuple(planes)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _invert(mat: list[list[float]]) -> list[list[float]]:
    n = len(mat)
    if n == 0:
        return []
    a = [row[:] + [0.0] * n for row in mat]
    for i in range(n):
        a[i][n + i] = 1.0
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[pivot] = a[pivot], a[i]
        diag = a[i][i]
        if abs(diag) < 1e-18:
            raise ZeroDivisionError("singular Gram matrix (locus not transversal)")
        inv = 1.0 / diag
        a[i] = [v * inv for v in a[i]]
        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            a[r] = [a[r][c] - factor * a[i][c] for c in range(2 * n)]
    return [row[n:] for row in a]


def pseudoinverse(df: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Moore-Penrose right inverse ``DF^+ = DF^T (DF DF^T)^{-1}``, shape ``D x m``."""
    m = len(df)
    d = len(df[0])
    gram = [[_dot(df[i], df[j]) for j in range(m)] for i in range(m)]
    ginv = _invert(gram)
    out: list[list[float]] = []
    for k in range(d):
        col_df = [df[i][k] for i in range(m)]
        out.append([_dot(col_df, [ginv[i][j] for i in range(m)]) for j in range(m)])
    return tuple(tuple(row) for row in out)


def newton_project(
    sys: EqualitySystem,
    x0: Sequence[float],
    *,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> NewtonResult:
    """Gauss-Newton projection ``x <- x - DF^+ F``. Minimum-norm correction.

    Differentiating this iterate is **not** the IFT path; tensor twins in
    ``omnibias.fields.locus`` attach the implicit-function theorem on the
    landing point so memory does not scale with ``max_iter``.
    """
    x = tuple(float(v) for v in x0)
    last_f = residual(sys, x)
    it = 0
    cond = 0.0
    transversal = False
    for it in range(1, max_iter + 1):  # noqa: B007 -- `it` is returned as n_iter
        f = residual(sys, x)
        last_f = f
        nrm = _residual_norm(f)
        df = jacobian(sys, x)
        try:
            pinv = pseudoinverse(df)
        except ZeroDivisionError:
            transversal = False
            cond = 0.0
            break
        cond = smallest_singular(df)
        transversal = cond > 1e-10
        step = tuple(_dot(pinv[k], f) for k in range(sys.dim))
        x = tuple(x[k] - step[k] for k in range(sys.dim))
        if nrm <= tol:
            f = residual(sys, x)
            last_f = f
            break
    nrm = _residual_norm(last_f)
    return NewtonResult(
        point=x,
        residual_norm=nrm,
        iterations=it,
        converged=nrm <= tol,
        condition=cond,
        branch=branch_signature(sys, x),
        transversal=transversal,
    )


def locus_tangent(sys: EqualitySystem, x: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """An orthonormal basis of ``ker DF`` (empty when ``m = D`` and transversal)."""
    df = jacobian(sys, x)
    m, d = len(df), sys.dim
    if m >= d:
        return ()
    # Null-space via one Householder-free SVD-free pass: project standard basis.
    pinv = pseudoinverse(df)
    # P = I - DF^+ DF; columns span the tangent (rank D-m).
    proj_cols: list[list[float]] = []
    for ell in range(d):
        df_col = [df[j][ell] for j in range(m)]
        applied = [_dot(pinv[k], df_col) for k in range(d)]
        col = [float(i == ell) - applied[i] for i in range(d)]
        proj_cols.append(col)
    # Gram-Schmidt, drop tiny columns.
    basis: list[tuple[float, ...]] = []
    for col in proj_cols:
        vec = list(col)
        for b in basis:
            s = _dot(vec, b)
            vec = [vec[i] - s * b[i] for i in range(d)]
        nrm = math.sqrt(_dot(vec, vec))
        if nrm <= 1e-12:
            continue
        basis.append(tuple(v / nrm for v in vec))
        if len(basis) >= d - m:
            break
    return tuple(basis)


def residual_iv(sys: EqualitySystem, xs: Sequence[Interval]) -> list[Interval]:
    vals = [
        Interval.point(t.weight) * _sigma_n_iv(sys.base, _z_iv(t, xs), t.order)
        for t in sys.terms
    ]
    return [vals[i] - vals[i + 1] for i in range(sys.codimension)]


def jacobian_iv(sys: EqualitySystem, xs: Sequence[Interval]) -> list[list[Interval]]:
    grads: list[list[Interval]] = []
    for t in sys.terms:
        scale = Interval.point(t.weight) * _sigma_n_iv(sys.base, _z_iv(t, xs), t.order + 1)
        grads.append([scale * Interval.point(w) for w in t.normal])
    return [
        [grads[i][d] - grads[i + 1][d] for d in range(sys.dim)]
        for i in range(sys.codimension)
    ]


def _float_inverse(mat: Sequence[Sequence[float]]) -> list[list[float]]:
    return _invert([list(row) for row in mat])


def certify_locus_point(
    sys: EqualitySystem,
    box: Sequence[tuple[float, float]],
    *,
    free_axes: Sequence[int] | None = None,
) -> KrawczykCertificate | None:
    """Krawczyk unique-zero test on ``box``.

    ``krawczyk_certificate`` is square. When ``codimension < dim`` the test
    runs on ``free_axes`` (default: the last ``m`` coordinates) with the
    others pinned at the box centre. Never forges a certificate: ``None``
    means the test did not fire.
    """
    if len(box) != sys.dim:
        raise ValueError(f"box has dim {len(box)}, system has dim {sys.dim}")
    m = sys.codimension
    if free_axes is None:
        axes = tuple(range(sys.dim - m, sys.dim)) if m <= sys.dim else tuple(range(sys.dim))
    else:
        axes = tuple(int(i) for i in free_axes)
    if len(axes) != m:
        raise ValueError(f"Krawczyk needs {m} free axes, got {len(axes)}")
    pinned = [0.5 * (lo + hi) for lo, hi in box]
    x_bar = [pinned[i] for i in axes]
    radii = [0.5 * (box[i][1] - box[i][0]) for i in axes]
    r = max(radii)
    if r <= 0.0:
        raise ValueError("box radius must be positive")

    def _embed(ys: Sequence[Interval]) -> list[Interval]:
        xs: list[Interval] = [Interval.point(v) for v in pinned]
        for k, ax in enumerate(axes):
            xs[ax] = ys[k] if isinstance(ys[k], Interval) else Interval.from_value(ys[k])
        return xs

    def func(ys: Sequence[Interval]) -> list[Interval]:
        return residual_iv(sys, _embed(ys))

    def jac(ys: Sequence[Interval]) -> list[list[Interval]]:
        full = jacobian_iv(sys, _embed(ys))
        return [[row[ax] for ax in axes] for row in full]

    center_f = [Interval.point(v) for v in x_bar]
    j_c = [[v.mid for v in row] for row in jac(center_f)]
    try:
        a_inv = _float_inverse(j_c)
    except ZeroDivisionError:
        return None
    return krawczyk_certificate(func, jac, x_bar, a_inv, r)


def dF_d_weights(sys: EqualitySystem, x: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """``partial F / partial c``, shape ``m x (m+1)``, closed form for the IFT."""
    sigmas = [sigma_n(sys.base, _z(t, x), t.order) for t in sys.terms]
    m = sys.codimension
    n_c = sys.n_terms
    rows: list[list[float]] = []
    for k in range(m):
        row = [0.0] * n_c
        row[k] = sigmas[k]
        row[k + 1] = -sigmas[k + 1]
        rows.append(row)
    return tuple(tuple(r) for r in rows)


__all__ = [
    "AffineSet",
    "EqualitySystem",
    "NewtonResult",
    "UnitTerm",
    "affine_locus",
    "branch_signature",
    "certify_locus_point",
    "dF_d_weights",
    "hessian_blocks",
    "is_transversal",
    "jacobian",
    "jacobian_iv",
    "locus_tangent",
    "newton_project",
    "pseudoinverse",
    "residual",
    "residual_iv",
    "sigma_n",
    "smallest_singular",
]
