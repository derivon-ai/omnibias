# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact rational irregular / Birkhoff finite-difference stencils (theory 01-04).

Generate weights for arbitrary node sets and per-node order sets by solving a
confluent Vandermonde system over ``Q``. This is the founding ``delta -> 0``
register: finite differences becoming derivatives. No temperature collapse.

Nodes in :class:`StencilRequest` are dimensionless ``c_i`` (units of the scale
``h``). Scale-free weights ``A_{i,p}`` satisfy ``a_{i,p} = A_{i,p} / h^{q-p}``
so that

.. math::

    \sum_{(i,p)} a_{i,p}\, f^{(p)}(c_i h) = f^{(q)}(0) + O(h^r).

The theory spec wrote ``A_{i,p} = h^q a_{i,p}``; the worked example only closes
if ``A_{i,p} = a_{i,p}\, h^{q-p}`` (the value-at-``h`` weight is ``1/3``, not
``h/3``). This module uses the latter.

Pure Python: no tensor imports. The exact rank test is the authoritative
poisedness oracle; :func:`omnibias.core.multipack.is_poised` stays the
numerical rank test and must not be read as exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

from omnibias.core.verified.interval import Interval
from omnibias.difference._core.extraction import FiniteDifferenceCertificate

_LEAD_SCAN = 12


def _as_frac(x: Fraction | int | str) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def _fact(n: int) -> Fraction:
    if n < 0:
        raise ValueError(f"factorial of negative: {n}")
    return Fraction(factorial(n))


def _pow_c(c: Fraction, k: int) -> Fraction:
    if k < 0:
        raise ValueError(f"negative power {k}")
    if k == 0:
        return Fraction(1)
    return c**k


@dataclass(frozen=True)
class StencilRequest:
    """A Birkhoff finite-difference request.

    Parameters
    ----------
    nodes:
        Dimensionless node locations ``c_i`` (physical nodes are ``c_i h``).
    orders:
        Available derivative orders at each node (non-empty, unique, ``>= 0``).
    target_order:
        Derivative order ``q`` to recover at the expansion point ``0``.
    """

    nodes: tuple[Fraction, ...]
    orders: tuple[tuple[int, ...], ...]
    target_order: int

    def __post_init__(self) -> None:
        nodes = tuple(_as_frac(c) for c in self.nodes)
        if not nodes:
            raise ValueError("StencilRequest requires at least one node")
        if len(nodes) != len(self.orders):
            raise ValueError("nodes and orders must have the same length")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be distinct")
        cleaned: list[tuple[int, ...]] = []
        for i, ords in enumerate(self.orders):
            seq = tuple(int(p) for p in ords)
            if not seq:
                raise ValueError(f"orders[{i}] must be non-empty")
            if any(p < 0 for p in seq):
                raise ValueError(f"orders[{i}] must be >= 0")
            if len(set(seq)) != len(seq):
                raise ValueError(f"orders[{i}] must be unique")
            cleaned.append(tuple(sorted(seq)))
        q = int(self.target_order)
        if q < 0:
            raise ValueError(f"target_order must be >= 0, got {q}")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "orders", tuple(cleaned))
        object.__setattr__(self, "target_order", q)

    @property
    def n_conditions(self) -> int:
        return sum(len(o) for o in self.orders)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        """``(node_index, order)`` pairs in request order."""
        out: list[tuple[int, int]] = []
        for i, ords in enumerate(self.orders):
            for p in ords:
                out.append((i, p))
        return tuple(out)


@dataclass(frozen=True)
class IrregularStencil:
    """Exact rational irregular stencil (scale-free weights ``A_{i,p}``)."""

    request: StencilRequest
    weights: tuple[tuple[Fraction, ...], ...]
    accuracy: int
    leading_coeff: Fraction
    weight_magnitude: Fraction


def _conditions(request: StencilRequest) -> list[tuple[int, int]]:
    return list(request.pairs())


def _vandermonde(request: StencilRequest) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Rows ``j = 0 .. N-1``, columns ``(i, p)``: ``c_i^{j-p} / (j-p)!``."""
    pairs = _conditions(request)
    n = len(pairs)
    q = request.target_order
    nodes = request.nodes
    mat: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for j in range(n):
        row: list[Fraction] = []
        for i, p in pairs:
            if j < p:
                row.append(Fraction(0))
            else:
                row.append(_pow_c(nodes[i], j - p) / _fact(j - p))
        mat.append(row)
        rhs.append(Fraction(1) if j == q else Fraction(0))
    return mat, rhs


def _solve_over_q(
    mat: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction] | None:
    """Gaussian elimination with partial pivoting over ``Q``. ``None`` if singular."""
    n = len(mat)
    if n == 0:
        return []
    aug = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for col in range(n):
        pivot = col
        best = abs(aug[col][col])
        for r in range(col + 1, n):
            mag = abs(aug[r][col])
            if mag > best:
                best = mag
                pivot = r
        if aug[pivot][col] == 0:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _pack_weights(
    request: StencilRequest, flat: Sequence[Fraction]
) -> tuple[tuple[Fraction, ...], ...]:
    pairs = _conditions(request)
    if len(flat) != len(pairs):
        raise ValueError("flat weight length mismatch")
    by_node: list[list[Fraction]] = [[] for _ in request.nodes]
    for (i, _p), a in zip(pairs, flat, strict=True):
        by_node[i].append(a)
    return tuple(tuple(row) for row in by_node)


def _monomial_residual(request: StencilRequest, flat: Sequence[Fraction], j: int) -> Fraction:
    """``C_j = sum A_{i,p} c_i^{j-p} / (j-p)!``."""
    acc = Fraction(0)
    for (i, p), a in zip(_conditions(request), flat, strict=True):
        if j < p:
            continue
        acc += a * _pow_c(request.nodes[i], j - p) / _fact(j - p)
    return acc


def _leading(request: StencilRequest, flat: Sequence[Fraction]) -> tuple[int, Fraction]:
    """First unmatched monomial ``j >= N`` and its residual ``C_j``."""
    n = len(flat)
    q = request.target_order
    last_c = Fraction(0)
    last_j = n
    for j in range(n, n + _LEAD_SCAN + 1):
        c = _monomial_residual(request, flat, j)
        last_c = c
        last_j = j
        if c != 0:
            return j - q, c
    return last_j - q, last_c


def solve_irregular_stencil(request: StencilRequest) -> IrregularStencil | None:
    """Exact rational solve. ``None`` when the scheme is not poised."""
    mat, rhs = _vandermonde(request)
    flat = _solve_over_q(mat, rhs)
    if flat is None:
        return None
    packed = _pack_weights(request, flat)
    accuracy, coeff = _leading(request, flat)
    mag = max((abs(a) for a in flat), default=Fraction(0))
    return IrregularStencil(
        request=request,
        weights=packed,
        accuracy=int(accuracy),
        leading_coeff=coeff,
        weight_magnitude=mag,
    )


def is_poised_exact(request: StencilRequest) -> bool:
    """Authoritative poisedness oracle: exact rank of the system over ``Q``."""
    mat, rhs = _vandermonde(request)
    return _solve_over_q(mat, rhs) is not None


def polya_screen(request: StencilRequest) -> bool:
    """Cheap necessary condition. ``False`` proves not poised; ``True`` proves nothing."""
    max_p = 0
    for ords in request.orders:
        if ords:
            max_p = max(max_p, max(ords))
    n_cols = max_p + 1
    counts = [0] * n_cols
    for ords in request.orders:
        for p in ords:
            counts[p] += 1
    for j in range(n_cols):
        total = sum(counts[: j + 1])
        if total < j + 1:
            return False
    return True


def physical_weights(stencil: IrregularStencil, h: Fraction | int) -> tuple[tuple[Fraction, ...], ...]:
    """``a_{i,p} = A_{i,p} / h^{q-p}``."""
    hh = _as_frac(h)
    if hh == 0:
        raise ValueError("h must be nonzero")
    q = stencil.request.target_order
    out: list[tuple[Fraction, ...]] = []
    for ords, row in zip(stencil.request.orders, stencil.weights, strict=True):
        phys: list[Fraction] = []
        for p, a in zip(ords, row, strict=True):
            phys.append(a / (hh ** (q - p)))
        out.append(tuple(phys))
    return tuple(out)


def apply_irregular_stencil(
    stencil: IrregularStencil,
    h: Fraction | int,
    samples: Sequence[Sequence[float]],
) -> float:
    """Evaluate ``sum a_{i,p} f^{(p)}(c_i h)`` in float (numerical register)."""
    phys = physical_weights(stencil, h)
    acc = 0.0
    for row_a, row_s in zip(phys, samples, strict=True):
        if len(row_a) != len(row_s):
            raise ValueError("samples must match per-node orders")
        for a, val in zip(row_a, row_s, strict=True):
            acc += float(a) * float(val)
    return acc


def certified_irregular_error(
    stencil: IrregularStencil,
    *,
    h: Fraction | int,
    deriv_bound: Interval,
    estimate: float = 0.0,
    z: float = 0.0,
    name: str = "irregular",
) -> FiniteDifferenceCertificate:
    r"""``|E| <= |C| h^r M_N``, outward rounded, as a :class:`FiniteDifferenceCertificate`.

    ``deriv_bound`` encloses ``f^{(q+r)}`` (or a bound on its magnitude) over the
    hull of the physical nodes. Truncation only; float cancellation in the
    applied stencil is a separate register.
    """
    hh = _as_frac(h)
    if hh == 0:
        raise ValueError("h must be nonzero")
    r = int(stencil.accuracy)
    coeff_iv = Interval.from_rational(abs(stencil.leading_coeff))
    h_iv = Interval.from_rational(hh)
    hr = h_iv**r if r >= 1 else Interval.from_rational(1)
    mag_f = Interval(-deriv_bound.mag, deriv_bound.mag)
    err_iv = coeff_iv * hr * mag_f
    error_bound = float(err_iv.mag)
    enclosure = Interval.point(float(estimate))
    return FiniteDifferenceCertificate(
        name,
        stencil.request.target_order,
        float(z),
        float(hh),
        "irregular",
        float(estimate),
        enclosure,
        error_bound,
        r,
    )


__all__ = [
    "IrregularStencil",
    "StencilRequest",
    "apply_irregular_stencil",
    "certified_irregular_error",
    "is_poised_exact",
    "physical_weights",
    "polya_screen",
    "solve_irregular_stencil",
]
