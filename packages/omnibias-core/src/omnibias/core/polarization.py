# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Polarization plans: mixed partial derivatives from directional ones (pure Python).

The ``n``-th derivative of a smooth function at a point is a symmetric ``n``-tensor ``T``, and a mixed partial derivative
is one of its entries, ``\partial^\alpha u = T(e_{i_1}, \dots, e_{i_n})`` for the index list ``i_1 \dots i_n`` that repeats
axis ``i`` ``\alpha_i`` times. The polarization identity writes that entry through directional derivatives
``D^n_w u = T(w, \dots, w)`` along a handful of directions ``w``:

.. math::

    T(e_{i_1}, \dots, e_{i_n}) = \frac{1}{2^n n!} \sum_{\epsilon \in \{\pm 1\}^n}
        \Big(\prod_j \epsilon_j\Big)\, D^n_{\sum_j \epsilon_j e_{i_j}}\, u.

Directions are reduced to a canonical integer representative (``D^n_{c w} = c^n D^n_w`` and ``D^n_{-w} = (-1)^n D^n_w``),
so shared directions across the requested derivatives are carried once. A pure derivative ``\partial_x^n`` is one
direction; ``\partial_x^2 \partial_y^2`` is four; ``\partial_x \partial_y`` is two. A deep network then needs one Taylor
stream per (direction, order): see :mod:`omnibias.jax.experimental.streams` and its PyTorch twin.

Everything here is integer and rational arithmetic with no tensor framework, so every backend reads the same plan.

A plan is stored sparsely: the nonzero (request, direction, order) weights as integer index arrays and exact ``Fraction`` values
(``rows``, ``cols``, ``slots``, ``values``). A mixed partial involves at most four axes, so a request carries at most sixteen
sign patterns; a plan for the bi-Laplacian at ``m = 256`` inputs has about 1.3e5 nonzeros where its dense
``requests x directions x order`` table would hold 8.6e9 entries. The dense views (``coefficients``, ``float_coefficients``,
``dense_floats``) are built on demand and refuse a plan whose table would not fit; plans are memoized.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import factorial, gcd
from typing import Any

import numpy as np

__all__ = [
    "DENSE_CONTRACTION_ENTRIES",
    "MAX_POLARIZED_ORDER",
    "PolarizationPlan",
    "bilaplacian_plan",
    "laplacian_gradient_plan",
    "polarization_plan",
    "polarization_plan_sparse",
]

#: Highest total derivative order the stacked directional-jet engines carry.
MAX_POLARIZED_ORDER = 4

#: ``requests x directions x order`` up to which consumers contract with the dense table (one ``einsum``); above it they gather
#: the nonzeros and add them into their requests, which does the same arithmetic on the entries that exist.
DENSE_CONTRACTION_ENTRIES = 1 << 16

_DENSE_VIEW_ENTRIES = 1 << 22  # the nested-tuple compatibility views refuse a larger table
_DENSE_FLOATS_ENTRIES = 1 << 27  # dense_floats refuses a larger table (1 GiB of float64)
_PLAN_CACHE = 64

SparseKey = tuple[tuple[int, int], ...]  # a direction as ((axis, integer coefficient), ...) with ascending axes


class PolarizationPlan:
    """Directions and rational weights that assemble each requested mixed partial.

    Sparse form: entry ``e`` says that request ``rows[e]`` takes ``values[e]`` (exactly ``Fraction`` ``fractions[e]``) times the
    ``slots[e] + 1``-th directional derivative along direction ``cols[e]``; there are no other nonzero weights. The weight is
    nonzero only for the request's own order. ``order`` is the highest order requested.

    Compatibility views, built on demand: ``directions[j]`` is an integer vector of length ``dim`` and
    ``coefficients[r][j][p - 1]`` the weight of the ``p``-th directional derivative along ``directions[j]`` in request ``r``.
    """

    __slots__ = (
        "orders",
        "order",
        "rows",
        "cols",
        "slots",
        "values",
        "_fractions",
        "_keys",
        "_dim",
        "_directions",
        "_coefficients",
        "memo",
    )

    def __init__(
        self,
        directions: tuple[tuple[int, ...], ...],
        coefficients: tuple[tuple[tuple[Fraction, ...], ...], ...],
        orders: tuple[int, ...],
    ) -> None:
        """Dense construction (the historical signature); :func:`polarization_plan` builds plans through the sparse path."""
        keys = tuple(tuple((i, c) for i, c in enumerate(d) if c) for d in directions)
        rows: list[int] = []
        cols: list[int] = []
        slots: list[int] = []
        fractions: list[Fraction] = []
        for r, req in enumerate(coefficients):
            for j, per in enumerate(req):
                for p, c in enumerate(per):
                    if c != 0:
                        rows.append(r)
                        cols.append(j)
                        slots.append(p)
                        fractions.append(Fraction(c))
        dim = len(directions[0]) if directions else 0
        self._set(keys, dim, tuple(orders), rows, cols, slots, fractions, [float(c) for c in fractions])

    def _set(
        self,
        keys: tuple[SparseKey, ...],
        dim: int,
        orders: tuple[int, ...],
        rows: Sequence[int],
        cols: Sequence[int],
        slots: Sequence[int],
        fractions: Sequence[Fraction],
        values: Sequence[float],
    ) -> None:
        self._keys = keys
        self._dim = dim
        self.orders = orders
        self.order = max(orders) if orders else 0
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.slots = np.asarray(slots, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)
        for a in (self.rows, self.cols, self.slots, self.values):
            a.setflags(write=False)
        self._fractions = tuple(fractions)
        self.memo: dict[Any, Any] = {}  # consumers keep per-device copies of the arrays here; the plan never reads it
        self._directions: tuple[tuple[int, ...], ...] | None = None
        self._coefficients: tuple[tuple[tuple[Fraction, ...], ...], ...] | None = None

    @classmethod
    def _sparse(
        cls,
        keys: tuple[SparseKey, ...],
        dim: int,
        orders: tuple[int, ...],
        rows: Sequence[int],
        cols: Sequence[int],
        slots: Sequence[int],
        fractions: Sequence[Fraction],
        values: Sequence[float],
    ) -> PolarizationPlan:
        plan = object.__new__(cls)
        plan._set(keys, dim, orders, rows, cols, slots, fractions, values)
        return plan

    def __repr__(self) -> str:
        return f"PolarizationPlan(directions={len(self._keys)}, requests={len(self.orders)}, order={self.order})"

    # ------------------------------------------------------------------------------------------ sparse accessors
    @property
    def num_directions(self) -> int:
        return len(self._keys)

    @property
    def num_requests(self) -> int:
        return len(self.orders)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    @property
    def dense_entries(self) -> int:
        """Size of the dense ``requests x directions x order`` table (never built unless asked)."""
        return self.num_requests * self.num_directions * self.order

    @property
    def dense_is_cheap(self) -> bool:
        """Whether a consumer should contract with the dense table (small plans) rather than gather the nonzeros."""
        return self.dense_entries <= DENSE_CONTRACTION_ENTRIES

    def nonzeros(self) -> Iterator[tuple[int, int, int, Fraction]]:
        """``(request, direction, order p, exact weight)`` of every nonzero entry."""
        for r, j, p, c in zip(self.rows.tolist(), self.cols.tolist(), self.slots.tolist(), self._fractions, strict=True):
            yield r, j, p + 1, c

    def direction_matrix(self) -> np.ndarray:
        """The directions as an integer array ``(num_directions, dim)`` built without a Python row per direction."""
        out = np.zeros((len(self._keys), self._dim), dtype=np.int64)
        if self._keys:
            counts = np.fromiter((len(k) for k in self._keys), dtype=np.int64, count=len(self._keys))
            axes = np.fromiter((a for k in self._keys for a, _ in k), dtype=np.int64, count=int(counts.sum()))
            vals = np.fromiter((c for k in self._keys for _, c in k), dtype=np.int64, count=int(counts.sum()))
            out[np.repeat(np.arange(len(self._keys)), counts), axes] = vals
        return out

    def dense_floats(self) -> np.ndarray:
        """The dense float64 table ``(requests, directions, order)``: the same floats the nested view rounds to."""
        if self.dense_entries > _DENSE_FLOATS_ENTRIES:
            raise ValueError(
                f"the dense table of this plan has {self.dense_entries:,} entries; use rows/cols/slots/values "
                f"({self.nnz:,} nonzeros)"
            )
        out = np.zeros((self.num_requests, self.num_directions, self.order), dtype=np.float64)
        out[self.rows, self.cols, self.slots] = self.values
        return out

    # ------------------------------------------------------------------------------------ compatibility views
    @property
    def directions(self) -> tuple[tuple[int, ...], ...]:
        if self._directions is None:
            self._directions = tuple(tuple(row) for row in self.direction_matrix().tolist())
        return self._directions

    @property
    def coefficients(self) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
        if self._coefficients is None:
            if self.dense_entries > _DENSE_VIEW_ENTRIES:
                raise ValueError(
                    f"the nested coefficient table of this plan would hold {self.dense_entries:,} entries; use "
                    f"nonzeros() or rows/cols/slots/values ({self.nnz:,} nonzeros)"
                )
            zero = Fraction(0)
            table = [[[zero] * self.order for _ in range(self.num_directions)] for _ in range(self.num_requests)]
            for r, j, p, c in self.nonzeros():
                table[r][j][p - 1] = c
            self._coefficients = tuple(tuple(tuple(per) for per in req) for req in table)
        return self._coefficients

    def float_coefficients(self) -> list[list[list[float]]]:
        return [[[float(c) for c in row] for row in req] for req in self.coefficients]


def _canonical(w: tuple[int, ...]) -> tuple[tuple[int, ...] | None, int]:
    """``(w', sign, scale)`` factors with ``w = sign * scale * w'``; ``(None, ...)`` for the zero direction."""
    g = 0
    for c in w:
        g = gcd(g, abs(c))
    if g == 0:
        return None, 1
    w = tuple(c // g for c in w)
    sign = 1
    for c in w:
        if c != 0:
            sign = 1 if c > 0 else -1
            break
    w = tuple(sign * c for c in w)
    return w, sign * g


@lru_cache(maxsize=256)
def _template(pattern: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], Fraction, float], ...]:
    """Weights of one request whose axes carry the multiplicities ``pattern`` (ascending axes, every entry >= 1).

    Returns ``(local direction, exact weight, float weight)`` in the order each direction first appears over the sign
    patterns ``product((1, -1), repeat=n)``, including a direction whose weights cancel (it still takes an index, as it
    always has). The local direction has one coordinate per axis of the request; relabeling the axes monotonically
    keeps the gcd and the sign of the first nonzero coordinate, so every request with this pattern shares the template.
    """
    n = sum(pattern)
    axes = [k for k, m in enumerate(pattern) for _ in range(m)]
    acc: dict[tuple[int, ...], Fraction] = {}
    for eps in product((1, -1), repeat=n):
        w = [0] * len(pattern)
        sign = 1
        for e, axis in zip(eps, axes, strict=True):
            w[axis] += e
            sign *= e
        direction, scale = _canonical(tuple(w))
        if direction is None:
            continue
        # D^n_w = scale^n D^n_{w'} with the sign of an odd flip carried inside `scale`
        acc[direction] = acc.get(direction, Fraction(0)) + Fraction(sign * scale**n, 2**n * factorial(n))
    return tuple((d, c, float(c)) for d, c in acc.items())


def _build(requests: tuple[tuple[tuple[int, int], ...], ...], dim: int) -> PolarizationPlan:
    index: dict[SparseKey, int] = {}
    keys: list[SparseKey] = []
    rows: list[int] = []
    cols: list[int] = []
    slots: list[int] = []
    fractions: list[Fraction] = []
    values: list[float] = []
    orders: list[int] = []
    for r, request in enumerate(requests):
        axes = tuple(a for a, _ in request)
        pattern = tuple(m for _, m in request)
        n = sum(pattern)
        orders.append(n)
        for local, c, f in _template(pattern):
            key = tuple((axes[k], coef) for k, coef in enumerate(local) if coef)
            j = index.get(key)
            if j is None:
                j = len(keys)
                index[key] = j
                keys.append(key)
            if c != 0:
                rows.append(r)
                cols.append(j)
                slots.append(n - 1)
                fractions.append(c)
                values.append(f)
    return PolarizationPlan._sparse(tuple(keys), dim, tuple(orders), rows, cols, slots, fractions, values)


@lru_cache(maxsize=_PLAN_CACHE)
def _cached(requests: tuple[tuple[tuple[int, int], ...], ...], dim: int) -> PolarizationPlan:
    return _build(requests, dim)


def _check_sparse(request: Sequence[tuple[int, int]], dim: int) -> tuple[tuple[int, int], ...]:
    merged: dict[int, int] = {}
    for axis, power in request:
        axis, power = int(axis), int(power)
        if not 0 <= axis < dim:
            raise ValueError(f"axis {axis} outside 0..{dim - 1} in {tuple(request)}")
        if power < 0:
            raise ValueError(f"multi-index entries must be nonnegative, got {tuple(request)}")
        if power:
            merged[axis] = merged.get(axis, 0) + power
    n = sum(merged.values())
    if not 1 <= n <= MAX_POLARIZED_ORDER:
        raise ValueError(f"total derivative order must be in 1..{MAX_POLARIZED_ORDER}, got {n} for {tuple(request)}")
    return tuple(sorted(merged.items()))


def polarization_plan_sparse(requests: Sequence[Sequence[tuple[int, int]]], dim: int) -> PolarizationPlan:
    """Plan for requests given as ``((axis, power), ...)`` pairs: ``O(requests)`` to validate, with no ``dim``-long tuple.

    Equal to ``polarization_plan`` of the dense multi-indices; plans are memoized on the request tuple.
    """
    checked = tuple(_check_sparse(r, dim) for r in requests)
    if not checked:
        raise ValueError("at least one multi-index is required")
    return _cached(checked, int(dim))


def polarization_plan(multi_indices: Sequence[Sequence[int]], dim: int | None = None) -> PolarizationPlan:
    """Plan for the partial derivatives ``multi_indices`` (each a length-``dim`` tuple of per-axis orders).

    Every multi-index must have total order between 1 and :data:`MAX_POLARIZED_ORDER`. Plans are memoized.
    """
    requests = [tuple(int(a) for a in alpha) for alpha in multi_indices]
    if not requests:
        raise ValueError("at least one multi-index is required")
    dim = len(requests[0]) if dim is None else dim
    for alpha in requests:
        if len(alpha) != dim:
            raise ValueError(f"every multi-index must have length {dim}, got {alpha}")
        if any(a < 0 for a in alpha):
            raise ValueError(f"multi-index entries must be nonnegative, got {alpha}")
        n = sum(alpha)
        if not 1 <= n <= MAX_POLARIZED_ORDER:
            raise ValueError(f"total derivative order must be in 1..{MAX_POLARIZED_ORDER}, got {n} for {alpha}")
    sparse = tuple(tuple((i, a) for i, a in enumerate(alpha) if a) for alpha in requests)
    return _cached(sparse, int(dim))


@lru_cache(maxsize=_PLAN_CACHE)
def _bilaplacian(dim: int, coordinates: tuple[int, ...]) -> PolarizationPlan:
    requests: list[tuple[tuple[int, int], ...]] = [((i, 4),) for i in coordinates]
    for p, i in enumerate(coordinates):
        for j in coordinates[p + 1 :]:
            requests.append(_check_sparse(((i, 2), (j, 2)), dim))
    return _build(tuple(requests), dim)


def bilaplacian_plan(dim: int, coordinates: Sequence[int] | None = None) -> PolarizationPlan:
    """Requests ``u_iiii`` for ``i`` in ``coordinates`` (default all axes), then ``u_iijj`` for every pair ``i`` before ``j``.

    This is the request list of ``bilaplacian_by_jets`` (``Delta^2 u = sum_i u_iiii + 2 sum_{i<j} u_iijj``), built without the
    ``requests x dim`` dense multi-indices and memoized on ``(dim, coordinates)``.
    """
    idx = tuple(range(dim)) if coordinates is None else tuple(int(c) for c in coordinates)
    for i in idx:
        if not 0 <= i < dim:
            raise ValueError(f"coordinate {i} outside 0..{dim - 1}")
    return _bilaplacian(int(dim), idx)


@lru_cache(maxsize=_PLAN_CACHE)
def _laplacian_gradient(dim: int) -> PolarizationPlan:
    requests = tuple(_check_sparse(((j, 2), (i, 1)), dim) for i in range(dim) for j in range(dim))
    return _build(requests, dim)


def laplacian_gradient_plan(dim: int) -> PolarizationPlan:
    """Requests ``u_{jji}`` for ``i`` major and ``j`` minor over all axes: the ``dim^2`` third derivatives of ``grad Delta u``."""
    return _laplacian_gradient(int(dim))
