# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Vectorized outward-rounded interval arithmetic.

``IntervalArray`` stores lower and upper binary64 endpoint arrays separately.
Each elementary operation is rounded one representable step outwards with
``numpy.nextafter``.  Consequently it encloses the elementwise real operation
for all values in the input boxes, under the same IEEE-754 assumptions as
:mod:`omnibias.core.verified.interval`.

The type is deliberately a small dense-array substrate rather than a replacement
for a sparse package.  :func:`sparse_matvec` accepts COO entries so rigorous
Gram and residual operators can avoid materialising a dense matrix while keeping
the accumulation order explicitly outward rounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from omnibias.core.verified.interval import Interval

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _as_float_array(value: ArrayLike) -> FloatArray:
    """Make an owned binary64 array, preserving a supplied float datum exactly."""
    return cast(FloatArray, np.array(value, dtype=np.float64, copy=True))


def _pred(value: FloatArray) -> FloatArray:
    return cast(FloatArray, np.nextafter(value, -np.inf))


def _succ(value: FloatArray) -> FloatArray:
    return cast(FloatArray, np.nextafter(value, np.inf))


def _mul_bounds(
    a_lo: FloatArray, a_hi: FloatArray, b_lo: FloatArray, b_hi: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Outward-rounded elementwise interval product."""
    products = np.stack((a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi))
    return (
        _pred(cast(FloatArray, np.min(products, axis=0))),
        _succ(cast(FloatArray, np.max(products, axis=0))),
    )


@dataclass(frozen=True, init=False)
class IntervalArray:
    """A shape-preserving array of closed real intervals.

    The constructor accepts matching or broadcastable lower and upper endpoint
    arrays.  It does not round those endpoints; callers must provide values that
    already enclose the intended reals.  :meth:`point` is appropriate when a
    binary64 array is the literal input datum.
    """

    lo: FloatArray
    hi: FloatArray

    __array_priority__ = 1000

    def __init__(self, lo: ArrayLike, hi: ArrayLike) -> None:
        lo_raw = _as_float_array(lo)
        hi_raw = _as_float_array(hi)
        try:
            lo_arr, hi_arr = np.broadcast_arrays(lo_raw, hi_raw)
        except ValueError as exc:
            raise ValueError("interval endpoint shapes are not broadcastable") from exc
        lo_out = cast(FloatArray, np.array(lo_arr, dtype=np.float64, copy=True))
        hi_out = cast(FloatArray, np.array(hi_arr, dtype=np.float64, copy=True))
        if bool(np.any(np.isnan(lo_out))) or bool(np.any(np.isnan(hi_out))):
            raise ValueError("interval endpoints must not be NaN")
        if bool(np.any(lo_out > hi_out)):
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        lo_out.setflags(write=False)
        hi_out.setflags(write=False)
        object.__setattr__(self, "lo", lo_out)
        object.__setattr__(self, "hi", hi_out)

    @classmethod
    def point(cls, value: ArrayLike) -> IntervalArray:
        """Exact point intervals for binary64 input data."""
        values = _as_float_array(value)
        return cls(values, values)

    @classmethod
    def from_value(cls, value: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        """Promote interval or scalar/array data to an interval array."""
        if isinstance(value, IntervalArray):
            return value
        if isinstance(value, Interval):
            return cls(np.asarray(value.lo), np.asarray(value.hi))
        return cls.point(value)

    @classmethod
    def from_intervals(cls, values: ArrayLike) -> IntervalArray:
        """Build a one-dimensional interval array from scalar :class:`Interval`s."""
        raw = np.asarray(values, dtype=object)
        if raw.ndim != 1:
            raise ValueError("from_intervals requires a one-dimensional sequence")
        if not all(isinstance(value, Interval) for value in raw):
            raise TypeError("from_intervals requires only Interval values")
        return cls(
            np.asarray([value.lo for value in raw], dtype=np.float64),
            np.asarray([value.hi for value in raw], dtype=np.float64),
        )

    @classmethod
    def hull(cls, *values: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        """Smallest elementwise interval array containing every supplied input."""
        if not values:
            raise ValueError("hull requires at least one value")
        intervals = [cls.from_value(value) for value in values]
        try:
            lows = np.broadcast_arrays(*(interval.lo for interval in intervals))
            highs = np.broadcast_arrays(*(interval.hi for interval in intervals))
        except ValueError as exc:
            raise ValueError("hull inputs are not broadcastable") from exc
        return cls(
            cast(FloatArray, np.min(np.stack(lows), axis=0)),
            cast(FloatArray, np.max(np.stack(highs), axis=0)),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The common endpoint shape."""
        return self.lo.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.lo.ndim

    @property
    def size(self) -> int:
        """Total number of scalar intervals."""
        return self.lo.size

    @property
    def mid(self) -> FloatArray:
        """A representable midpoint array contained in each interval."""
        midpoint = cast(FloatArray, 0.5 * self.lo + 0.5 * self.hi)
        midpoint = cast(
            FloatArray,
            np.where(np.isneginf(self.lo) & np.isposinf(self.hi), 0.0, midpoint),
        )
        midpoint = cast(FloatArray, np.maximum(self.lo, midpoint))
        return cast(FloatArray, np.minimum(self.hi, midpoint))

    @property
    def rad(self) -> FloatArray:
        """Outward-rounded radius array."""
        midpoint = self.mid
        return _succ(cast(FloatArray, np.maximum(midpoint - self.lo, self.hi - midpoint)))

    @property
    def width(self) -> FloatArray:
        """Outward-rounded widths."""
        return _succ(cast(FloatArray, self.hi - self.lo))

    @property
    def mag(self) -> FloatArray:
        """Outward-rounded elementwise magnitude bounds."""
        return _succ(cast(FloatArray, np.maximum(np.abs(self.lo), np.abs(self.hi))))

    @property
    def mig(self) -> FloatArray:
        """Outward-rounded elementwise mignitudes."""
        straddles_zero = (self.lo <= 0.0) & (0.0 <= self.hi)
        nonzero = _pred(cast(FloatArray, np.minimum(np.abs(self.lo), np.abs(self.hi))))
        return cast(FloatArray, np.where(straddles_zero, 0.0, nonzero))

    def contains(self, value: ArrayLike) -> BoolArray:
        """Elementwise containment test, with NumPy broadcasting."""
        values = _as_float_array(value)
        try:
            lo, hi, points = np.broadcast_arrays(self.lo, self.hi, values)
        except ValueError as exc:
            raise ValueError("containment value is not broadcastable") from exc
        return cast(BoolArray, (lo <= points) & (points <= hi))

    def contains_zero(self) -> BoolArray:
        """Elementwise test for whether zero belongs to each interval."""
        return cast(BoolArray, (self.lo <= 0.0) & (0.0 <= self.hi))

    def _broadcast(
        self, other: IntervalArray | Interval | ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        right = self.from_value(other)
        try:
            left_lo, left_hi, right_lo, right_hi = np.broadcast_arrays(
                self.lo, self.hi, right.lo, right.hi
            )
        except ValueError as exc:
            raise ValueError("interval operand shapes are not broadcastable") from exc
        return (
            cast(FloatArray, left_lo),
            cast(FloatArray, left_hi),
            cast(FloatArray, right_lo),
            cast(FloatArray, right_hi),
        )

    def __add__(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        left_lo, left_hi, right_lo, right_hi = self._broadcast(other)
        return IntervalArray(_pred(left_lo + right_lo), _succ(left_hi + right_hi))

    __radd__ = __add__

    def __neg__(self) -> IntervalArray:
        return IntervalArray(-self.hi, -self.lo)

    def __sub__(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        left_lo, left_hi, right_lo, right_hi = self._broadcast(other)
        return IntervalArray(_pred(left_lo - right_hi), _succ(left_hi - right_lo))

    def __rsub__(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        return self.from_value(other).__sub__(self)

    def __mul__(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        left_lo, left_hi, right_lo, right_hi = self._broadcast(other)
        lo, hi = _mul_bounds(left_lo, left_hi, right_lo, right_hi)
        return IntervalArray(lo, hi)

    __rmul__ = __mul__

    def reciprocal(self) -> IntervalArray:
        """Elementwise reciprocal; every entry must exclude zero."""
        if bool(np.any(self.contains_zero())):
            raise ZeroDivisionError("interval reciprocal requires zero outside every interval")
        return IntervalArray(_pred(1.0 / self.hi), _succ(1.0 / self.lo))

    def __truediv__(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        return self * self.from_value(other).reciprocal()

    def __rtruediv__(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        return self.from_value(other) * self.reciprocal()

    def __pow__(self, exponent: int) -> IntervalArray:
        return self.pow_int(exponent)

    def pow_int(self, exponent: int) -> IntervalArray:
        """Outward-rounded elementwise integer powers via binary exponentiation."""
        if not isinstance(exponent, int):
            raise TypeError("interval power exponent must be an integer")
        if exponent < 0:
            return self.pow_int(-exponent).reciprocal()
        result = IntervalArray.point(np.ones(self.shape, dtype=np.float64))
        base = self.abs() if exponent % 2 == 0 else self
        remaining = exponent
        while remaining > 0:
            if remaining & 1:
                result = result * base
            remaining >>= 1
            if remaining > 0:
                base = base * base
        return result

    def abs(self) -> IntervalArray:
        """Elementwise enclosure of absolute value."""
        nonnegative = self.lo >= 0.0
        nonpositive = self.hi <= 0.0
        straddles_zero = ~(nonnegative | nonpositive)
        lower = cast(
            FloatArray,
            np.where(nonnegative, self.lo, np.where(nonpositive, -self.hi, 0.0)),
        )
        upper_raw = cast(
            FloatArray,
            np.where(nonnegative, self.hi, np.where(nonpositive, -self.lo, 0.0)),
        )
        upper = cast(
            FloatArray,
            np.where(
                straddles_zero,
                _succ(cast(FloatArray, np.maximum(-self.lo, self.hi))),
                upper_raw,
            ),
        )
        return IntervalArray(lower, upper)

    __abs__ = abs

    def sqrt(self) -> IntervalArray:
        """Elementwise square root; every entry must be non-negative."""
        if bool(np.any(self.lo < 0.0)):
            raise ValueError("sqrt requires non-negative intervals")
        return IntervalArray(_pred(np.sqrt(self.lo)), _succ(np.sqrt(self.hi)))

    def intersect(self, other: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        """Elementwise intersection; raises when any entry is empty."""
        left_lo, left_hi, right_lo, right_hi = self._broadcast(other)
        lo = cast(FloatArray, np.maximum(left_lo, right_lo))
        hi = cast(FloatArray, np.minimum(left_hi, right_hi))
        if bool(np.any(lo > hi)):
            raise ValueError("empty interval intersection")
        return IntervalArray(lo, hi)

    def dot(self, other: IntervalArray | Interval | ArrayLike) -> Interval:
        """Rigorous dot product of two one-dimensional interval arrays."""
        right = self.from_value(other)
        if self.ndim != 1 or right.ndim != 1 or self.shape != right.shape:
            raise ValueError("dot requires one-dimensional operands of equal shape")
        products = self * right
        lo = 0.0
        hi = 0.0
        for lower, upper in zip(products.lo, products.hi, strict=True):
            lo = float(np.nextafter(lo + float(lower), -np.inf))
            hi = float(np.nextafter(hi + float(upper), np.inf))
        return Interval(lo, hi)

    def matvec(self, vector: IntervalArray | Interval | ArrayLike) -> IntervalArray:
        """Rigorous dense matrix-vector product with outward accumulation."""
        values = self.from_value(vector)
        if self.ndim != 2 or values.ndim != 1 or self.shape[1] != values.shape[0]:
            raise ValueError("matvec requires an (m, n) matrix and length-n vector")
        products = self * values
        lower = np.zeros(self.shape[0], dtype=np.float64)
        upper = np.zeros(self.shape[0], dtype=np.float64)
        for column in range(self.shape[1]):
            lower = _pred(lower + products.lo[:, column])
            upper = _succ(upper + products.hi[:, column])
        return IntervalArray(lower, upper)

    def __repr__(self) -> str:
        return f"IntervalArray(lo={self.lo!r}, hi={self.hi!r})"


IntervalArrayLike: TypeAlias = IntervalArray | Interval | ArrayLike


def dot(left: IntervalArrayLike, right: IntervalArrayLike) -> Interval:
    """Rigorous one-dimensional interval dot product."""
    return IntervalArray.from_value(left).dot(right)


def sparse_matvec(
    rows: ArrayLike,
    columns: ArrayLike,
    data: IntervalArrayLike,
    vector: IntervalArrayLike,
    shape: tuple[int, int],
) -> IntervalArray:
    """Rigorous COO sparse matrix-vector product.

    ``rows[k]``, ``columns[k]``, and ``data[k]`` specify one matrix entry.
    Duplicate entries are accumulated separately, as required for COO input.
    """
    n_rows, n_columns = shape
    if n_rows < 0 or n_columns < 0:
        raise ValueError("sparse matrix dimensions must be non-negative")
    raw_rows = np.asarray(rows)
    raw_columns = np.asarray(columns)
    if raw_rows.ndim != 1 or raw_columns.ndim != 1:
        raise ValueError("COO row and column indices must be one-dimensional")
    if raw_rows.shape != raw_columns.shape:
        raise ValueError("COO row and column indices must have equal length")
    if not np.issubdtype(raw_rows.dtype, np.integer) or not np.issubdtype(
        raw_columns.dtype, np.integer
    ):
        raise TypeError("COO row and column indices must be integers")
    row_indices = np.asarray(raw_rows, dtype=np.intp)
    column_indices = np.asarray(raw_columns, dtype=np.intp)
    if bool(np.any(row_indices < 0)) or bool(np.any(row_indices >= n_rows)):
        raise ValueError("COO row index outside sparse matrix shape")
    if bool(np.any(column_indices < 0)) or bool(np.any(column_indices >= n_columns)):
        raise ValueError("COO column index outside sparse matrix shape")

    entries = IntervalArray.from_value(data)
    values = IntervalArray.from_value(vector)
    if entries.ndim != 1 or entries.shape[0] != row_indices.shape[0]:
        raise ValueError("COO data must be one-dimensional with one entry per index")
    if values.ndim != 1 or values.shape[0] != n_columns:
        raise ValueError("sparse matvec vector length does not match shape")

    lower = np.zeros(n_rows, dtype=np.float64)
    upper = np.zeros(n_rows, dtype=np.float64)
    for row, column, entry_lo, entry_hi in zip(
        row_indices, column_indices, entries.lo, entries.hi, strict=True
    ):
        product_lo, product_hi = _mul_bounds(
            np.asarray(entry_lo, dtype=np.float64),
            np.asarray(entry_hi, dtype=np.float64),
            np.asarray(values.lo[column], dtype=np.float64),
            np.asarray(values.hi[column], dtype=np.float64),
        )
        row_index = int(row)
        lower[row_index] = np.nextafter(lower[row_index] + product_lo.item(), -np.inf)
        upper[row_index] = np.nextafter(upper[row_index] + product_hi.item(), np.inf)
    return IntervalArray(lower, upper)


__all__ = [
    "BoolArray",
    "FloatArray",
    "IntervalArray",
    "IntervalArrayLike",
    "dot",
    "sparse_matvec",
]
