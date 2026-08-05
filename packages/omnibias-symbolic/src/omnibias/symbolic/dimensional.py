# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Dimensional analysis: Buckingham-Pi groups and a dimensionless library filter.

The Buckingham :math:`\Pi` theorem says that a physical relation among ``n``
dimensional variables built from ``k`` independent base dimensions can be recast
as a relation among ``n - rank(D)`` **dimensionless** groups, where ``D`` is the
``k x n`` *unit-dimension matrix* (column ``j`` = the exponents of the base
dimensions in variable ``j``).  A dimensionless product
``prod_j v_j^{p_j}`` corresponds exactly to an integer vector ``p`` in the
**null space** of ``D`` (``D p = 0``).

This module computes that null space **exactly** over the rationals (no floating
point), returns primitive integer :math:`\Pi`-groups, and exposes a filter that
restricts a candidate monomial library to its dimensionless members -- the
dimensional-analysis prior for symbolic discovery.

Everything here is pure Python (``fractions`` + ``math.gcd``); nothing imports a
numerical backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "DimensionalSystem",
    "PiGroup",
    "buckingham_pi_groups",
    "dimension_matrix",
    "dimensionless_residual",
    "filter_dimensionless_monomials",
    "integer_null_space",
    "is_dimensionless",
    "n_dimensionless_groups",
]


@dataclass(frozen=True)
class DimensionalSystem:
    """A set of named variables with integer exponents over base dimensions.

    ``exponents[i][r]`` is the power of base dimension ``base_dimensions[r]`` in
    variable ``variable_names[i]`` (e.g. velocity ``= L^1 T^-1`` over ``(M, L,
    T)`` is ``(0, 1, -1)``).
    """

    variable_names: tuple[str, ...]
    base_dimensions: tuple[str, ...]
    exponents: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.exponents) != len(self.variable_names):
            raise ValueError("exponents must have one row per variable")
        for row in self.exponents:
            if len(row) != len(self.base_dimensions):
                raise ValueError("each exponent row must match base_dimensions length")

    @staticmethod
    def from_dimensions(
        dimensions: Mapping[str, Mapping[str, int]],
        *,
        base_dimensions: Sequence[str] | None = None,
    ) -> DimensionalSystem:
        """Build from ``{variable: {base_dim: exponent}}`` (missing dims = 0)."""
        names = tuple(dimensions.keys())
        if base_dimensions is None:
            seen: list[str] = []
            for spec in dimensions.values():
                for dim in spec:
                    if dim not in seen:
                        seen.append(dim)
            bases = tuple(sorted(seen))
        else:
            bases = tuple(base_dimensions)
        rows = tuple(
            tuple(int(dimensions[name].get(dim, 0)) for dim in bases) for name in names
        )
        return DimensionalSystem(variable_names=names, base_dimensions=bases, exponents=rows)


@dataclass(frozen=True)
class PiGroup:
    """A dimensionless product ``prod_j variable_j^{powers[j]}`` (integer powers)."""

    powers: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, int]:
        return {name: power for name, power in self.powers}

    def formula(self) -> str:
        """Human-readable monomial, e.g. ``rho * U * L / mu``."""
        num: list[str] = []
        den: list[str] = []
        for name, power in self.powers:
            if power == 0:
                continue
            mag = abs(power)
            token = name if mag == 1 else f"{name}^{mag}"
            (num if power > 0 else den).append(token)
        num_s = " * ".join(num) if num else "1"
        if not den:
            return num_s
        den_s = " * ".join(den)
        return f"{num_s} / {den_s}" if len(den) == 1 else f"{num_s} / ({den_s})"


def dimension_matrix(system: DimensionalSystem) -> list[list[int]]:
    """The ``k x n`` unit-dimension matrix ``D`` (base dimensions x variables)."""
    k = len(system.base_dimensions)
    return [[system.exponents[j][r] for j in range(len(system.variable_names))] for r in range(k)]


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """Reduced row echelon form over the rationals; returns ``(rref, pivot_cols)``."""
    rows = [row[:] for row in matrix]
    if not rows:
        return rows, []
    n_rows = len(rows)
    n_cols = len(rows[0])
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        pivot = None
        for i in range(r, n_rows):
            if rows[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        head = rows[r][c]
        rows[r] = [val / head for val in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r], strict=True)]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def _primitive_integer(vec: Sequence[Fraction]) -> list[int]:
    """Scale a rational vector to a primitive (gcd 1) integer vector, sign-fixed."""
    denom_lcm = 1
    for value in vec:
        denom_lcm = denom_lcm * value.denominator // math.gcd(denom_lcm, value.denominator)
    ints = [int(value * denom_lcm) for value in vec]
    g = 0
    for entry in ints:
        g = math.gcd(g, abs(entry))
    if g > 1:
        ints = [entry // g for entry in ints]
    for entry in ints:  # fix global sign so the first nonzero entry is positive
        if entry != 0:
            if entry < 0:
                ints = [-x for x in ints]
            break
    return ints


def integer_null_space(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    """Exact primitive-integer basis of the null space ``{p : matrix @ p = 0}``.

    Computed via rational RREF (no floating point); each returned vector is
    reduced to primitive integers with a canonical (first-nonzero-positive) sign.
    """
    if not matrix:
        return []
    n_cols = len(matrix[0])
    frac = [[Fraction(int(v)) for v in row] for row in matrix]
    rref, pivots = _rref(frac)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]
    basis: list[list[int]] = []
    for free in free_cols:
        vec = [Fraction(0) for _ in range(n_cols)]
        vec[free] = Fraction(1)
        for row_idx, pivot_col in enumerate(pivots):
            vec[pivot_col] = -rref[row_idx][free]
        basis.append(_primitive_integer(vec))
    return basis


def n_dimensionless_groups(system: DimensionalSystem) -> int:
    """Buckingham count ``n_vars - rank(D)`` = number of independent Pi groups."""
    frac = [[Fraction(v) for v in row] for row in dimension_matrix(system)]
    _, pivots = _rref(frac)
    return len(system.variable_names) - len(pivots)


def buckingham_pi_groups(system: DimensionalSystem) -> list[PiGroup]:
    """Independent dimensionless :math:`\\Pi`-groups (primitive integer powers)."""
    null = integer_null_space(dimension_matrix(system))
    groups: list[PiGroup] = []
    for vec in null:
        powers = tuple((system.variable_names[j], vec[j]) for j in range(len(vec)))
        groups.append(PiGroup(powers=powers))
    return groups


def dimensionless_residual(
    system: DimensionalSystem, powers: Mapping[str, int] | Sequence[int]
) -> tuple[int, ...]:
    """Net dimension vector of ``prod_j v_j^{powers_j}`` (all zeros = dimensionless)."""
    if isinstance(powers, Mapping):
        power_list = [int(powers.get(name, 0)) for name in system.variable_names]
    else:
        power_list = [int(p) for p in powers]
        if len(power_list) != len(system.variable_names):
            raise ValueError("powers sequence must match the number of variables")
    k = len(system.base_dimensions)
    residual = [0] * k
    for j, p in enumerate(power_list):
        for r in range(k):
            residual[r] += p * system.exponents[j][r]
    return tuple(residual)


def is_dimensionless(
    system: DimensionalSystem, powers: Mapping[str, int] | Sequence[int]
) -> bool:
    """``True`` iff the monomial with the given variable powers is dimensionless."""
    return all(value == 0 for value in dimensionless_residual(system, powers))


def filter_dimensionless_monomials(
    system: DimensionalSystem,
    monomials: Sequence[Mapping[str, int] | Sequence[int]],
) -> list[Mapping[str, int] | Sequence[int]]:
    """Keep only the dimensionless members of a candidate monomial library."""
    return [m for m in monomials if is_dimensionless(system, m)]
