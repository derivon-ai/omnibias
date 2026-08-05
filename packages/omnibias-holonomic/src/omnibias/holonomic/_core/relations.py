# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact minimal polynomial linear relations among vectors over ``Q(x)``.

Given an ordered list of vectors ``v_0, v_1, ...`` in ``Q(x)^L`` this finds the smallest
``m`` and polynomials ``u_0, ..., u_m`` (not all zero, ``deg u_k <= max_degree``) with

.. math::

    \sum_{k=0}^{m} u_k(x)\, v_k = 0 \quad\text{in } Q(x)^L.

It is the exact structural engine behind Ore ``lclm`` and ``symmetric_product``
(:mod:`.oreops`), the fast Zeilberger telescoper (:mod:`.zeilberger`), and the D-finite /
algebraic guessers (:mod:`.guess`). The relation is found by clearing denominators
row-by-row (each row is an independent scalar identity) and solving the resulting
homogeneous rational linear system with the exact null space of :mod:`.linalg` -- no
floats, no rounding. Minimality is by construction: ``m`` is searched upward and the full
degree range is exhausted at each order before ``m`` is incremented.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.holonomic._core.linalg import null_space
from omnibias.holonomic._core.ratfunc import RatFunc, rf_is_zero
from omnibias.holonomic._core.rational_poly import Poly, is_zero, pmul, to_poly

Vector = Sequence[RatFunc]


def _plcm(a: Poly, b: Poly) -> Poly:
    from omnibias.holonomic._core.rational_poly import pdivmod, pgcd

    if is_zero(a) or is_zero(b):
        return ()
    g = pgcd(a, b)
    prod = pmul(a, b)
    q, _ = pdivmod(prod, g)
    return q


def _relation_for_order(columns: Sequence[Vector], m: int, max_degree: int) -> list[Poly] | None:
    """Find ``u_0..u_m`` (deg <= ``max_degree``) with ``sum_k u_k columns[k] = 0``, or None."""
    length = len(columns[0])
    used = columns[: m + 1]
    for D in range(max_degree + 1):
        n_unknowns = (m + 1) * (D + 1)
        equations: list[list[Fraction]] = []
        for r in range(length):
            entries = [used[k][r] for k in range(m + 1)]
            den_lcm: Poly = (Fraction(1),)
            for _num, den in entries:
                den_lcm = _plcm(den_lcm, den)
            # Clear denominators: P_{k} = num_k * (den_lcm / den_k) is a polynomial.
            row_polys: list[Poly] = []
            max_pdeg = 0
            for num, den in entries:
                from omnibias.holonomic._core.rational_poly import pdivmod

                factor, rem = pdivmod(den_lcm, den)
                assert is_zero(rem), "lcm not divisible by denominator (normalisation bug)"
                p = pmul(num, factor)
                row_polys.append(p)
                max_pdeg = max(max_pdeg, len(p) - 1 if p else -1)
            highest = max_pdeg + D
            for power in range(highest + 1):
                coeffs = [Fraction(0)] * n_unknowns
                any_nonzero = False
                for k in range(m + 1):
                    p = row_polys[k]
                    for e in range(D + 1):
                        idx = k * (D + 1) + e
                        deg = power - e
                        if 0 <= deg < len(p):
                            coeffs[idx] += p[deg]
                            if p[deg] != 0:
                                any_nonzero = True
                if any_nonzero:
                    equations.append(coeffs)
        if not equations:
            continue
        basis = null_space(equations)
        for sol in basis:
            polys: list[Poly] = []
            for k in range(m + 1):
                block = sol[k * (D + 1) : (k + 1) * (D + 1)]
                polys.append(to_poly(block))
            if any(not is_zero(p) for p in polys):
                return polys
    return None


def find_poly_relation(
    columns: Sequence[Vector], *, max_order: int | None = None, max_degree: int = 6
) -> list[Poly] | None:
    r"""Minimal ``[u_0, ..., u_m]`` (Polys) with ``sum_k u_k columns[k] = 0``, or ``None``.

    ``columns[k]`` are equal-length vectors over :data:`~.ratfunc.RatFunc`. Orders
    ``m = 0, 1, ...`` are tried in turn (``max_order`` caps the search, default
    ``len(columns) - 1``); at each order every degree ``0..max_degree`` is tried, so the
    returned relation has the least possible order. Returns ``None`` when no relation
    exists within the bounds (a genuine finding).
    """
    if not columns:
        return None
    length = len(columns[0])
    if any(len(c) != length for c in columns):
        raise ValueError("all column vectors must have the same length")
    cap = len(columns) - 1 if max_order is None else min(max_order, len(columns) - 1)
    # A single zero column is itself a relation of order 0.
    if all(rf_is_zero(v) for v in columns[0]):
        return [(Fraction(1),)]
    for m in range(cap + 1):
        rel = _relation_for_order(columns, m, max_degree)
        if rel is not None:
            return rel
    return None


__all__ = ["find_poly_relation"]
