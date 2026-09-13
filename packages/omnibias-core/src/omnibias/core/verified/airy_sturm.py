# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rational Sturm certificates for the first two Dirichlet Airy eigenvalues.

The operator is ``-d²/dr²+r`` on the half-line, with its Friedrichs
Dirichlet condition at zero. No Airy evaluation, floating arithmetic or
tabulated zero enters a certificate. The finite calculation uses a lower
staircase on [0,6], a Neumann cut at 6, and exact power-series enclosures.
The infinite-domain implication is proved in
``docs/api/gauge-commutator-matrix.md``; it is not a Lean theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import factorial
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest


@dataclass(frozen=True)
class _Interval:
    lo: Q
    hi: Q

    def __add__(self, other: _Interval) -> _Interval:
        return _Interval(self.lo + other.lo, self.hi + other.hi)

    def __mul__(self, other: _Interval) -> _Interval:
        corners = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return _Interval(min(corners), max(corners))

    def rounded(self, bits: int) -> _Interval:
        scale = 1 << bits
        low = self.lo * scale
        high = self.hi * scale
        return _Interval(
            Q(low.numerator // low.denominator, scale),
            Q(-((-high.numerator) // high.denominator), scale),
        )

    def serialized(self) -> list[str]:
        return [str(self.lo), str(self.hi)]

    def sign(self) -> int:
        return 1 if self.lo > 0 else -1 if self.hi < 0 else 0


def _integer(value: int, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not bool or a floating value")
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low},{high}]")
    return value


def _transfer(q: Q, h: Q, order: int, bits: int) -> tuple[_Interval, _Interval, Q, Q]:
    """Enclose exp(h [[0,1],[q,0]]) through its even/odd scalar series."""
    z = abs(q) * h * h
    ratio_c = z / ((2 * order + 3) * (2 * order + 4))
    ratio_s = z / ((2 * order + 4) * (2 * order + 5))
    if max(ratio_c, ratio_s) >= 1:
        raise ValueError("the factorial-tail ratio must be strictly below one")
    c = sum((q**k * h ** (2 * k) / factorial(2 * k) for k in range(order + 1)), Q(0))
    s = sum((q**k * h ** (2 * k + 1) / factorial(2 * k + 1) for k in range(order + 1)), Q(0))
    error_c = z ** (order + 1) / factorial(2 * order + 2) / (1 - ratio_c)
    error_s = h * z ** (order + 1) / factorial(2 * order + 3) / (1 - ratio_s)
    return (
        _Interval(c - error_c, c + error_c).rounded(bits),
        _Interval(s - error_s, s + error_s).rounded(bits),
        error_c,
        error_s,
    )


def _shoot(energy: Q, index: int, cells: int, order: int, bits: int) -> dict[str, Any]:
    h = Q(6, cells)
    u = _Interval(Q(0), Q(0))
    derivative = _Interval(Q(1), Q(1))
    previous_sign = 1  # u(0)=0 and u'(0)=1 imply positivity immediately to the right.
    all_signs = True
    zero_cells: list[int] = []
    rows: list[dict[str, Any]] = []
    for j in range(cells):
        q = j * h - energy
        c, s, error_c, error_s = _transfer(q, h, order, bits)
        u_next = (c * u + s * derivative).rounded(bits)
        derivative_next = (_Interval(q, q) * s * u + c * derivative).rounded(bits)
        sign = u_next.sign()
        # An oscillatory cell has zero separation pi/sqrt(-q)>h since pi>3.
        # For q>=0 a nonzero exponential/affine solution has at most one zero.
        one_zero = max(-q, Q(0)) * h * h < 9
        all_signs = all_signs and sign != 0 and one_zero
        if sign and sign != previous_sign:
            zero_cells.append(j)
        if sign:
            previous_sign = sign
        rows.append(
            {
                "cell": j,
                "potential_lower": str(j * h),
                "q": str(q),
                "C": c.serialized(),
                "S": s.serialized(),
                "C_remainder_upper": str(error_c),
                "S_remainder_upper": str(error_s),
                "u_right": u_next.serialized(),
                "derivative_right": derivative_next.serialized(),
                "u_right_sign": sign,
                "at_most_one_zero_verified": one_zero,
            }
        )
        u, derivative = u_next, derivative_next
    final_phase = all_signs and derivative.sign() != 0
    count = len(zero_cells) + int(u.sign() != derivative.sign()) if final_phase else None
    passed = count is not None and count <= index and energy < 6
    return {
        "energy": str(energy),
        "eigenvalue_index_zero_based": index,
        "initial_data": ["0", "1"],
        "cells": rows,
        "zero_cells": zero_cells if all_signs else None,
        "all_endpoint_signs_and_cell_separations_verified": all_signs,
        "final_u": u.serialized(),
        "final_derivative": derivative.serialized(),
        "dirichlet_neumann_count_below_energy": count,
        "finite_sturm_count_verified": final_phase,
        "neumann_exterior_potential_floor": "6",
        "target_half_line_lower_bound_verified": passed,
    }


def airy_half_line_lower_bounds(
    *, cells: int = 192, series_order: int = 6, rounding_bits: int = 128
) -> dict[str, Any]:
    """Prove lambda_0>23/10 and lambda_1>4 by complete finite Sturm checks.

    Alternative meshes/precision are allowed within explicit resource
    bounds. An unresolved sign or an inadequate lower staircase produces
    INCONCLUSIVE, never an assertion that the Airy inequality is false.
    All certificate numbers are integers or canonical rational strings.
    """
    cells = _integer(cells, "cells", 12, 4096)
    series_order = _integer(series_order, "series_order", 1, 32)
    rounding_bits = _integer(rounding_bits, "rounding_bits", 16, 1024)
    rows = [
        _shoot(Q(23, 10), 0, cells, series_order, rounding_bits),
        _shoot(Q(4), 1, cells, series_order, rounding_bits),
    ]
    passed = all(row["target_half_line_lower_bound_verified"] for row in rows)
    payload = {
        "type": "airy_half_line_sturm_v1",
        "inputs": {"cells": cells, "series_order": series_order, "rounding_bits": rounding_bits},
        "status": "PASS" if passed else "INCONCLUSIVE",
        "operator": "-d^2/dr^2+r on L2((0,infinity),dr), Friedrichs Dirichlet at0",
        "comparison": "lower staircase V_j=j*h on[0,6], Dirichlet at0/Neumann at6; exterior floor6",
        "arithmetic": {"length": "6", "step": str(Q(6, cells)), "thresholds": ["23/10", "4"]},
        "shooting": rows,
        "finite_sturm_counts_verified": all(row["finite_sturm_count_verified"] for row in rows),
        "half_line_eigenvalue_lower_bounds_verified_in_written_analysis": passed,
        "transcendental_evaluation_used": False,
        "tabulated_airy_zeros_used": False,
        "floating_arithmetic_used": False,
        "analytic_proof_formally_verified": False,
        "continuum_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    return {
        **payload,
        "certificate": make_certificate(
            claim="two half-line Dirichlet Airy eigenvalue lower bounds from complete rational Sturm counts",
            payload=payload,
            meta={
                "transcend_backend": "not_used",
                "analytic_implication": "docs/api/gauge-commutator-matrix.md",
            },
        ),
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_airy_sturm_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute every cell and scope field of a PASS or INCONCLUSIVE source."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "airy_half_line_sturm_v1":
            return False
        inputs = payload["inputs"]
        expected = airy_half_line_lower_bounds(
            cells=inputs["cells"],
            series_order=inputs["series_order"],
            rounding_bits=inputs["rounding_bits"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["airy_half_line_lower_bounds", "replay_airy_sturm_certificate"]
