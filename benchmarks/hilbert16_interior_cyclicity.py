#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay finite algebra for the restricted interior two-zero criterion.

The companion proof is packages/omnibias-dynamics/HILBERT16-INTERIOR-CYCLICITY.md.
Rational examples exercise the sufficient inequality. They do not estimate
the singular remainder for an actual vector field or prove Hilbert 16.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def symbolic_checks() -> list[str]:
    v, t, a, b = sp.symbols("v t a b", positive=True)
    H, e = sp.Function("H"), sp.Function("e")
    M = a * v / (1 - b * v)
    q = (M + e(v)) ** 2 - M**2
    F = H(v**2) - (M + e(v)) ** 2
    target = (
        4 * v * sp.diff(H(t), t, 2).subs(t, v**2)
        - 6 * a**2 * b / (1 - b * v) ** 4
        - sp.diff(q, v, 2) / v
        + sp.diff(q, v) / v**2
    )
    checks = {
        "actual_weighted_curvature": sp.diff(sp.diff(F, v) / v, v) - target,
        "entry_error_first": sp.diff(q, v)
        - (2 * sp.diff(M, v) * e(v) + 2 * M * sp.diff(e(v), v) + 2 * e(v) * sp.diff(e(v), v)),
        "entry_error_second": sp.diff(q, v, 2)
        - (
            2 * sp.diff(M, v, 2) * e(v)
            + 4 * sp.diff(M, v) * sp.diff(e(v), v)
            + 2 * M * sp.diff(e(v), v, 2)
            + 2 * sp.diff(e(v), v) ** 2
            + 2 * e(v) * sp.diff(e(v), v, 2)
        ),
    }
    rho, aa, dd = sp.symbols("rho aa dd", positive=True)
    s = sp.sqrt(dd + rho**2 * t)
    u = aa + s
    Z = 2 / u - 1
    checks["section_first_derivative"] = sp.diff(Z, t) + rho**2 / (s * u**2)
    checks["section_second_derivative"] = sp.diff(Z, t, 2) - rho**4 * (
        1 / (2 * s**3 * u**2) + 1 / (s**2 * u**3)
    )
    z = sp.symbols("z")
    K = 4 / rho**2 * ((1 + z) ** -2 - aa * (1 + z) ** -1)
    checks["inverse_first_derivative"] = sp.diff(K, z) - (
        4 * (aa * (1 + z) - 2) / (rho**2 * (1 + z) ** 3)
    )
    checks["inverse_second_derivative"] = sp.diff(K, z, 2) - (
        8 * (3 - aa * (1 + z)) / (rho**2 * (1 + z) ** 4)
    )
    U, G, V = sp.Function("U"), sp.Function("G"), sp.Function("V")
    composed = U(G(V(t)))
    chain = sp.diff(U(z), z, 2).subs(z, G(V(t))) * sp.diff(G(z), z).subs(z, V(t)) ** 2 * sp.diff(
        V(t), t
    ) ** 2 + sp.diff(U(z), z).subs(z, G(V(t))) * (
        sp.diff(G(z), z, 2).subs(z, V(t)) * sp.diff(V(t), t) ** 2
        + sp.diff(G(z), z).subs(z, V(t)) * sp.diff(V(t), t, 2)
    )
    checks["regular_coordinate_chain_rule"] = sp.diff(composed, t, 2) - chain
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def gate(*, rho: Q, delta: Q) -> dict[str, str | bool]:
    """An illustrative exact box; its derivative premises are inputs."""
    require(0 < rho <= Q(1, 4) and delta >= 0, "valid example inputs")
    vmin, vmax = Q(1, 8), Q(1, 2)
    amin, amax, bmin, bmax = Q(1), Q(2), Q(1, 3), Q(2, 3)
    ell = 1 - bmax * vmax
    M1, M2 = Q(1), Q(10)
    CH = 320 * M1**2 + 80 * M2 + 160 * M1
    A0, A1 = amax * vmax / ell, amax / ell**2
    A2 = 2 * amax * bmax / ell**3
    Q1 = 2 * A1 * delta + 2 * A0 * delta + 2 * delta**2
    Q2 = 2 * A2 * delta + 4 * A1 * delta + 2 * A0 * delta + 4 * delta**2
    regular = 4 * vmax * CH * rho**2
    entry = Q2 / vmin + Q1 / vmin**2
    margin = 6 * amin**2 * bmin  # Lmax <= 1 throughout this positive interval.
    return {
        "rho": str(rho),
        "entry_C2_input_bound": str(delta),
        "regular_term": str(regular),
        "entry_term": str(entry),
        "strict_margin": str(margin),
        "slack": str(margin - regular - entry),
        "sufficient_gate": regular + entry < margin,
    }


def sharpness_example() -> dict[str, object]:
    """Two roots for abstract maps, not a realization by the quadratic family."""

    # H(t)=-1/10+4t, M(v)=v/(1-2v/3); here H'' and the entry error vanish.
    def residual(v: Q) -> Q:
        return -Q(1, 10) + 4 * v**2 - (v / (1 - Q(2, 3) * v)) ** 2

    points = (Q(1, 8), Q(1, 2), Q(1))
    values = tuple(residual(v) for v in points)
    require(values[0] < 0 < values[1] and values[2] < 0, "two IVT sign changes")
    return {
        "scope": "abstract maps; IVT and the analytic criterion give exactly two roots",
        "points": [str(v) for v in points],
        "residuals": [str(v) for v in values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    good = gate(rho=Q(1, 1000), delta=Q(1, 100000))
    large_regular = gate(rho=Q(1, 10), delta=Q(0))
    large_entry = gate(rho=Q(1, 1000), delta=Q(1, 10))
    require(good["sufficient_gate"], "small-error gate passes")
    require(not large_regular["sufficient_gate"], "large regular error is inconclusive")
    require(not large_entry["sufficient_gate"], "large entry error is inconclusive")
    require(Q(512, 27) < 20 and Q(2048, 27) < 80, "inverse derivative constants")
    report = {
        "symbolic_checks": symbolic_checks(),
        "rational_gate_examples": [good, large_regular, large_entry],
        "sharpness_example": sharpness_example(),
        "honesty": {
            "finite_algebra_replayed": True,
            "example_error_bounds_measured_on_actual_field": False,
            "analytic_theorem_formally_verified": False,
            "full_graphic_cyclicity_proved": False,
            "hilbert16_solved": False,
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
