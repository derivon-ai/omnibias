#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact symbolic replay for the local invariant-conic branch classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as s


def symbolic_audit() -> dict[str, object]:
    x, y, A, C, p, r, h, a, b, d, e, f, L, M, N, t, k, m = s.symbols(
        "x y A C p r h a b d e f L M N t k m"
    )

    def zero(expression, label):
        value = s.factor(expression)
        if value != 0:
            raise ArithmeticError(f"{label}: {value}")

    P = A * x - y + x * x + (p + r) * x * y + h * y * y
    Q = C * x + x * x + x * y + r * y * y
    F = a * x * x + b * x * y + d * y * y + e * x + y + f
    res = s.Poly(s.expand(s.diff(F, x) * P + s.diff(F, y) * Q - (L * x + M * y + N) * F), x, y)
    base = {
        A: 1,
        p: 0,
        r: 0,
        h: 0,
        a: -s.Rational(1, 2),
        b: 0,
        d: 0,
        e: 0,
        f: C / 2,
        L: 2,
        M: 0,
        N: 0,
    }
    sel = [v for ij, v in res.terms() if ij not in ((1, 2), (0, 3))]
    det8 = s.factor(s.Matrix(sel).jacobian([a, b, d, e, f, L, M, N]).subs(base).det())
    zero(det8 + 2 * C * (C + 3), "eight-equation conic Jacobian")
    D = (M - 2 * p - 2 * r + t * t) / 2
    H = t * (3 * M + t * t - 4 * p - 6 * r) / 4
    zero(
        D * (2 * r - M) + t * H - (-M + t * t + 2 * r) * (2 * M + t * t - 4 * p - 4 * r) / 4,
        "cubic factorization",
    )
    branch_data = []
    for branch, mm in [("I", t * t + 2 * r), ("II", 2 * p + 2 * r - t * t / 2)]:
        dd = s.factor(D.subs(M, mm))
        hh = s.factor(H.subs(M, mm))
        eq = [
            a * t + mm * hh * f + mm - r,
            2 * A * a + C * a * t + (1 + t) * mm * f + 1,
            a * (A * t + 2 * C * dd - 2) + mm * f * (mm - p - r) - t - 1,
            f * (2 + t + A * mm) - C,
        ]
        jbase = {t: 0, p: 0, r: 0, a: -s.Rational(1, 2), f: C / 2, A: 1}
        jac = s.Matrix(eq).jacobian([t, a, f, A]).subs(jbase)
        zero(jac.det() + 2, branch + " reduced determinant")
        # All ten original residuals reduce exactly to the four displayed lower equations.
        rr = {L: 2 + t, M: mm, N: 0, b: a * t, d: a * dd, e: -mm * f, h: hh}
        rrpoly = s.Poly(s.expand(res.as_expr().subs(rr, simultaneous=True)), x, y)
        for ij, z in rrpoly.terms():
            expected = {(2, 0): eq[1], (1, 1): eq[2], (1, 0): -eq[3], (0, 2): -eq[0]}.get(
                ij, s.S(0)
            )
            zero(z - expected, branch + str(ij) + " reduced equation")
        branch_data.append((branch, dd, hh, eq))

    K = C * m * m - 6 * C * m * k * k - 2 * C * m * k + 2 * m + 12 * k * k + 16 * k + 4
    ff = 2 * C * (1 + 3 * k) / K
    aa = -(1 + 3 * k) * (m + 6 * k * k + 8 * k + 2) / K
    AA = (2 + C * m - 2 * C * k * (3 * k + 1)) / (2 * (1 + 3 * k))
    pp = (-m + 8 * k * k + 2 * k) / 2
    rr = m - 3 * k * k - k
    hh = k * (-m + 6 * k * k + 2 * k) / 2
    param = {
        A: AA,
        p: pp,
        r: rr,
        h: hh,
        a: aa,
        b: 2 * k * aa,
        d: k * k * aa,
        e: -ff * m,
        f: ff,
        L: 2 * (1 + k),
        M: m,
        N: 0,
    }
    for ij, z in res.terms():
        zero(z.subs(param, simultaneous=True), "rational parabola parameter " + str(ij))
    zero(2 * pp + rr - k - 5 * k * k, "parabola local inverse")
    # Easy exact section of the nonparabolic branch.
    section = {
        A: 1 + C * p,
        r: 0,
        h: 0,
        a: -1 / (2 * (1 + C * p)),
        b: 0,
        d: p / (2 * (1 + C * p)),
        e: 0,
        f: C / 2,
        L: 2,
        M: 0,
        N: 0,
    }
    for ij, z in res.terms():
        zero(z.subs(section, simultaneous=True), "branch I r=0 section " + str(ij))
    # The branch-I elimination factorization, verified after clearing denominators.
    branch_one_equations = branch_data[0][3]
    lin = s.expand(2 * branch_one_equations[2] - t * branch_one_equations[1])
    mat, vec = s.linear_eq_to_matrix([branch_one_equations[0], lin], [a, f])
    sol = mat.inv() * vec
    aI, fI = map(s.factor, sol)
    AI = s.factor(-(C * aI * t + (1 + t) * (t * t + 2 * r) * fI + 1) / (2 * aI))
    resI = s.factor(fI * (2 + t + AI * (t * t + 2 * r)) - C)
    B = (
        2 * C * t**5
        + 2 * C * t**4
        - 3 * C * t**3 * p
        + 3 * C * t**3 * r
        - 2 * C * t**2 * p
        + 4 * C * t**2 * r
        - 6 * C * t * p * r
        - 2 * C * t * r * r
        - 4 * C * p * r
        - 2 * t**3
        + 3 * t * t
        - 8 * t * r
        + 2 * t
        - 4 * r
    )
    F1 = C * t**4 - C * t * t * p + C * t * t * r + C * r * r + t * t + t + r
    numer, denom = s.fraction(resI)
    zero(numer - (3 * t * t - 4 * p) * F1 * B, "branch I eliminated numerator")
    zero(s.diff(B, t).subs({t: 0, p: 0, r: 0}) - 2, "branch I scalar local uniqueness")

    # Nondegeneracy of the projective conic and the local parameter inverse.
    conic_matrix = s.Matrix(
        [[a, b / 2, e / 2], [b / 2, d, s.Rational(1, 2)], [e / 2, s.Rational(1, 2), f]]
    )
    zero(conic_matrix.det().subs(base) - s.Rational(1, 8), "base projective nonsingularity")
    zero(
        s.Matrix([pp, rr]).jacobian([k, m]).subs({k: 0, m: 0}).det() - s.Rational(1, 2),
        "parabola parameter inverse",
    )

    # The branch intersection satisfies the full original equations via the
    # exact rational parabola, as well as both cubic factors.
    ki = t / 2
    mi = t + t * t / 2
    pi = 3 * t * t / 4
    ri = t / 2 - t * t / 4
    hi = t**3 / 4
    Ai = (1 - C * t * t / 2) / (1 + 3 * t / 2)
    for expression, expected, label in [(pp, pi, "p"), (rr, ri, "r"), (hh, hi, "h"), (AA, Ai, "A")]:
        zero(expression.subs({k: ki, m: mi}) - expected, "intersection " + label)
    zero(mi - t * t - 2 * ri, "intersection first cubic factor")
    zero(2 * mi + t * t - 4 * pi - 4 * ri, "intersection second cubic factor")

    # Independent finite Taylor checks from the exact rational parametrization.
    eps, u, v = s.symbols("eps u v")
    ss = 2 * u + v
    ks = ss * eps - 5 * ss**2 * eps**2 + 50 * ss**3 * eps**3
    ms = 2 * eps * (u + v) - 2 * ks**2
    As = s.series(AA.subs({k: ks, m: ms}, simultaneous=True), eps, 0, 4).removeO()
    hs = s.series(hh.subs({k: ks, m: ms}, simultaneous=True), eps, 0, 4).removeO()
    A_expected = 1 - eps * (C * u + 3 * ss) + eps**2 * ((C + 24) * ss**2 + 3 * C * u * ss)
    A_expected -= eps**3 * ((13 * C + 267) * ss**3 + 24 * C * u * ss**2)
    h_expected = eps**2 * u * ss - eps**3 * (5 * u + ss) * ss**2
    zero(s.expand(As - A_expected), "parabola A cubic expansion")
    zero(s.expand(hs - h_expected), "parabola h cubic expansion")
    return {
        "status": "all exact conic reductions, determinants, parameterizations and parabola expansions passed",
        "reduced_jacobian_determinant": "-2",
        "fixed_parameter_conic_jacobian": str(det8),
        "projective_base_determinant": "1/8",
        "parabola_parameter_jacobian": "1/2",
        "branch_I_scalar_equation": str(B),
        "scope": "Finite exact algebra. The local analytic classification uses the written implicit-function argument; no formal analytic verification or cyclicity claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hilbert16/invariant-conics.json")
    )
    args = parser.parse_args()
    result = symbolic_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
