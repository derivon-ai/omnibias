#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact algebra for a quadratic logarithmic compensation obstruction.

Run with the workspace Python; --numeric also checks convergence by mpmath
quadrature.  Numerical quadrature is diagnostic, not a certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


def require_check(condition: object, label: str) -> None:
    """Keep exact obligations active even when Python runs with -O."""
    if not condition:
        raise ArithmeticError(label)


def exact_checks() -> dict[str, str]:
    x, y, w, eps = sp.symbols("x y w eps")
    C, p, r, m1, alpha2 = sp.symbols("C p r m1 alpha2")
    alpha1 = -(C + 6) * p - 3 * r
    q = (x**2 + 2 * x + C) / 2
    y0 = (x**2 - C) / 2
    P = (1 + eps * alpha1 + eps**2 * alpha2) * x - y + x**2
    P += eps * (p + r) * x * y + eps**2 * m1 * y**2
    Q = C * x + x**2 + x * y + eps * r * y**2
    denominator = sp.expand(P.subs(y, y0 + w))
    numerator = sp.expand((Q - x * P).subs(y, y0 + w))
    b = -alpha1 * x**2 - (p + r) * x**2 * y0 + r * y0**2
    B = -p * x**2 - C * r
    h = alpha1 * x + (p + r) * x * y0
    b1 = -x * (x**2 - C) ** 2 / 4
    require_check(
        sp.expand(numerator.subs({m1: 0, alpha2: 0}) - 2 * x * w - eps * (b + B * w + r * w**2))
        == 0,
        "exact identity check at source line 30",
    )
    require_check(
        sp.expand(denominator.subs({m1: 0, alpha2: 0}) - q + w - eps * (h + (p + r) * x * w)) == 0,
        "exact identity check at source line 34",
    )

    U = p * (x**3 + 3 * x**2 + 3 * C / 2)
    U += r * (x**3 / 2 + 3 * x**2 / 2 + C * x / 2 + C)

    def L(expression):
        return sp.expand(q * sp.diff(expression, x) - 2 * x * expression)

    require_check(sp.expand(L(U) - b) == 0, "exact identity check at source line 42")
    Palpha = -(x**4 + 8 * x**3 + 2 * (C + 12) * x**2 + C * (C + 12))
    Palpha /= 16 * (C + 3)
    require_check(sp.factor(L(Palpha) + x**2) == 0, "exact identity check at source line 45")
    F2 = sp.expand((U - h) * sp.diff(U, x) + B * U)
    u, v, up, vp = sp.symbols("u v up vp")
    residual = sp.expand(
        denominator.subs(w, eps * u + eps**2 * v) * (eps * up + eps**2 * vp)
        - numerator.subs(w, eps * u + eps**2 * v)
    )
    require_check(
        sp.expand(residual.coeff(eps, 1) - (q * up - 2 * x * u - b)) == 0,
        "exact identity check at source line 52",
    )
    expected2 = q * vp - 2 * x * v - ((u - h) * up + B * u)
    expected2 -= m1 * b1 - alpha2 * x**2
    require_check(
        sp.expand(residual.coeff(eps, 2) - expected2) == 0, "exact identity check at source line 55"
    )
    beta = p * (2 * p + r)
    require_check(
        sp.factor(F2.coeff(x, 5) - beta / 4) == 0, "exact identity check at source line 57"
    )
    require_check(
        sp.factor(sp.limit(x * (F2 + m1 * b1) / q**3, x, sp.oo) - 2 * (beta - m1)) == 0,
        "exact identity check at source line 58",
    )

    # A polynomial normal-form identity, independently of the asymptotic proof.
    k = -(2 * p + r) * ((25 * C + 108) * p + (11 * C + 54) * r)
    V = -C * (5 * C * p**2 + 5 * C * p * r + C * r**2 + 99 * p**2 + 105 * p * r + 28 * r**2) / 2
    V -= C * r * (3 * p + 2 * r) * x / 2
    V -= (
        (11 * C * p**2 + 9 * C * p * r + 2 * C * r**2 + 180 * p**2 + 180 * p * r + 45 * r**2)
        * x**2
        / 2
    )
    V -= 9 * (2 * p + r) ** 2 * x**3 / 2
    require_check(
        sp.expand(F2 + beta * b1 - L(V) + k * x**2) == 0, "exact identity check at source line 69"
    )

    # Exact invariant parabola at mu1=0, mu2=t, mu3=-2t, A=1-C*t.
    t = sp.symbols("t")
    d0 = 1 - t + C * t**2
    conic = (t - 1) * x**2 + 2 * C * t * x + 2 * d0 * y + C
    Pt = (1 - C * t) * x - y + x**2 - t * x * y
    Qt = C * x + x**2 + x * y - 2 * t * y**2
    require_check(
        sp.expand(sp.diff(conic, x) * Pt + sp.diff(conic, y) * Qt - (2 * x - 2 * t * y) * conic)
        == 0,
        "exact identity check at source line 77",
    )

    # Necessary obstruction for ANY analytic normalized invariant-conic family.
    a, bb, d, e, f, ell, mm, nn, A, pp, rr = sp.symbols("a bb d e f ell mm nn A pp rr")
    conic_general = a * x**2 + bb * x * y + d * y**2 + e * x + y + f
    fieldP = A * x - y + x**2 + (pp + rr) * x * y
    fieldQ = C * x + x**2 + x * y + rr * y**2
    darboux = sp.Poly(
        sp.expand(
            sp.diff(conic_general, x) * fieldP
            + sp.diff(conic_general, y) * fieldQ
            - (ell * x + mm * y + nn) * conic_general
        ),
        x,
        y,
    )
    a1, b_first, d1, e1, f1, l1, m_first, n1 = sp.symbols("a1 b_first d1 e1 f1 l1 m_first n1")
    first_sub = {
        a: -sp.Rational(1, 2) + eps * a1,
        bb: eps * b_first,
        d: eps * d1,
        e: eps * e1,
        f: C / 2 + eps * f1,
        ell: 2 + eps * l1,
        mm: eps * m_first,
        nn: eps * n1,
        A: 1 + eps * alpha1,
        pp: eps * p,
        rr: eps * r,
    }
    first_eqs = [sp.expand(value.subs(first_sub)).coeff(eps, 1) for value in darboux.coeffs()]
    first_sol = sp.solve(first_eqs, [a1, b_first, d1, e1, f1, l1, m_first, n1], dict=True)[0]
    require_check(
        sp.factor(first_sol[b_first] + 2 * p + r) == 0, "exact identity check at source line 98"
    )
    require_check(first_sol[d1] == 0, "exact identity check at source line 99")
    # With n=0, m=rr-bb and ell=2+bb/a, the xy^2 coefficient is
    # bb*(bb+pp+rr) - d*bb/a. Its second-order term is beta.
    obstruction = (bb * (bb + pp + rr) - d * bb / a).subs(first_sub)
    require_check(
        sp.factor(sp.limit(obstruction / eps**2, eps, 0).subs(first_sol) - beta) == 0,
        "exact identity check at source line 103",
    )
    return {
        "status": "all exact identities passed",
        "first_particular_U": str(U),
        "h": str(h),
        "B": str(B),
        "second_forcing_F2": str(sp.factor(F2)),
        "logarithmic_factor": str(beta - m1),
        "normal_form_k": str(k),
        "normal_form_V": str(V),
        "exact_invariant_conic": str(conic),
        "conic_cofactor": str(2 * x - 2 * t * y),
    }


def numeric_diagnostic() -> list[dict[str, float | int]]:
    import mpmath as mp

    with mp.workdps(40):
        C, p, r = mp.mpf(2), mp.mpf(1), mp.mpf(0)
        a = mp.sqrt(C - 1)
        c = mp.exp(-4 * mp.pi / a)

        def q(x):
            return (x * x + 2 * x + C) / 2

        def D(x):
            return mp.exp(4 * (mp.atan((x + 1) / a) - mp.pi / 2) / a)

        def U(x):
            return p * (x**3 + 3 * x * x + 3 * C / 2) + r * (
                x**3 / 2 + 3 * x * x / 2 + C * x / 2 + C
            )

        def Up(x):
            return p * (3 * x * x + 6 * x) + r * (3 * x * x / 2 + 3 * x + C / 2)

        alpha = -(C + 6) * p - 3 * r

        def h(x):
            return alpha * x + (p + r) * x * (x * x - C) / 2

        def B(x):
            return -p * x * x - C * r

        rows = []
        for R in (20, 80, 320, 1280):
            K = -D(-R) * U(-R) / q(-R) ** 2

            def weighted(x, K=K):
                H = q(x) ** 2 / D(x)
                Hp = 2 * x * H / q(x)
                u, up = U(x) + K * H, Up(x) + K * Hp
                return D(x) * ((u - h(x)) * up + B(x) * u) / q(x) ** 3

            integral = mp.quad(weighted, [-R, -10, -1, 0, 10, R])
            g2 = q(R) * integral / D(R)
            rows.append(
                {
                    "R": R,
                    "g2_over_R2_log_R": float(g2 / (R**2 * mp.log(R))),
                    "proved_limit": float(p * (2 * p + r) * (1 - c)),
                }
            )
        return rows


def coefficients(cv, pv, rv, mv, R):
    import mpmath as mp

    with mp.workdps(65):
        cv, pv, rv, mv, R = map(mp.mpf, (cv, pv, rv, mv, R))
        root = mp.sqrt(cv - 1)

        def qv(z):
            return (z * z + 2 * z + cv) / 2

        def D(z):
            return mp.exp(4 / root * (mp.atan((z + 1) / root) - mp.pi / 2))

        def PA(z):
            return -(z**4 + 8 * z**3 + 2 * (cv + 12) * z * z + cv * (cv + 12)) / (16 * (cv + 3))

        def P2(z):
            return (cv + 6) * PA(z) + z**3 + 3 * z * z + 3 * cv / 2

        def P3(z):
            return 3 * PA(z) + z**3 / 2 + 3 * z * z / 2 + cv * z / 2 + cv

        dl, dr, ql, qr = D(-R), D(R), qv(-R), qv(R)

        def g(P):
            return (P(R) - dl / dr * qr**2 * P(-R) / ql**2) / qr

        ga = g(PA)
        a1 = -(pv * g(P2) + rv * g(P3)) / ga

        def PP(z):
            return a1 * PA(z) + pv * P2(z) + rv * P3(z)

        K = -dl * PP(-R) / ql**2

        def integrand(z):
            qq, dd = qv(z), D(z)
            yy = (z * z - cv) / 2
            u = PP(z) + K * qq**2 / dd
            bb = -a1 * z * z - (pv + rv) * z * z * yy + rv * yy * yy
            up = (2 * z * u + bb) / qq
            hh = a1 * z + (pv + rv) * z * yy
            BB = -pv * z * z - cv * rv
            source = (u - hh) * up + BB * u - mv * z * (z * z - cv) ** 2 / 4
            return dd * source / qq**3

        val = mp.quad(integrand, [-R, -4, -1, 0, 4, R])
        a2 = -qr / dr * val / ga
        logcoeff = 8 * (cv + 3) * (pv * (2 * pv + rv) - mv)
        return {
            "R": float(R),
            "a1": float(a1),
            "a2": float(a2),
            "predicted_log_coefficient": float(logcoeff),
            "a2_minus_log": float(a2 - logcoeff * mp.log(R)),
        }


def compensation_diagnostic():
    """High-precision quadrature, without a claimed interval enclosure."""
    return [
        {
            "parameters": parameters,
            "cutoffs": [coefficients(*parameters, cutoff) for cutoff in (16, 64, 256, 1024)],
        }
        for parameters in ((2, 1, 0, 0), (2, 1, 0, 2), (2, 1, -2, 0))
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hilbert16/compensation.json")
    )
    args = parser.parse_args()
    output: dict[str, object] = exact_checks()
    output["scope"] = (
        "Exact polynomial identities; the accompanying analytic argument is not formally verified."
    )
    if args.numeric:
        output["noncertifying_fixed_direction_quadrature"] = numeric_diagnostic()
        output["noncertifying_actual_compensation_quadrature"] = compensation_diagnostic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
