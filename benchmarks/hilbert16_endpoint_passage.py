#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact endpoint projection identities and rational contraction checks.

The analytic argument is in
packages/omnibias-dynamics/HILBERT16-ENDPOINT-PASSAGE.md. These checks verify
finite algebra and rational inequalities, not the infinite analytic theorem.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as F
from pathlib import Path

import sympy as sp


def require_check(condition: object, label: str) -> None:
    """Keep every finite check active under Python optimization."""
    if not condition:
        raise ArithmeticError(label)


def boundary_identities() -> None:
    x, q, H, W, J, Jx, Bminus, Bplus = sp.symbols("x q H W J Jx Bminus Bplus")
    deltaB = Bplus - Bminus
    alpha_b = -deltaB / J
    v = H * (Bminus + deltaB * Jx / J)
    # H'=2xH/q, Jx'=W*x^2, and H*W=1/q fix the boundary projection sign.
    vprime = sp.diff(v, H) * (2 * x * H / q) + sp.diff(v, Jx) * W * x**2
    require_check(
        sp.factor((q * vprime - 2 * x * v + alpha_b * x**2).subs(W, 1 / (q * H))) == 0,
        "sp.factor((q * vprime - 2 * x * v + alpha_b * x ** 2).subs(W, 1 / (q * H))) == 0",
    )
    require_check(
        sp.factor(v.subs(Jx, 0) - H * Bminus) == 0, "sp.factor(v.subs(Jx, 0) - H * Bminus) == 0"
    )
    require_check(
        sp.factor(v.subs(Jx, J) - H * Bplus) == 0, "sp.factor(v.subs(Jx, J) - H * Bplus) == 0"
    )
    xx, w, C, eps, p, r, m, alpha = sp.symbols("xx w C eps p r m alpha")
    qx = (xx**2 + 2 * xx + C) / 2
    y = (xx**2 - C) / 2 + w
    P = (1 + alpha) * xx - y + xx**2 + eps * (p + r) * xx * y + eps**2 * m * y**2
    Q = C * xx + xx**2 + xx * y + eps * r * y**2
    slope = (Q - xx * P) / P
    expected = -(xx**2 * (qx + w) + eps * r * xx * y**2) / P**2
    require_check(
        sp.cancel(sp.diff(slope, alpha) - expected) == 0,
        "sp.cancel(sp.diff(slope, alpha) - expected) == 0",
    )
    # The weighted perturbation of the scalar variational coefficient.
    numerator = Q - xx * P
    d = P - qx
    T_w = sp.diff(numerator, w) - 2 * xx
    variational = -2 * xx * d / (qx * P) + T_w / P - numerator * sp.diff(P, w) / P**2
    require_check(
        sp.cancel(sp.diff(slope, w) - 2 * xx / qx - variational) == 0,
        "exact weighted variational quotient identity",
    )
    qminus = qx.subs(xx, -xx)
    require_check(
        sp.cancel((xx - 1) / qx + (-xx - 1) / qminus + (3 * xx**2 + C) / (qx * qminus)) == 0,
        "sp.cancel((xx - 1) / qx + (-xx - 1) / qminus + (3 * xx ** 2 + C) / (qx * qminus)) == 0",
    )


def rational_constants(cmin: F, cmax: F, L: F) -> dict[str, object]:
    require_check(1 < cmin <= cmax and L >= 1, "1 < cmin <= cmax and L >= 1")
    Q = cmax / 2
    kappa = min(F(1), cmin - 1) / 16
    W0 = kappa ** (-3)
    J0 = 8 / (9 * (cmax + 8) ** 3)
    A4 = 2 * W0 / J0
    B4 = 81 * Q**2 * W0 * (1 + A4 / 3)
    K = max(A4, B4)
    Kb = max(81 * Q**2 / kappa * (1 + W0 / (3 * J0)), 2 / (kappa * J0))
    F0, F1 = (L * (Q**2 + 2 * Q), L * Q**2)
    U0, A0 = (L * (6 + 3 * cmax), L * (cmax + 9))
    M = max(F(1), 2 * (Kb + K * (F0 + 1)), 2 * U0, 2 * A0)
    Y = Q + 1
    d0 = 2 * M + 2 * L * Y + L * Y**2
    z0 = 3 * M + F0 + F1
    b0 = L * (1 + cmax)
    t0 = b0 * M + L * M + 2 * L * Q + L
    e0 = 2 * t0 + 2 * d0 * z0 / kappa
    d1 = 2 + 2 * L + 2 * L * Y
    t1 = b0 + 4 * L + 2 * L * Q
    e1 = 2 * t1 + 2 * d1 * z0 / kappa + 6 * d0 / kappa + 2 * e0 * d1 / kappa
    eta = min(F(1), 1 / M, kappa / (2 * d0), 1 / (F1 + e0), 1 / (2 * K * e1))
    p0, n0 = (1 + 2 * L + 2 * L * Y, z0 + t0)
    Kw = 4 * d0 / kappa**2 + 2 * t1 / kappa + 4 * n0 * p0 / kappa**2
    Ealpha = 4 * L * Y**2 * kappa ** (-4)
    eta_t = min(eta, kappa / (2 * M), 1 / (4 * Kw), J0 / (36 * Ealpha))
    gamma = 4 * cmin / ((cmax + 3) * (cmax + 1))
    theta = 1 / (1 + gamma / 2)
    # Include the companion zero-endpoint sector in the final map theorem.
    M0 = max(F(1), 2 * K * (F0 + 1), 2 * U0, 2 * A0)
    d00 = 2 * M0 + 2 * L * Y + L * Y**2
    z00 = 3 * M0 + F0 + F1
    t00 = b0 * M0 + L * M0 + 2 * L * Q + L
    e00 = 2 * t00 + 2 * d00 * z00 / kappa
    e10 = 2 * t1 + 2 * d1 * z00 / kappa + 6 * d00 / kappa + 2 * e00 * d1 / kappa
    eta0 = min(F(1), 1 / M0, kappa / (2 * d00), (F0 + 1) / (F1 + e00), 1 / (2 * K * e10))
    eta_p = min(eta_t, eta0, gamma / (4 * Kw))
    constants = {
        "Q": Q,
        "kappa": kappa,
        "W0": W0,
        "J0": J0,
        "A4": A4,
        "B4": B4,
        "K": K,
        "K_b": Kb,
        "F0": F0,
        "F1": F1,
        "U0": U0,
        "A0": A0,
        "M_e": M,
        "Y": Y,
        "d0": d0,
        "z0": z0,
        "b0": b0,
        "t0": t0,
        "e0": e0,
        "d1": d1,
        "t1": t1,
        "e1": e1,
        "eta_e": eta,
        "p0": p0,
        "n0": n0,
        "K_w": Kw,
        "E_alpha": Ealpha,
        "eta_t": eta_t,
        "gamma": gamma,
        "theta": theta,
        "M_zero": M0,
        "d0_zero": d00,
        "z0_zero": z00,
        "t0_zero": t00,
        "e0_zero": e00,
        "e1_zero": e10,
        "eta_zero": eta0,
        "eta_p": eta_p,
    }
    gates = {
        "fraction_arithmetic": all(isinstance(value, F) for value in constants.values()),
        "positive_constants": all(value > 0 for value in constants.values()),
        "unit_sector": eta <= 1,
        "height_amplitude": M * eta <= 1,
        "positive_denominator": d0 * eta / kappa <= F(1, 2),
        "half_tube_image": Kb + K * F0 + K * eta * (F1 + e0) <= M / 2,
        "contraction_at_most_half": K * eta * e1 <= F(1, 2),
        "variational_kernel": 2 * Kw * eta_t <= F(1, 2),
        "integrated_shooting_source": 2 * eta_t * Ealpha <= J0 / 18,
        "transversality_height": M * eta_t / kappa <= F(1, 2),
        "zero_tube_inclusion": M0 <= M,
        "zero_denominator": d00 * eta0 / kappa <= F(1, 2),
        "zero_self_map": K * (F0 + eta0 * (F1 + e00)) < M0,
        "zero_contraction": K * eta0 * e10 <= F(1, 2),
        "common_zero_sector": eta_p <= eta0,
        "common_transversality_sector": eta_p <= eta_t,
        "strict_map_contraction": 0 < theta < 1,
        "map_variational_exponent": 2 * Kw * eta_p <= gamma / 2,
    }
    for label, condition in gates.items():
        require_check(condition, label)
    return {
        "parameter_box": {"C_min": str(cmin), "C_max": str(cmax), "L": str(L)},
        "exact_constants": {label: str(value) for label, value in constants.items()},
        "gates": gates,
        "scope": "Finite exact rational checks for the displayed compact parameter box.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hilbert16/endpoint_passage.json")
    )
    args = parser.parse_args()
    boundary_identities()
    output = {
        "status": "exact endpoint and transversality identities passed",
        "constant_checks": [
            rational_constants(F(2), F(3), F(1)),
            rational_constants(F(5, 4), F(7, 4), F(2)),
        ],
        "scope": "Finite symbolic identities and rational constant inequalities. The companion analytic proof is not formally verified; no singular closing-map or full-return zero bound is checked.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
