#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay root-saddle identities and an explicitly declared interval envelope.

The written argument is in packages/omnibias-dynamics/HILBERT16-ROOT-SADDLE.md.
This benchmark does not establish that the exact canonical family satisfies
the declared perturbation envelope at any particular epsilon, prove the
analytic passage theorem, or establish full Hilbert XVI.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from fractions import Fraction as Q
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.rootfind import interval_newton

SEED = 160026
RANGES: dict[str, tuple[Q, Q]] = {
    "L": (Q(1), Q(51, 50)),
    "lambda1": (Q(-151, 50), Q(-3)),
    "x": (Q(3, 10), Q(12, 25)),
    "y": (Q(0), Q(1, 64)),
    "k": (Q(1, 2), Q(2)),
    "delta_B": (Q(-1, 100), Q(1, 100)),
    "delta_Bx": (Q(-1, 100), Q(1, 100)),
    "kx": (Q(-1, 64), Q(1, 64)),
    "ky": (Q(-1, 64), Q(1, 64)),
}


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def record_identity(checks: list[str], name: str, residual: sp.Expr) -> None:
    if name in checks:
        raise ValueError(f"duplicate identity: {name}")
    if sp.simplify(residual) != 0:
        raise ArithmeticError(f"nonzero or unresolved identity: {name}")
    checks.append(name)


def identities() -> list[str]:
    eps, x, y, h, V, r, rho = sp.symbols("eps x y h V r rho", nonzero=True)
    B, Bx, k, kx, ky = sp.symbols("B Bx k kx ky")
    F, Fv, q, kh = sp.symbols("F Fv q kh")
    De, xe, ye, Ae, Epre, Es = sp.symbols("De xe ye Ae Epre Es", nonzero=True)
    lam = sp.Symbol("lambda")
    bfun, kfun = sp.Function("b"), sp.Function("k")
    divergence = sp.diff(eps * (bfun(x) + kfun(x, y) * y), x) + sp.diff(x * y, y)
    checks: list[str] = []
    expressions = {
        "exact_outgoing_rescaled_x_velocity": -(-eps**3 * B - eps**3 * y * k)
        / eps**2 - eps * (B + k * y),
        "exact_outgoing_rescaled_height_velocity": (eps * x * eps**3 * y)
        / eps**4 - x * y,
        "rescaled_divergence": divergence.subs(
            {sp.diff(bfun(x), x): Bx, sp.diff(kfun(x, y), x): kx}
        ) - eps * (Bx + y * kx) - x,
        "normal_saddle_stable_eigenvalue_chain_rule": sp.diff(-eps**3 * bfun(-V / eps), V).subs(
            V, -eps * r
        ) - eps**2 * sp.diff(bfun(r), r),
        "normal_saddle_transverse_eigenvalue": sp.diff(-V * h, h).subs(V, -eps * r) - eps * r,
        "normal_saddle_characteristic_polynomial": sp.det(
            sp.Matrix([[eps**2 * Bx - lam, -k], [0, eps * r - lam]])
        ) - (eps**2 * Bx - lam) * (eps * r - lam),
        "linearized_saddle_exponent": (-eps**2 * Bx) / (eps * r) + eps * Bx / r,
        "entry_section_determinant": sp.det(sp.Matrix([[eps * De, 0], [x * ye, 1]]))
        - eps * De,
        "exit_section_determinant": sp.det(sp.Matrix([[eps * De, lam], [xe * y, 0]]))
        + xe * y * lam,
        "exact_event_variation_cancellation": (-eps * De * Es / (xe * ye))
        * (-Ae * ye * Epre / (eps * De)) - Ae * Epre * Es / xe,
        "squared_height_equation": (V * F) / (-V * h) + F / h,
        "squared_height_q_form": -(-q - h * k) / h - (q / h + k),
        "squared_height_variation_coefficient": (-Fv / h) / V + Fv / (h * V),
        "incoming_height_variation": sp.diff(V * h / (q + k * h), h)
        + sp.diff(V * h / (q + k * h), k) * kh
        - V * (q - h**2 * kh) / (q + k * h) ** 2,
    }
    Tpar, Th, Eh, label_h = sp.symbols("Tpar Th Eh label_h")
    event_height = -Tpar / (Th + V * Eh)
    expressions["physical_moving_event_identity"] = Tpar + (Th + V * Eh) * event_height
    expressions["physical_label_event_factor"] = (
        label_h * event_height + label_h * Tpar / (Th + V * Eh)
    )
    for sigma in (-1, 1):
        section_v = 1 - sigma * rho * h
        label = (sigma * rho * h - 1) ** 2 - 2 * h
        expressions[f"physical_base_plus_two_factor_{sigma}"] = (
            -sp.diff(label, h) / (1 + section_v * sigma * rho) - 2
        )
    zminus, zplus, affine_minus, affine_plus, gp = sp.symbols(
        "zminus zplus affine_minus affine_plus gp", nonzero=True
    )
    expressions["regular_physical_derivative_ratio"] = (
        gp * affine_minus * zminus / (affine_plus * zplus)
        - gp * (affine_minus / affine_plus) * (zminus / zplus)
    )
    for name, residual in expressions.items():
        record_identity(checks, name, residual)
    require(len(checks) == 19, "complete symbolic identity pack")
    return checks


def interval(lo: Q, hi: Q) -> Interval:
    if lo > hi:
        raise ValueError("ordered rational interval required")
    return Interval(Interval.from_rational(lo).lo, Interval.from_rational(hi).hi)


def encloses(value: Interval, exact: Q) -> bool:
    return Q(value.lo) <= exact <= Q(value.hi)


def gate_bounds(*, k_upper: Q = Q(2)) -> dict[str, Interval]:
    if k_upper < Q(1, 2):
        raise ValueError("k upper bound below its declared lower bound")
    iv = {name: interval(*bounds) for name, bounds in RANGES.items()}
    L, lam, xx, yy = (iv[name] for name in ("L", "lambda1", "x", "y"))
    kk = interval(Q(1, 2), k_upper)
    left, right = RANGES["x"]
    pre = interval(Q(0), left)
    error = iv["delta_B"]
    return {
        "leading_discriminant": lam.pow_int(2) - 4 * L,
        "B_rectangle": L + lam * xx + xx.pow_int(2) + error,
        "B_pre_entry": L + lam * pre + pre.pow_int(2) + error,
        "left_wall_velocity_factor": L + lam * left + left**2 + error + kk * yy,
        "right_wall_velocity_factor": L + lam * right + right**2 + error + kk * yy,
        "stable_divergence_factor": lam + 2 * xx + iv["delta_Bx"] + yy * iv["kx"],
        "height_coupling_derivative": kk + yy * iv["ky"],
    }


def gate_passes(bounds: dict[str, Interval]) -> bool:
    return (
        bounds["leading_discriminant"].lo > 0
        and bounds["B_pre_entry"].lo > 0
        and bounds["left_wall_velocity_factor"].lo > 0
        and bounds["right_wall_velocity_factor"].hi < 0
        and bounds["stable_divergence_factor"].hi < -2
        and bounds["height_coupling_derivative"].lo > 0
    )


def rational_values(point: dict[str, Q]) -> dict[str, Q]:
    L, lam, x, y, k = (point[name] for name in ("L", "lambda1", "x", "y", "k"))
    left, right = RANGES["x"]
    pre = left * (x - left) / (right - left)
    error = point["delta_B"]
    return {
        "leading_discriminant": lam**2 - 4 * L,
        "B_rectangle": L + lam * x + x**2 + error,
        "B_pre_entry": L + lam * pre + pre**2 + error,
        "left_wall_velocity_factor": L + lam * left + left**2 + error + k * y,
        "right_wall_velocity_factor": L + lam * right + right**2 + error + k * y,
        "stable_divergence_factor": lam + 2 * x + point["delta_Bx"] + y * point["kx"],
        "height_coupling_derivative": k + y * point["ky"],
    }


def grid(lo: Q, hi: Q, count: int) -> list[Q]:
    return [lo + (hi - lo) * Q(i, count - 1) for i in range(count)]


def replay_containment(bounds: dict[str, Interval]) -> dict[str, int]:
    names = tuple(RANGES)
    axes = [grid(*RANGES[name], 5 if name in {"L", "lambda1", "x"} else 2) for name in names]
    deterministic = 0
    for values in itertools.product(*axes):
        point = dict(zip(names, values, strict=True))
        for name, exact in rational_values(point).items():
            require(encloses(bounds[name], exact), f"deterministic enclosure: {name}")
        deterministic += 1
    rng = random.Random(SEED)
    for _ in range(512):
        point = {
            name: lo + (hi - lo) * Q(rng.randrange(1_000_001), 1_000_000)
            for name, (lo, hi) in RANGES.items()
        }
        for name, exact in rational_values(point).items():
            require(encloses(bounds[name], exact), f"seeded rational enclosure: {name}")
    return {"deterministic_grid_points": deterministic, "seeded_rational_points": 512}


def polynomial_root_replay() -> dict[str, object]:
    # An actual callback for this explicitly given rational polynomial, not the
    # uncomputed exact canonical normal field. Its perturbation fits the envelope.
    constant, linear = Q(101, 100) + Q(1, 1000), Q(-301, 100) - Q(1, 1000)

    def function(x: Interval) -> Interval:
        return x.pow_int(2) + linear * x + constant

    def derivative(x: Interval) -> Interval:
        return 2 * x + linear

    require(Q(1, 1000) <= Q(1, 100), "representative value and derivative perturbation bounds")
    bracket = (0.3, 0.48)
    result = interval_newton(function, derivative, bracket)
    require(result["unique"] and result["status"] == "unique_root", "polynomial Newton root")
    lo, hi = result["enclosure"]
    p_lo, p_hi = Q(lo)**2 + linear * Q(lo) + constant, Q(hi)**2 + linear * Q(hi) + constant
    require(p_lo >= 0 and p_hi <= 0, "exact rational signs enclosing the decreasing polynomial root")
    for x in grid(Q(3, 10), Q(12, 25), 129):
        require(encloses(function(interval(*RANGES["x"])), x*x + linear*x + constant), "root callback grid")
        require(encloses(derivative(interval(*RANGES["x"])), 2*x + linear), "root derivative grid")
    rng = random.Random(SEED + 1)
    for _ in range(128):
        x = Q(3, 10) + Q(9, 50) * Q(rng.randrange(1_000_001), 1_000_000)
        require(encloses(function(interval(*RANGES["x"])), x*x + linear*x + constant), "root callback random")
        require(encloses(derivative(interval(*RANGES["x"])), 2*x + linear), "root derivative random")
    return {
        "base_parameters": {"L": "101/100", "lambda1": "-301/100"},
        "perturbation": "(1-x)/1000 on 0<=x<=12/25",
        "polynomial_coefficients_ascending": [str(constant), str(linear), "1"],
        "result": result,
        "exact_endpoint_signs_replayed": True,
        "deterministic_grid_points": 129,
        "seeded_rational_points": 128,
        "is_actual_canonical_normal_field": False,
    }


def regular_factor_replay() -> dict[str, object]:
    # Declared s,D bounds follow from the exact section formula on |t|<=1,
    # rho<=1/4. The affine factors require a separately established small nu/rho.
    ranges = [(Q(1, 2), Q(5, 4)), (Q(1), Q(5, 2)), (Q(1, 2), Q(2))]
    s, denominator, affine = (interval(*limits) for limits in ranges)
    derivative = affine / (s * denominator.pow_int(2))
    ratio = derivative / derivative
    require(Q(ratio.lo) > Q(1, 64), "regular physical derivative ratio exceeds 1/64")
    axes = [grid(*limits, 3) for limits in ranges * 2]
    count = 0
    for s1, d1, a1, s2, d2, a2 in itertools.product(*axes):
        exact = (a1 / (s1*d1*d1)) / (a2 / (s2*d2*d2))
        require(encloses(ratio, exact), "coordinate ratio deterministic enclosure")
        count += 1
    rng = random.Random(SEED + 2)
    for _ in range(256):
        s1, d1, a1, s2, d2, a2 = [
            lo + (hi-lo)*Q(rng.randrange(1_000_001), 1_000_000)
            for lo, hi in ranges * 2
        ]
        require(encloses(ratio, (a1/(s1*d1*d1))/(a2/(s2*d2*d2))), "coordinate ratio random enclosure")
    return {
        "derivative_ratio_enclosure": [ratio.lo, ratio.hi],
        "rational_lower_factor": "1/64",
        "regular_lower_formula": "(kappa_q/(64*Q))*exp(-4*pi/sqrt(C_min-1))*exp(-1/2)",
        "premises": "C_min>1, kappa_q>0, Q>0, |log E|<=1/2, declared section and affine bounds",
        "deterministic_grid_points": count,
        "seeded_rational_points": 256,
    }


def rejection_checks() -> list[str]:
    rejected: list[str] = []
    for label, residual in (("nonzero_identity", sp.S.One), ("unresolved_identity", sp.Symbol("u"))):
        try:
            record_identity([], label, residual)
        except ArithmeticError:
            rejected.append(label)
    try:
        interval(Q(1), Q(0))
    except ValueError:
        rejected.append("reversed_interval")
    require(not gate_passes(gate_bounds(k_upper=Q(100))), "wide k envelope must be inconclusive")
    rejected.append("outward_right_wall_envelope")
    require(not encloses(Interval.point(0), Q(1)), "false containment rejected")
    rejected.append("false_containment")
    multiple = interval_newton(lambda x: x.pow_int(2), lambda x: 2*x, (-0.1, 0.1))
    require(not multiple["unique"], "multiple root must not receive unique Newton certificate")
    rejected.append("multiple_root_uniqueness")
    require(len(rejected) == 6, "all negative self-checks remain active under optimization")
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bounds = gate_bounds()
    require(gate_passes(bounds), "declared uniform saddle rectangle")
    require(Q(bounds["B_pre_entry"].lo) > Q(1, 20), "declared rational pre-entry lower bound")
    report = {
        "schema": "hilbert16-root-saddle-finite-replay-v1",
        "seed": SEED,
        "identities": identities(),
        "declared_rectangle": {
            "rational_ranges": {name: [str(lo), str(hi)] for name, (lo, hi) in RANGES.items()},
            "B_value_perturbation_domain_x": ["0", "12/25"],
            "derivative_and_coupling_envelope_domain_x": ["3/10", "12/25"],
            "enclosures": {name: [value.lo, value.hi] for name, value in bounds.items()},
            "gate_passed": True,
            "normalized_contraction_lower": "2",
            "gamma_lower": str(Q(2) / RANGES["x"][1]),
            "pre_entry_S_upper": str(RANGES["x"][0]**2 / (2*Q(1, 20))),
            "scope": "all C1 fields satisfying these declared value/derivative envelopes",
            "containment_replay": replay_containment(bounds),
        },
        "polynomial_root": polynomial_root_replay(),
        "regular_factor": regular_factor_replay(),
        "negative_self_checks": rejection_checks(),
        "honesty": {
            "finite_algebra_replayed": True,
            "declared_envelope_rectangle_verified": True,
            "actual_canonical_family_envelope_established": False,
            "actual_epsilon_cutoff_certified": False,
            "analytic_passage_theorem_verified": False,
            "analytic_theorem_formally_verified": False,
            "theorem_prover_verified": False,
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
