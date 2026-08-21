# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet line-search algebra (theory 03-12): worked example, radius, Wolfe."""

from __future__ import annotations

import math

import pytest
from omnibias.core.line_search import (
    JetLineSearchConfig,
    apply_verification,
    certified_truncation_radius,
    lagrange_remainder_bound,
    poly_eval,
    polynomial_wolfe,
    resolve_trust_radius,
    run_model_line_search,
    select_model_step,
    taylor_coeffs_from_derivatives,
)

# Spec §5: phi^(k)(0) = (1, -2, 6, -12, 48) => p(s) = 1 - 2s + 3s^2 - 2s^3 + 2s^4
_QUARTIC_DERIVS = (1.0, -2.0, 6.0, -12.0, 48.0)


def test_worked_example_matches_spec() -> None:
    coeffs = taylor_coeffs_from_derivatives(_QUARTIC_DERIVS)
    assert coeffs == pytest.approx([1.0, -2.0, 3.0, -2.0, 2.0])
    result = select_model_step(
        _QUARTIC_DERIVS,
        radius=1.0,
        config=JetLineSearchConfig(order=4, verify=False, isolate_roots=True),
    )
    assert result.step == pytest.approx(0.409461, abs=5e-5)
    assert result.model_value == pytest.approx(0.602974, abs=5e-5)
    assert result.model_value < 1.0
    # Unique real stationary point of a cubic with negative discriminant.
    assert result.isolated_root or abs(result.step - 0.409461) < 5e-5


def test_g2_polynomial_remainder_is_zero() -> None:
    """Degree-4 model of a degree-4 polynomial: M = 0, radius is the cap."""
    radius = certified_truncation_radius(0.0, order=4, atol=1e-8, max_step=2.5)
    assert radius == 2.5
    bound = lagrange_remainder_bound(0.0, 2.5, 4)
    assert bound.hi == 0.0


def test_g2_exp_remainder_encloses_grid_and_sample() -> None:
    """On [0, r] for exp, |R_N| <= e^r r^{N+1}/(N+1)! is sound."""
    order = 4
    cap = 0.5
    m = math.exp(cap)
    radius = certified_truncation_radius(m, order, atol=1e-6, max_step=cap)
    assert 0.0 < radius <= cap
    enclosure = lagrange_remainder_bound(m, radius, order)
    assert enclosure.hi <= 1e-6 + 1e-15

    def remainder(s: float) -> float:
        # exp(s) - Taylor_N(s) at 0
        model = sum((s**k) / math.factorial(k) for k in range(order + 1))
        return abs(math.exp(s) - model)

    grid = [radius * i / 40.0 for i in range(41)]
    rng_vals = [radius * ((i * 17 + 3) % 97) / 97.0 for i in range(20)]
    for s in grid + rng_vals:
        err = remainder(s)
        assert err <= enclosure.hi + 1e-14


def test_polynomial_wolfe_contains_descent_step() -> None:
    coeffs = taylor_coeffs_from_derivatives(_QUARTIC_DERIVS)
    hull = polynomial_wolfe(coeffs, c1=1e-4, c2=0.9, bracket=(0.0, 1.0))
    assert hull is not None
    result = select_model_step(_QUARTIC_DERIVS, radius=1.0)
    assert hull.contains(result.step) or result.step == 0.0


def test_verify_never_worse_on_pathological_direction() -> None:
    # Model wants a large step; true phi rises immediately (ascent).
    derivs = (1.0, -1.0, 0.0, 0.0, 0.0)
    model = select_model_step(
        derivs,
        radius=1.0,
        config=JetLineSearchConfig(order=4, verify=False),
    )
    assert model.step > 0.0

    def actual(s: float) -> float:
        return 1.0 + 10.0 * s

    verified = apply_verification(model, actual, start_value=1.0)
    assert verified.fell_back
    assert verified.actual_value is not None
    assert verified.actual_value <= 1.0 + 1e-14
    assert verified.step >= 0.0


def test_run_model_requires_actual_when_verify() -> None:
    with pytest.raises(ValueError, match="actual_fn"):
        run_model_line_search(_QUARTIC_DERIVS, config=JetLineSearchConfig(order=4))


def test_config_rejects_bad_wolfe() -> None:
    with pytest.raises(ValueError, match="wolfe_c1"):
        JetLineSearchConfig(wolfe_c1=0.0)
    with pytest.raises(ValueError, match="wolfe_c2"):
        JetLineSearchConfig(wolfe_c1=0.5, wolfe_c2=0.4)


def test_explicit_cap_with_bound_is_certified() -> None:
    radius, certified = resolve_trust_radius(
        JetLineSearchConfig(order=4, trust_radius=1.0),
        next_derivative_bound=0.0,
    )
    assert certified
    assert radius == 1.0


def test_poly_eval_matches_hand() -> None:
    coeffs = [1.0, -2.0, 3.0, -2.0, 2.0]
    # p(1) = 1 - 2 + 3 - 2 + 2 = 2
    assert poly_eval(coeffs, 1.0) == pytest.approx(2.0)
    # p(0.5) = 1 - 1 + 0.75 - 0.25 + 0.125 = 0.625 (spec backtracking point)
    assert poly_eval(coeffs, 0.5) == pytest.approx(0.625)
