# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hilger exponential: group laws, the dynamic equation it solves, and the mu -> 0 limit."""

from __future__ import annotations

import math

import pytest
from omnibias.timescale import (
    circle_minus,
    circle_plus,
    cylinder,
    delta_derivative,
    h_integers,
    hilger_exponential,
    is_regressive,
    quantum,
    reals,
)


def test_hZ_product_form() -> None:
    H = h_integers(0.5)
    # e_p(2, 0) with constant p = 1 is (1 + 0.5)^4 (four steps).
    assert hilger_exponential(1.0, 2.0, 0.0, H) == pytest.approx(1.5**4)


def test_semigroup_law() -> None:
    H = h_integers(0.25)
    p = lambda t: 0.3 + 0.1 * t  # noqa: E731
    lhs = hilger_exponential(p, 2.0, 1.0, H) * hilger_exponential(p, 1.0, 0.0, H)
    rhs = hilger_exponential(p, 2.0, 0.0, H)
    assert lhs == pytest.approx(rhs)


def test_inverse_is_circle_minus() -> None:
    H = h_integers(0.5)
    p = 0.7
    # e_{ominus p}(t, s) = 1 / e_p(t, s).
    ominus_p = circle_minus(p, H)
    assert hilger_exponential(ominus_p, 2.0, 0.0, H) == pytest.approx(
        1.0 / hilger_exponential(p, 2.0, 0.0, H)
    )


def test_circle_plus_multiplies_exponentials() -> None:
    H = h_integers(0.2)
    p, q = 0.4, -0.3
    p_plus_q = circle_plus(p, q, H)
    lhs = hilger_exponential(p_plus_q, 1.0, 0.0, H)
    rhs = hilger_exponential(p, 1.0, 0.0, H) * hilger_exponential(q, 1.0, 0.0, H)
    assert lhs == pytest.approx(rhs)


def test_solves_dynamic_equation() -> None:
    # y(t) = e_p(t, 0) satisfies y^Delta = p y on the scale.
    H = h_integers(0.5)
    p = 0.6
    y = lambda t: hilger_exponential(p, t, 0.0, H)  # noqa: E731
    for t in (0.0, 0.5, 1.0, 1.5):
        assert delta_derivative(y, t, H) == pytest.approx(p * y(t))


def test_mu_to_zero_recovers_exp() -> None:
    # e_p(2, 0) with constant p on a fine mesh -> exp(2 p).
    p = 0.8
    errs = [abs(hilger_exponential(p, 2.0, 0.0, h_integers(h)) - math.exp(2.0 * p)) for h in (0.2, 0.05, 0.01)]
    assert errs == sorted(errs, reverse=True)
    # On R it is exactly exp.
    assert hilger_exponential(p, 2.0, 0.0, reals()) == pytest.approx(math.exp(2.0 * p))


def test_cylinder_transformation() -> None:
    assert cylinder(0.5, 0.0) == 0.5  # mu = 0 -> identity
    assert cylinder(0.5, 0.5) == pytest.approx(math.log(1.25) / 0.5)


def test_regressivity() -> None:
    H = h_integers(0.5)
    assert is_regressive(0.7, H, 0.0, 3.0)
    assert not is_regressive(-2.0, H, 0.0, 3.0)  # 1 + 0.5 * (-2) = 0
    assert is_regressive(-2.0, reals(), 0.0, 3.0)  # mu = 0 -> always regressive
    assert is_regressive(5.0, quantum(2.0), 1.0, 4.0)
