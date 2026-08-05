# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Self-test of the oracle harness + differential soundness of transcendentals.

Doubles as a soundness sweep: the ``omnibias.core.verified`` interval
transcendentals must enclose an independent ~200-bit mpmath oracle on a dense
grid, random points, and a Hypothesis-driven property sweep.
"""

from __future__ import annotations

import math
import random

import mpmath
from _oracles import (
    assert_encloses,
    assert_superset,
    mp_value,
    oracle_exp,
    oracle_sigmoid,
    oracle_tanh,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from omnibias.core.verified import exp_iv, sigmoid_iv, tanh_iv
from omnibias.core.verified.interval import Interval


def test_oracle_brackets_known_values() -> None:
    assert_encloses(oracle_exp(0.0), 1.0, name="exp(0)")
    assert_encloses(oracle_tanh(0.0), 0.0, name="tanh(0)")
    assert_encloses(oracle_sigmoid(0.0), 0.5, name="sigmoid(0)")
    # exp(1) = e
    assert_encloses(oracle_exp(1.0), math.e, name="exp(1)~e")


def test_exp_iv_supersets_oracle_on_grid() -> None:
    xs = [i / 8.0 for i in range(-80, 81)]
    for x in xs:
        assert_superset(exp_iv(Interval.point(x)), oracle_exp(x), name=f"exp@{x}")


def test_tanh_sigmoid_iv_superset_oracle_random() -> None:
    rng = random.Random(0)
    for _ in range(500):
        x = rng.uniform(-12.0, 12.0)
        assert_superset(tanh_iv(Interval.point(x)), oracle_tanh(x), name=f"tanh@{x}")
        assert_superset(sigmoid_iv(Interval.point(x)), oracle_sigmoid(x), name=f"sigmoid@{x}")


@settings(max_examples=400, deadline=None)
@given(st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False))
def test_property_transcendentals_contain_true_value(x: float) -> None:
    # The strongest soundness check: the exact ~200-bit value lies inside.
    assert_encloses(exp_iv(Interval.point(x)), mp_value(mpmath.exp, x), name="exp")
    assert_encloses(tanh_iv(Interval.point(x)), mp_value(mpmath.tanh, x), name="tanh")
    assert_encloses(
        sigmoid_iv(Interval.point(x)),
        mp_value(lambda z: 1 / (1 + mpmath.e ** (-z)), x),
        name="sigmoid",
    )
