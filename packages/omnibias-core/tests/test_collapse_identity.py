# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Identity collapse: exact Q zero is {0}; a fat remainder is blocked."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    IDENTITY_SPEC,
    are_distinct,
    get_collapse,
    identity_collapse,
    remainder_collapse,
    reset_collapse_registry,
)
from omnibias.core.collapse.identity import difference_coeffs
from omnibias.core.verified.interval import Interval


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_identity_is_registered_and_distinct() -> None:
    assert get_collapse("identity") == IDENTITY_SPEC
    for name in ("bias", "temperature", "enclosure", "verdict"):
        assert are_distinct(IDENTITY_SPEC, get_collapse(name)).distinct


def test_binomial_square_is_an_identity() -> None:
    # (x+1)^2 = x^2 + 2x + 1
    verdict = identity_collapse((1, 2, 1), (1, 2, 1), Interval(-2.0, 2.0))
    assert verdict.proved
    assert verdict.outcome.surviving == "germ_identity"
    assert verdict.outcome.spec_name == "identity"
    assert verdict.outcome.honesty["identity_collapse"] is True
    assert verdict.outcome.honesty["founding_bias_collapse"] is False
    assert verdict.outcome.honesty["float_residual_is_proof"] is False


def test_nonzero_constant_difference_is_disproved() -> None:
    verdict = identity_collapse((1,), (2,), Interval(-1.0, 1.0))
    assert verdict.disproved


def test_x_on_a_symmetric_interval_is_blocked() -> None:
    verdict = identity_collapse((0, 1), (0,), Interval(-1.0, 1.0))
    assert verdict.blocked
    assert verdict.outcome.inconclusive


def test_remainder_at_full_degree_collapses() -> None:
    cubic = (1, -3, 2, 1)
    assert remainder_collapse(cubic, 3, Interval(-1.0, 1.0)).proved
    fat = remainder_collapse(cubic, 1, Interval(-1.0, 1.0))
    assert fat.blocked


def test_difference_coeffs_trim_trailing_zeros() -> None:
    assert difference_coeffs((1, 2, 1), (1, 2, 1, 0)) == ()
    assert difference_coeffs((1, 0, 1), (1,))[2] == 1


def test_negative_remainder_order_is_refused() -> None:
    with pytest.raises(ValueError, match="order must be"):
        remainder_collapse((1, 1), -1, Interval.point(0.0))
