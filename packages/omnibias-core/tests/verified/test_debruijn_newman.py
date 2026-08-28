# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""De Bruijn-Newman Phi enclosure + named Lambda bound attempt (unearned)."""

from __future__ import annotations

import pytest
from omnibias.core.verified.debruijn_newman import (
    FAR_FIELD_CITATION,
    H0_FIRST_ZERO,
    PRE_REGISTERED_T0,
    attempt_named_lambda_bound,
    phi_enclosure,
)
from omnibias.core.verified.interval import Interval


def test_phi_at_zero_is_strictly_positive() -> None:
    enclosure = phi_enclosure(Interval.point(0.0), 6)
    assert enclosure.lo > 0.0


def test_h0_first_zero_is_twice_the_riemann_zero() -> None:
    assert H0_FIRST_ZERO == pytest.approx(28.269450283469387, rel=0, abs=1e-9)


def test_named_lambda_attempt_is_unearned_and_not_rh() -> None:
    result = attempt_named_lambda_bound()
    assert result.t0 == PRE_REGISTERED_T0
    assert result.t0 < 0.22
    assert result.certified is False
    assert result.far_field_premise_discharged is False
    assert result.rh_claim is False
    assert "far-field" in result.missing_piece.lower()
    assert "Polymath" in result.far_field_citation or "polymath" in FAR_FIELD_CITATION.lower()
    assert result.first_zero_crosscheck is True


def test_named_lambda_refuses_t0_at_or_above_022() -> None:
    with pytest.raises(ValueError, match="t0"):
        attempt_named_lambda_bound(t0=0.22)
