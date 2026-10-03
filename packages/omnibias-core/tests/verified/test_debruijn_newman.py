# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""De Bruijn-Newman Phi enclosure + named Lambda bound attempt (unearned)."""

from __future__ import annotations

import pytest
from omnibias.core.collapse.winding import integers_in
from omnibias.core.verified.debruijn_newman import (
    FAR_FIELD_CITATION,
    H0_FIRST_ZERO,
    PRE_REGISTERED_T0,
    FiniteHtRectanglePack,
    attempt_named_lambda_bound,
    declared_local_ht_rectangle_pack,
    finite_ht_rectangle_pack,
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


def test_finite_ht_pack_is_local_and_not_a_cover() -> None:
    pack = finite_ht_rectangle_pack(
        boxes=((2.0, 1.0),),
        half_height=0.2,
        truncation=1.0,
        phi_terms=4,
        panels=8,
        contour_segments=8,
        real_lo=0.0,
        real_hi=32.0,
    )
    assert pack.finite_cover_certified is False
    assert pack.rh_claim is False
    assert pack.n_certified + pack.n_blocked == 1
    assert len(pack.counts) == 1
    attempt = attempt_named_lambda_bound()
    assert attempt.certified is False
    assert attempt.finite_cover_certified is False
    assert "far-field" in attempt.missing_piece.lower()
    assert "cover" in attempt.missing_piece.lower()


def _count_for(pack: FiniteHtRectanglePack, center: float):
    matches = [
        item for item in pack.counts if item.contract.center.real == center
    ]
    assert len(matches) == 1
    return matches[0]


def test_declared_local_ht_pack_isolates_first_zero() -> None:
    """t=0 pack: one empty box, the first H_0 zero, and a boundary block."""
    pack = declared_local_ht_rectangle_pack()
    assert pack.t == 0.0
    assert pack.real_interval == (0.0, 32.0)
    assert pack.n_certified + pack.n_blocked == len(pack.counts) == 3
    assert pack.finite_cover_certified is False
    assert pack.rh_claim is False

    empty = _count_for(pack, 4.0)
    assert empty.contract.half_width == 1.0
    assert 0.0 <= 4.0 - 1.0 and 4.0 + 1.0 <= 32.0
    assert not (4.0 - 1.0 <= H0_FIRST_ZERO <= 4.0 + 1.0)
    assert empty.certified is True
    assert empty.count == 0
    assert empty.winding is not None
    assert integers_in(empty.winding) == (0,)

    zero = _count_for(pack, H0_FIRST_ZERO)
    assert zero.contract.half_width == 1.0
    assert zero.certified is True
    assert zero.count == 1
    assert zero.winding is not None
    assert integers_in(zero.winding) == (1,)

    blocked = _count_for(pack, H0_FIRST_ZERO - 1.0)
    assert blocked.contract.center.real + blocked.contract.half_width == H0_FIRST_ZERO
    assert blocked.certified is False
    assert blocked.count is None
    assert blocked.winding is None

    for item in pack.counts:
        if item.winding is None or len(integers_in(item.winding)) != 1:
            assert item.certified is False
            assert item.count is None
        else:
            assert item.certified is True
            assert item.count == integers_in(item.winding)[0]
    assert pack.n_certified == sum(item.certified for item in pack.counts)
    assert pack.n_blocked == sum(not item.certified for item in pack.counts)

    with pytest.raises(ValueError, match="finite_cover"):
        FiniteHtRectanglePack(
            t=pack.t,
            real_interval=pack.real_interval,
            counts=pack.counts,
            n_certified=pack.n_certified,
            n_blocked=pack.n_blocked,
            finite_cover_certified=True,
        )
    with pytest.raises(ValueError, match="rh_claim"):
        FiniteHtRectanglePack(
            t=pack.t,
            real_interval=pack.real_interval,
            counts=pack.counts,
            n_certified=pack.n_certified,
            n_blocked=pack.n_blocked,
            rh_claim=True,
        )

    attempt = attempt_named_lambda_bound()
    assert attempt.t0 == PRE_REGISTERED_T0 == 0.2
    assert attempt.t0 < 0.22
    assert attempt.certified is False
    assert attempt.finite_cover_certified is False
    assert attempt.far_field_premise_discharged is False
    assert attempt.rh_claim is False
    assert "far-field" in attempt.missing_piece.lower()
    assert "cover" in attempt.missing_piece.lower()
