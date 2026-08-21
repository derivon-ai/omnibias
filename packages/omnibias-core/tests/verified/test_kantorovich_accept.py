# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Kantorovich-accepted Newton (theory 08-04): unique-zero ball policy."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.proof.certificate import (
    THEOREM_PROVER_VERIFIED_KEY,
    verify_certificate_digest,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    CONTINUUM_PDE_CLAIM_KEY,
    FINITE_RESIDUAL_CLAIM,
    kantorovich_accept_step,
    polynomial_sqrt2_maps,
    select_accepted_params,
)

SQRT2 = math.sqrt(2.0)
GOOD_TRIAL = 1.5
FAR_TRIAL = 3.0
GOOD_A = [[1.0 / 3.0]]
R_MAX = 0.2


def test_g1_accept_inside_named_ball() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [GOOD_TRIAL], lipschitz_df=lip, r_max=R_MAX
    )
    assert decision.accepted is True
    assert decision.reason == "ball"
    assert decision.certificate is not None
    assert abs(SQRT2 - GOOD_TRIAL) <= decision.certificate.radius
    assert decision.certificate.radius <= R_MAX


def test_g1_reject_named_far_point() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [FAR_TRIAL], lipschitz_df=lip, r_max=R_MAX
    )
    assert decision.accepted is False
    assert decision.reason == "empty"
    assert decision.certificate is None
    kept = select_accepted_params([0.0], [FAR_TRIAL], decision)
    assert kept == [0.0]


def test_g2_never_continuum() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [GOOD_TRIAL], lipschitz_df=lip, r_max=R_MAX
    )
    assert decision.certificate is not None
    sealed = decision.certificate.certificate
    assert verify_certificate_digest(sealed)
    payload = sealed["payload"]
    honesty = sealed["honesty"]
    assert payload[CONTINUUM_PDE_CLAIM_KEY] is False
    assert honesty[CONTINUUM_PDE_CLAIM_KEY] is False
    assert payload["finite_map"] is True
    claim = sealed["claim"]
    assert claim == FINITE_RESIDUAL_CLAIM
    assert "finite-dimensional" in claim
    assert "continuum PDE" in claim


def test_g3_no_forge_theorem_prover_verified() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [GOOD_TRIAL], lipschitz_df=lip, r_max=R_MAX
    )
    assert decision.certificate is not None
    honesty = decision.certificate.certificate["honesty"]
    assert THEOREM_PROVER_VERIFIED_KEY not in honesty


def test_interval_map_contains_grid_and_random() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    rng = random.Random(0)
    xs = [GOOD_TRIAL + 0.01 * k for k in range(-20, 21)]
    xs.extend(rng.uniform(1.3, 1.7) for _ in range(50))
    for x in xs:
        true_f = x * x - 2.0
        box = Interval.from_value(func([Interval.point(x)])[0])
        assert box.lo <= true_f <= box.hi
        true_df = 2.0 * x
        jbox = Interval.from_value(jac([Interval.point(x)])[0][0])
        assert jbox.lo <= true_df <= jbox.hi
        y = x + 1e-3
        slope = abs(2.0 * x - 2.0 * y) / abs(x - y)
        assert slope <= lip + 1e-15


def test_accepted_ball_has_unique_sqrt2_root() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [GOOD_TRIAL], lipschitz_df=lip, r_max=R_MAX
    )
    assert decision.certificate is not None
    radius = decision.certificate.radius
    lo, hi = GOOD_TRIAL - radius, GOOD_TRIAL + radius
    assert lo <= SQRT2 <= hi
    assert not (lo <= -SQRT2 <= hi)
    f_lo = lo * lo - 2.0
    f_hi = hi * hi - 2.0
    assert f_lo * f_hi <= 0.0


def test_mismatched_a_is_bounds_failed() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func,
        jac,
        [[1.0, 0.0], [0.0, 1.0]],
        [GOOD_TRIAL],
        lipschitz_df=lip,
        r_max=R_MAX,
    )
    assert decision.accepted is False
    assert decision.reason == "bounds_failed"


def test_nonfinite_trial_is_bounds_failed() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [math.nan], lipschitz_df=lip, r_max=R_MAX
    )
    assert decision.reason == "bounds_failed"


def test_r_max_must_be_positive() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    with pytest.raises(ValueError, match="r_max"):
        kantorovich_accept_step(
            func, jac, GOOD_A, [GOOD_TRIAL], lipschitz_df=lip, r_max=0.0
        )


def test_select_keeps_trial_on_accept() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, GOOD_A, [GOOD_TRIAL], lipschitz_df=lip, r_max=R_MAX
    )
    chosen = select_accepted_params([0.25], [GOOD_TRIAL], decision)
    assert chosen == [GOOD_TRIAL]
