# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Holomorphic Z_v bound; not fold Z_x, sep>0, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.z_v_bound import (
    identity_verdicts,
    q1_v,
    report,
    residual_q1_v,
    residual_wall_ratio_v,
    residual_zv_quotient,
    z0_v,
)


def test_z_v_identities_are_exact() -> None:
    nu, v, v0 = Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
    assert residual_q1_v(nu, v, v0) == 0
    assert residual_zv_quotient(nu, v, v0) == 0
    assert residual_wall_ratio_v(nu, v, v0) == 0
    assert q1_v(nu, v, v0) != 0
    assert z0_v(nu, v, v0) != 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_z_v_bound_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-z-v-bound-v1"
    assert payload["z_v_bound"] is True
    assert payload["z_x_bound"] is False
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["picard_included"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["below_declared"] is True
    assert enclosure["z_v_mag"] < 0.25
    honesty = payload["honesty"]
    assert honesty["z_v_bound"] is True
    assert honesty["z_x_bound"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["z_x_bound"]["reason"] == "unimplemented"
    assert "z_v_bound" not in reasons


def test_local_z_v_bound_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_z_v_bound")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
