# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Matching-chart fold I-map Z_x bound; not sep>0 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.fold_z_x import (
    identity_verdicts,
    report,
    residual_fold_x_interior,
    residual_zx_chain,
    residual_zx_declared,
    z0_x,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.z_v_bound import z0_v


def test_fold_z_x_identities_are_exact() -> None:
    nu, v, v0 = Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
    assert residual_zx_chain(nu, v, v0) == 0
    assert residual_zx_declared() == 0
    assert residual_fold_x_interior() == 0
    assert z0_x(nu, v, v0) != 0
    assert z0_v(nu, v, v0) != 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_fold_z_x_bound_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-fold-z-x-v1"
    assert payload["z_x_bound"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["picard_included"] is True
    assert enclosure["below_declared"] is True
    assert enclosure["z_x_mag"] < 0.01
    honesty = payload["honesty"]
    assert honesty["z_x_bound"] is True
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "z_x_bound" not in reasons


def test_local_fold_z_x_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_fold_z_x")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
