# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Slow-line Z_V chain plus fold holomorphic Z_v; not fold Z_x or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.z_slow_v import (
    V_v,
    ell,
    identity_verdicts,
    report,
    residual_ell_min,
    residual_V_v_slow,
    residual_ZV_chain,
    z0_V,
)
from omnibias.dynamics.z_v_bound import z0_v


def test_z_slow_v_identities_are_exact() -> None:
    nu, v, v0 = Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
    assert residual_V_v_slow(nu, v) == 0
    assert residual_ZV_chain(nu, v, v0) == 0
    assert residual_ell_min() == 0
    assert V_v(nu, v) + ell(nu, v) == 0
    assert z0_V(nu, v, v0) * ell(nu, v) + z0_v(nu, v, v0) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_z_slow_v_bound_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-z-slow-v-v1"
    assert payload["z_slow_v_bound"] is True
    assert payload["z_x_bound"] is False
    assert payload["sample_inside"] is True
    assert payload["fold_sample_inside"] is True
    kill = payload["kill_enclosure"]
    assert kill["picard_included"] is True
    assert kill["excludes_zero"] is True
    assert kill["below_declared"] is True
    assert kill["z_V_mag"] < 0.25
    fold = payload["fold_enclosure"]
    assert fold["picard_included"] is True
    assert fold["excludes_zero"] is True
    assert fold["below_declared"] is True
    assert fold["z_v_mag"] < 0.25
    assert fold["zV_below_declared"] is True
    assert fold["zV_excludes_zero"] is True
    assert fold["z_V_mag"] < 0.25
    honesty = payload["honesty"]
    assert honesty["z_slow_v_bound"] is True
    assert honesty["z_x_bound"] is False
    assert honesty["z_v_bound"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["z_x_bound"]["reason"] == "unimplemented"
    assert "z_slow_v_bound" not in reasons


def test_local_z_slow_v_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_z_slow_v")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
