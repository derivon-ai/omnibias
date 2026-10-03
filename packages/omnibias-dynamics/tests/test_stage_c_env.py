# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C C=0 T envelope; not a C!=0 orbit, first-hit, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_b import r1_kill
from omnibias.dynamics.stage_c_env import (
    identity_verdicts,
    report,
    residual_env_add,
    residual_half_gap,
    residual_te_unit,
)


def test_stage_c_env_identities_are_exact() -> None:
    assert residual_te_unit() == 0
    assert residual_half_gap() == 0
    assert residual_env_add() == 0
    kinetic = (r1_kill(Fraction(3, 5)) ** 2) / 2
    eps = Fraction(1, 16)
    t_orbit = kinetic * eps * eps + 1 - eps**3
    assert t_orbit / (eps * eps + 1) == Fraction(102473, 102800)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_env_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-env-v1"
    assert payload["stage_c_env"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["c_above_floor"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["c_lo"] > 0.03125
    assert enclosure["te_hi"] < 1.0
    honesty = payload["honesty"]
    assert honesty["stage_c_env"] is True
    assert honesty["stage_c_gap"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_env" not in reasons


def test_local_stage_c_env_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_env")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
