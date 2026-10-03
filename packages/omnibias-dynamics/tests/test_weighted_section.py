# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""H1 weighted-section obstruction; not G1 or Hilbert XVI."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_identities import IDENTITY_NAMES
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.weighted_section import (
    identity_verdicts,
    report,
    residual_origin_weighted_speed,
    residual_weighted_hit_d2q,
    residual_weighted_hit_dq,
)


def test_weighted_section_identities_are_exact() -> None:
    rate, q = Fraction(3, 2), Fraction(1, 16)
    theta, eta0 = Fraction(1, 8), Fraction(2)
    assert residual_weighted_hit_dq(rate, q, -1 / (rate * q)) == 0
    assert residual_weighted_hit_d2q(rate, q, 1 / (rate * q**2)) == 0
    assert residual_origin_weighted_speed(
        q,
        theta,
        eta0,
        q * (1 + theta) * eta0,
    ) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert len(IDENTITY_NAMES) == 296


def test_weighted_section_falsifies_h1_route_without_passing_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-weighted-section-v1"
    assert payload["section_collapses_at_sep_zero"] is True
    assert payload["weighted_section_obstruction"] is True
    assert payload["weighted_section_closes_g1"] is False
    growth = payload["coefficient_growth"]
    assert growth["first_grows"] is True
    assert growth["second_grows"] is True
    assert growth["d1_hi"] > growth["d1_lo"]
    assert growth["d2_hi"] > growth["d2_lo"]
    origin = payload["origin_collapse"]
    assert origin["speed_collapses"] is True
    assert 0 < origin["speed_hi"] < origin["speed_lo"]
    honesty = payload["honesty"]
    assert honesty["weighted_section_obstruction"] is True
    assert honesty["new_closing_map"] is False
    assert honesty["physical_c2_remainder"] is False
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False


def test_local_weighted_section_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_weighted_section")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
