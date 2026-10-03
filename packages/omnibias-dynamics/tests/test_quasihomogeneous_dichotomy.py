# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Quantified one-scale H2 dichotomy; not G1 or Hilbert XVI."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_identities import IDENTITY_NAMES
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.quasihomogeneous_dichotomy import (
    classify_monomial_weight,
    identity_verdicts,
    monomial_bounds_both,
    report,
    residual_event_log_expansion,
    residual_moving_section_log_expansion,
    residual_outgoing_log_expansion,
)


def test_quasihomogeneous_log_identities_are_exact() -> None:
    a, b = Fraction(2, 3), Fraction(5, 4)
    n, log_n = Fraction(7), Fraction(3, 2)
    event_log = (1 - b) * n**2 - a * log_n
    frozen_log = 2 * b * n + (2 * a + 3) * log_n / n
    moving_log = (2 * b - 2) * n + 2 * a * log_n / n
    assert residual_event_log_expansion(event_log, a, b, n**2, log_n) == 0
    assert residual_outgoing_log_expansion(
        frozen_log, a, b, 0, n, log_n
    ) == 0
    assert residual_moving_section_log_expansion(
        moving_log, a, b, Fraction(2), 3, n, log_n
    ) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert len(IDENTITY_NAMES) == 296


def test_monomial_weight_boundary_classification() -> None:
    assert classify_monomial_weight(Fraction(0), Fraction(1, 2))[
        "event_bounded"
    ] is False
    assert classify_monomial_weight(Fraction(-1), Fraction(1))[
        "event_bounded"
    ] is False
    assert classify_monomial_weight(Fraction(0), Fraction(1))[
        "event_bounded"
    ] is True
    assert classify_monomial_weight(Fraction(1), Fraction(1))[
        "event_bounded"
    ] is True
    assert classify_monomial_weight(Fraction(-10), Fraction(2))[
        "event_bounded"
    ] is True
    assert classify_monomial_weight(Fraction(3), Fraction(0))[
        "frozen_ratio_bounded_above"
    ] is True
    assert classify_monomial_weight(Fraction(3), Fraction(-1))[
        "frozen_ratio_bounded_above"
    ] is True
    for a_num in range(-4, 5):
        for b_num in range(-4, 9):
            assert not monomial_bounds_both(
                Fraction(a_num, 2), Fraction(b_num, 4)
            )


def test_h2_broad_atlas_claim_is_rejected_by_moving_section() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-quasihomogeneous-dichotomy-v1"
    assert payload["all_rational_monomial_weights_excluded"] is True
    assert payload["frozen_section_scale_no_go"] is True
    assert payload["moving_section_counterexample"] is True
    assert payload["all_quasihomogeneous_atlases_excluded"] is False
    assert payload["g1_passed"] is False
    honesty = payload["honesty"]
    assert honesty["frozen_section_scale_no_go"] is True
    assert honesty["all_quasihomogeneous_atlases_excluded"] is False
    assert honesty["new_closing_map"] is False
    assert honesty["full_hilbert16_solved"] is False


def test_local_quasi_dichotomy_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_quasihomogeneous_dichotomy")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
