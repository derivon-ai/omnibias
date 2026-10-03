# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""H6 Songling lower-bound reproduction-readiness audit."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)
from omnibias.dynamics.songling_lower_bound import (
    certify_songling_precision_audit,
    report,
    songling_field,
    verify_songling_precision_audit,
)


def test_songling_field_is_an_exact_quadratic_source() -> None:
    field = songling_field()
    assert field.degree == 2
    assert field.p.nvars == field.q.nvars == 2
    assert field.p.derivative(0).evaluate((0, 0)) == -Fraction(1, 10**200)
    assert field.p.derivative(1).evaluate((0, 0)) == -1
    assert field.q.derivative(0).evaluate((0, 0)) == 1


def test_binary64_backend_loses_the_governing_epsilon_sign() -> None:
    audit = certify_songling_precision_audit()
    assert audit.exact_epsilon_effect == -Fraction(8, 10**52)
    assert audit.recovered_epsilon_box.contains_zero()
    assert audit.recovered_epsilon_box.width > 10**35 * abs(
        float(audit.exact_epsilon_effect)
    )
    assert len(audit.root_boxes) == len(audit.displacement_margins) == 4
    assert min(audit.displacement_margins) == Fraction("5.03e-295")
    assert verify_songling_precision_audit(audit)
    assert not verify_songling_precision_audit(
        replace(audit, source_digest="tampered")
    )


def test_h6_is_known_externally_but_not_replayed_by_omnibias() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-songling-lower-bound-audit-v1"
    assert payload["exact_source_verified"] is True
    assert payload["published_h2_lower_bound"] == 4
    assert payload["published_exact_cycle_count"] == 4
    assert payload["published_certificate_precedes_omnibias"] is True
    assert payload["published_max_precision_bits"] == 2048
    assert payload["omnibias_interval_endpoint_bits"] == 53
    assert payload["governing_epsilon_sign_resolved"] is False
    assert payload["four_hyperbolic_returns_certified"] is False
    assert payload["h2_lower_bound_independently_certified"] is False
    assert payload["full_hilbert16_solved"] is False


def test_local_songling_audit_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_songling_lower_bound_audit")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False
