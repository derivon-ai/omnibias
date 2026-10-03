# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""H5 finite Bautin-jet audit; not an all-orders stabilization theorem."""

from __future__ import annotations

from dataclasses import replace

import pytest
from omnibias.dynamics.bautin_stabilization_barrier import (
    certify_finite_bautin_audit,
    report,
    verify_finite_bautin_audit,
)
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)


@pytest.fixture(scope="module")
def finite_audit():
    return certify_finite_bautin_audit(order=10)


def test_derived_v4_has_an_exact_membership_witness(finite_audit) -> None:
    assert [degree for degree, _ in finite_audit.quantities.quantities] == [
        4,
        6,
        8,
        10,
    ]
    assert finite_audit.finite_order_stabilization_verified is True
    assert [degree for degree, _, _ in finite_audit.checked_memberships] == [
        10
    ]
    assert verify_finite_bautin_audit(finite_audit)
    assert not verify_finite_bautin_audit(
        replace(finite_audit, source_digest="tampered")
    )


def test_same_finite_jet_has_a_formal_continuation_outside_the_ideal(
    finite_audit,
) -> None:
    witness = finite_audit.adversarial_nonmembership
    assert finite_audit.adversarial_degree == 12
    assert witness.is_member is False
    assert not witness.remainder.is_zero()
    assert witness.remainder == finite_audit.adversarial_candidate


def test_h5_report_keeps_all_orders_and_g2_false(finite_audit) -> None:
    payload = report(order=10, audit=finite_audit).to_payload()
    assert payload["schema"] == "hilbert16-bautin-stabilization-barrier-v1"
    assert payload["reference_degrees"] == [4, 6, 8]
    assert payload["checked_higher_degrees"] == [10]
    assert payload["finite_order_stabilization_verified"] is True
    assert payload["formal_jet_counterexample_verified"] is True
    assert payload["all_orders_focal_recurrence_proved"] is False
    assert payload["actual_singular_return_map_derived"] is False
    assert payload["bautin_ideal_stabilization_proved"] is False
    assert payload["g2_passed"] is False
    assert payload["full_hilbert16_solved"] is False
    with pytest.raises(ValueError, match="even integer"):
        certify_finite_bautin_audit(order=9)


def test_local_bautin_barrier_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_bautin_stabilization_barrier")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False
