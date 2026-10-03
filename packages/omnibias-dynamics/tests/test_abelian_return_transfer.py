# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""H4 Picard--Fuchs-to-return transfer test; not DRR closure."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
from omnibias.dynamics.abelian_return_transfer import (
    FirstOrderReturnBounds,
    certify_conditional_return_transfer,
    identity_verdicts,
    report,
    residual_derivative_threshold,
    residual_normalized_displacement,
    residual_value_threshold,
    verify_conditional_return_transfer,
)
from omnibias.dynamics.hilbert16_identities import IDENTITY_NAMES
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)


def test_first_order_transfer_identities_are_exact() -> None:
    assert residual_value_threshold(
        Fraction(1, 20),
        Fraction(2),
        Fraction(1, 40),
    ) == 0
    assert residual_derivative_threshold(
        Fraction(1, 4),
        Fraction(3),
        Fraction(1, 12),
    ) == 0
    epsilon = Fraction(1, 40)
    abelian, remainder = Fraction(2, 5), Fraction(1, 7)
    displacement = epsilon * abelian + epsilon**2 * remainder
    assert residual_normalized_displacement(
        displacement,
        epsilon,
        abelian,
        remainder,
    ) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert len(IDENTITY_NAMES) == 296


def test_conditional_transfer_threshold_is_exact_and_tamper_evident() -> None:
    bounds = FirstOrderReturnBounds.create(
        zero_count=2,
        endpoint_margin=Fraction(1, 10),
        off_root_margin=Fraction(1, 20),
        derivative_margin=Fraction(1, 4),
        remainder_value_bound=2,
        remainder_derivative_bound=3,
    )
    certificate = certify_conditional_return_transfer(bounds)
    assert certificate.epsilon_upper == Fraction(1, 40)
    assert verify_conditional_return_transfer(certificate)
    assert not verify_conditional_return_transfer(
        replace(certificate, epsilon_upper=Fraction(1, 20))
    )
    with pytest.raises(ValueError, match="positive"):
        FirstOrderReturnBounds.create(
            zero_count=1,
            endpoint_margin=0,
            off_root_margin=1,
            derivative_margin=1,
            remainder_value_bound=1,
            remainder_derivative_bound=1,
        )


def test_picard_fuchs_does_not_transfer_to_open_drr_graphics() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-abelian-drr-transfer-v1"
    assert payload["picard_fuchs_verified"] is True
    assert payload["conditional_transfer_verified"] is True
    assert payload["epsilon_upper"] == [1, 40]
    assert payload["open_drr_cases"] == ["I_2^1", "I_4^1"]
    assert payload["regular_hamiltonian_oval"] is True
    assert payload["named_graphic_reduction_supplied"] is False
    assert payload["physical_remainder_derived"] is False
    assert payload["endpoint_capture_proved"] is False
    assert payload["drr_transfer_certified"] is False
    assert payload["full_hilbert16_solved"] is False


def test_local_abelian_transfer_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_abelian_return_transfer")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False
