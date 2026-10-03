# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""H3 direct LN-format barrier; not a general LN impossibility theorem."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.dynamics.hilbert16_identities import IDENTITY_NAMES
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)
from omnibias.dynamics.ln_format_barrier import (
    identity_verdicts,
    report,
    residual_kill_log_w,
    residual_kill_norm,
    residual_kill_tau,
    truncated_matching_chain,
    truncated_matching_format,
)


def test_kill_sequence_format_identities_are_exact() -> None:
    n = Fraction(7)
    assert residual_kill_tau(n, n) == 0
    assert residual_kill_log_w(2 * n, n) == 0
    assert residual_kill_norm(3 * n, n) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert len(IDENTITY_NAMES) == 296


def test_every_finite_truncation_has_a_replayable_ln_chain() -> None:
    item = truncated_matching_format(16)
    assert item.chain_length == 2
    assert item.closure_degree == 1
    assert item.coefficient_norm == 2
    assert item.domain_inner == 1
    assert item.domain_outer == 16
    assert item.chain_sup_norm == 48
    assert item.chain_valid is True
    assert item.certificate_valid is True
    with pytest.raises(ValueError, match="at least two"):
        truncated_matching_chain(1)


def test_direct_format_grows_but_normalized_route_is_not_excluded() -> None:
    payload = report((4, 8, 16, 32)).to_payload()
    assert payload["schema"] == "hilbert16-ln-format-barrier-v1"
    assert payload["finite_truncations_certified"] is True
    assert payload["direct_chain_length_uniform"] is True
    assert payload["direct_coefficients_uniform"] is True
    assert payload["direct_domain_uniform"] is False
    assert payload["direct_norm_uniform"] is False
    assert payload["direct_bounded_format"] is False
    assert payload["actual_return_ln_membership_proved"] is False
    assert payload["normalized_zero_equivalent_route_open"] is True
    assert payload["g3_passed"] is False
    assert payload["full_hilbert16_solved"] is False
    honesty = payload["honesty"]
    assert honesty["direct_ln_format_barrier"] is True
    assert honesty["actual_return_ln_membership_proved"] is False
    assert honesty["normalized_ln_return_member"] is False
    with pytest.raises(ValueError, match="strictly increasing"):
        report((8, 4))


def test_local_ln_barrier_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_ln_format_barrier")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False
