# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Regression tests for collar-membership certification: positive
(``PROVED_COLLAR``, including a genuine unique-cycle proof) and blocked
cases, plus tamper rejection and input validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.dynamics.dulac import DulacExpansion
from omnibias.dynamics.graphic import named_rational_hyperbolic_graphic
from omnibias.dynamics.membership import (
    certify_collar_membership,
    certify_collar_sequence,
    verify_collar_membership,
    verify_collar_sequence,
)
from omnibias.dynamics.saddle_normal_form import (
    diagonalize_saddle,
    dulac_corner_expansion,
    resonant_normal_form,
)


def _base_fixture():
    target = named_rational_hyperbolic_graphic()
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    derived = dulac_corner_expansion(normal_form, Q(2), order=2)
    return target, derived


def test_collar_membership_blocked_when_declared_and_derived_disagree() -> None:
    # The named graphic's declared model starts at x^1; the field-derived
    # corner map (ratio r=2) starts at x^2 -- a genuine, structural
    # disagreement that must block rather than being silently averaged away.
    target, derived = _base_fixture()
    certificate = certify_collar_membership(
        target, derived, delta=Q(1, 1000), delta0=Q(1, 100), grid_points=4
    )
    assert certificate.status == "BLOCKED"
    assert not all(node.contained for node in certificate.grid)
    assert certificate.seal["honesty"]["collar_return_membership_proved"] is False
    assert certificate.seal["honesty"]["corner_window_external"] is True
    assert verify_collar_membership(certificate)


def test_collar_membership_proved_when_declared_matches_derived_exactly() -> None:
    target, derived = _base_fixture()
    # x^2 with unit coefficient exactly matches the trivial (no-resonance)
    # corner map derived above.
    matching_map = DulacExpansion.create(((2, 0, 1),), truncation_order=2, remainder_bound=Q(0))
    target2 = replace(target, return_map=matching_map)
    certificate = certify_collar_membership(
        target2, derived, delta=Q(1, 1000), delta0=Q(1, 100), grid_points=4
    )
    assert certificate.status == "PROVED_COLLAR"
    assert all(node.contained for node in certificate.grid)
    assert certificate.seal["honesty"]["collar_return_membership_proved"] is True
    assert certificate.seal["honesty"]["physical_return_membership_proved"] is False
    assert verify_collar_membership(certificate)


def test_collar_membership_proves_a_genuine_unique_limit_cycle() -> None:
    # L(x) = 0.9*x + 20*x^2 has displacement (a-1)*x+b*x^2 with an exact
    # interior root at x = 0.1/20 = 0.005; a tight collar around it lets
    # interval_newton certify existence and uniqueness.
    target, derived = _base_fixture()
    custom_map = DulacExpansion.create(
        ((1, 0, Q(9, 10)), (2, 0, 20)), truncation_order=2, remainder_bound=Q(0)
    )
    target2 = replace(target, return_map=custom_map)
    derived2 = replace(derived, expansion=custom_map)
    certificate = certify_collar_membership(
        target2, derived2, delta=Q(3, 1000), delta0=Q(7, 1000), grid_points=4
    )
    assert certificate.status == "PROVED_COLLAR"
    assert certificate.unique_cycle.attempted
    assert certificate.unique_cycle.proved_unique_cycle
    lo, hi = certificate.unique_cycle.enclosure
    assert lo <= 0.005 <= hi
    assert verify_collar_membership(certificate)


def test_collar_membership_can_disable_the_unique_cycle_attempt() -> None:
    target, derived = _base_fixture()
    matching_map = DulacExpansion.create(((2, 0, 1),), truncation_order=2, remainder_bound=Q(0))
    target2 = replace(target, return_map=matching_map)
    certificate = certify_collar_membership(
        target2,
        derived,
        delta=Q(1, 1000),
        delta0=Q(1, 100),
        grid_points=4,
        attempt_unique_cycle=False,
    )
    assert certificate.status == "PROVED_COLLAR"
    assert not certificate.unique_cycle.attempted
    assert not certificate.unique_cycle.proved_unique_cycle
    assert verify_collar_membership(certificate)


def test_collar_membership_tamper_rejected() -> None:
    target, derived = _base_fixture()
    matching_map = DulacExpansion.create(((2, 0, 1),), truncation_order=2, remainder_bound=Q(0))
    target2 = replace(target, return_map=matching_map)
    certificate = certify_collar_membership(
        target2, derived, delta=Q(1, 1000), delta0=Q(1, 100), grid_points=4
    )
    assert not verify_collar_membership(replace(certificate, source_digest="tampered"))
    altered = deepcopy(certificate.seal)
    altered["payload"]["status"] = "BLOCKED"
    altered = seal_certificate(altered)
    assert not verify_collar_membership(replace(certificate, seal=altered))


def test_collar_sequence_does_not_close_corner_window() -> None:
    target, derived = _base_fixture()
    matching_map = DulacExpansion.create(((2, 0, 1),), truncation_order=2, remainder_bound=Q(1, 100))
    target2 = replace(target, return_map=matching_map)
    derived2 = replace(derived, expansion=matching_map)
    sequence = certify_collar_sequence(
        target2,
        derived2,
        (Q(1, 64), Q(1, 128)),
        delta0=Q(1, 32),
        grid_points=4,
    )
    assert all(cert.status == "PROVED_COLLAR" for cert in sequence.certificates)
    assert sequence.corner_window_external is True
    assert sequence.seal["honesty"]["uniform_remainder_proved"] is False
    assert verify_collar_sequence(sequence)


def test_collar_membership_rejects_invalid_collar_bounds() -> None:
    target, derived = _base_fixture()
    with pytest.raises(ValueError, match="the collar must satisfy"):
        certify_collar_membership(target, derived, delta=Q(0), delta0=Q(1, 10))
    with pytest.raises(ValueError, match="the collar must satisfy"):
        certify_collar_membership(target, derived, delta=Q(1, 2), delta0=Q(1, 10))
    with pytest.raises(ValueError, match="the collar must satisfy"):
        certify_collar_membership(target, derived, delta=Q(1, 10), delta0=Q(2))


def test_collar_membership_rejects_nonpositive_grid_points() -> None:
    target, derived = _base_fixture()
    with pytest.raises(ValueError, match="grid_points"):
        certify_collar_membership(target, derived, delta=Q(1, 100), delta0=Q(1, 10), grid_points=0)
