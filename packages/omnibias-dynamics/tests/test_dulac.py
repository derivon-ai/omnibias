# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact and interval Dulac-model cyclicity regressions."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from random import Random

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.dynamics.dulac import (
    DulacExpansion,
    DulacTerm,
    RationalInterval,
    certify_dulac_cyclicity,
    certify_dulac_uniform_cover,
    displacement_expansion,
    dulac_to_confluent,
    enclose_dulac_term,
    verify_dulac_cyclicity,
    verify_dulac_cyclicity_formally,
    verify_dulac_uniform_cover,
    verify_dulac_uniform_cover_formally,
)


def test_displacement_cancels_identity_and_reports_first_three_terms() -> None:
    return_map = DulacExpansion.create(
        ((1, 0, 1), (3, 0, -1), (2, 1, 3), (Q(5, 2), 0, 2)),
        truncation_order=4,
        remainder_bound=Q(1, 100),
    )
    displacement = displacement_expansion(return_map)
    assert len(displacement.terms) == 3
    assert [
        (term.exponent.lo, term.log_power, term.coefficient.lo)
        for term in displacement.leading_terms
    ] == [(Q(2), 1, Q(3)), (Q(5, 2), 0, Q(2)), (Q(3), 0, Q(-1))]


def test_rational_dulac_model_maps_exactly_and_has_global_bound() -> None:
    displacement = DulacExpansion.create(
        ((2, 0, 2), (Q(5, 2), 0, 3), (3, 0, -1)),
        truncation_order=3,
        remainder_bound=Q(1, 1000),
    )
    transformed = dulac_to_confluent(displacement)
    assert transformed.terms == (
        (Q(-3), (Q(-1),)),
        (Q(-5, 2), (Q(3),)),
        (Q(-2), (Q(2),)),
    )
    certificate = certify_dulac_cyclicity(displacement)
    assert certificate.upper_bound == 2
    assert certificate.leading_terms == displacement.leading_terms
    assert verify_dulac_cyclicity(certificate)
    assert certificate.formal_seal["honesty"]["dulac_truncated_model_only"]
    assert certificate.formal_seal["honesty"]["physical_return_membership_proved"] is False
    formal = verify_dulac_cyclicity_formally(certificate)
    if formal.result.available:
        assert formal.theorem_prover_verified
    else:
        assert not formal.theorem_prover_verified


def test_resonant_logarithm_becomes_confluent_polynomial_factor() -> None:
    displacement = DulacExpansion.create(
        ((2, 0, 2), (2, 1, 3), (3, 0, Q(1, 2))),
        truncation_order=3,
    )
    transformed = dulac_to_confluent(displacement)
    assert transformed.terms == (
        (Q(-3), (Q(1, 2),)),
        (Q(-2), (Q(2), Q(-3))),
    )
    certificate = certify_dulac_cyclicity(displacement)
    assert certificate.upper_bound == 2


def test_interval_exponent_enclosure_contains_dense_and_random_values() -> None:
    term = DulacTerm.create(
        (Q(7, 5), Q(3, 2)),
        2,
        (Q(3, 4), Q(5, 4)),
    )
    x_box = RationalInterval.create((Q(1, 5), Q(1, 4)))
    enclosure = enclose_dulac_term(term, x_box)
    rng = Random(1600)
    probes = [
        (0.2 + 0.05 * i / 20, 1.4 + 0.1 * j / 20, 0.75 + 0.5 * k / 20)
        for i in range(21)
        for j in range(21)
        for k in (0, 10, 20)
    ]
    probes.extend(
        (
            rng.uniform(0.2, 0.25),
            rng.uniform(1.4, 1.5),
            rng.uniform(0.75, 1.25),
        )
        for _ in range(100)
    )
    for x, exponent, coefficient in probes:
        value = coefficient * x**exponent * math.log(x) ** 2
        assert enclosure.contains(value)


def test_irrational_ratio_box_has_uniform_global_nonoscillation_bound() -> None:
    expansion = DulacExpansion.create(
        (
            ((Q(3, 8), Q(2, 5)), 0, (1, 2)),
            ((Q(11, 8), Q(7, 5)), 0, (2, 3)),
            ((Q(19, 8), Q(12, 5)), 0, (1, 2)),
        ),
        truncation_order=3,
        remainder_bound=Q(1, 1000),
    )
    certificate = certify_dulac_uniform_cover(expansion)
    assert certificate.status == "PROVED"
    assert certificate.uniform_bound == 0
    assert len(certificate.tree.leaves()) == 1
    assert verify_dulac_uniform_cover(certificate)
    formal = verify_dulac_uniform_cover_formally(certificate)
    if formal.cover.available:
        assert formal.theorem_prover_verified
        assert all(result.verified for result in formal.leaves)
    else:
        assert not formal.theorem_prover_verified


def test_blocked_boxes_are_bisected_and_replayed_as_blocked() -> None:
    expansion = DulacExpansion.create(
        (((Q(3, 8), Q(2, 5)), 0, (-1, 1)), (2, 1, 1)),
        truncation_order=2,
    )
    certificate = certify_dulac_uniform_cover(
        expansion,
        max_derivative=2,
        max_depth=1,
    )
    assert certificate.status == "BLOCKED"
    assert not certificate.tree.is_leaf
    assert len(certificate.tree.leaves()) == 2
    assert verify_dulac_uniform_cover(certificate)
    assert not verify_dulac_uniform_cover(
        replace(certificate, source_digest="tampered")
    )


def test_tampered_exact_and_uniform_seals_are_rejected() -> None:
    exact = certify_dulac_cyclicity(
        DulacExpansion.create(((2, 0, 1), (3, 0, -1)), truncation_order=2)
    )
    altered = deepcopy(exact.formal_seal)
    altered["payload"]["upper_bound"] = 0
    altered = seal_certificate(altered)
    assert not verify_dulac_cyclicity(replace(exact, formal_seal=altered))

    uniform = certify_dulac_uniform_cover(
        DulacExpansion.create((((Q(7, 5), Q(3, 2)), 0, (1, 2)),), truncation_order=1)
    )
    assert uniform.seal is not None
    assert not verify_dulac_uniform_cover(replace(uniform, uniform_bound=1))
    altered_cover = deepcopy(uniform.seal)
    altered_cover["payload"]["uniform_bound"] = 1
    altered_cover = seal_certificate(altered_cover)
    assert not verify_dulac_uniform_cover(replace(uniform, seal=altered_cover))
    assert uniform.tree.sign_seal is not None
    altered_leaf = deepcopy(uniform.tree.sign_seal)
    altered_leaf["payload"]["derivative_order"] = 1
    altered_leaf = seal_certificate(altered_leaf)
    assert not verify_dulac_uniform_cover(
        replace(
            uniform,
            tree=replace(uniform.tree, sign_seal=altered_leaf),
        )
    )


def test_exact_route_refuses_interval_exponents() -> None:
    expansion = DulacExpansion.create(
        (((Q(7, 5), Q(3, 2)), 0, 1),),
        truncation_order=1,
    )
    with pytest.raises(TypeError, match="exact rational"):
        dulac_to_confluent(expansion)
