# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted enclosure-trace certificates."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.verified.coeffs import sigmoid_poly_coeffs_exact
from omnibias.formal.trace import (
    LEGAL_TRACE_FAMILIES,
    TOWER_HORNER_RESULT,
    enclosure_trace_certificate,
    eval_ops,
    family_nodes_hold,
)


def test_legal_families_are_the_four_plants() -> None:
    assert LEGAL_TRACE_FAMILIES == ("tower", "nk", "bernoulli", "ldlt")


def test_tower_horner_matches_verified_coeffs() -> None:
    coeffs = sigmoid_poly_coeffs_exact(2)
    x = Fraction(2, 3)
    acc = Fraction(0)
    for coeff in reversed(coeffs):
        acc = acc * x + coeff
    assert acc == TOWER_HORNER_RESULT[0] == Fraction(-2, 27)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown enclosure-trace family"):
        enclosure_trace_certificate("dottie")  # type: ignore[arg-type]


@pytest.mark.parametrize("family", ["tower", "nk", "bernoulli", "ldlt"])
def test_certificate_payload_is_exact_and_sealed(family: str) -> None:
    cert = enclosure_trace_certificate(family)  # type: ignore[arg-type]
    assert cert["payload"]["type"] == "enclosure_trace"
    assert cert["payload"]["family"] == family
    nodes = eval_ops(cert["payload"]["ops"])
    assert nodes is not None
    assert family_nodes_hold(family, nodes)
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False
