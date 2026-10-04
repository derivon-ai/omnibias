# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact arithmetic and refusal tests for the conditional cubic-beam gate."""

from fractions import Fraction as Q
from random import Random

import pytest
from omnibias.core.verified.clamped_biharmonic import (
    clamped_biharmonic_green_kernel,
    clamped_biharmonic_inverse_norm,
    clamped_cubic_contraction,
)


def test_exact_full_inverse_norm_and_length_scaling():
    assert clamped_biharmonic_inverse_norm() == Q(1, 384)
    assert clamped_biharmonic_inverse_norm(2) == Q(1, 24)
    assert clamped_biharmonic_inverse_norm(Q(1, 2)) == Q(1, 6144)
    for x in (Q(0), Q(1, 4), Q(1, 2), Q(1)):
        for t in (Q(0), Q(1, 3), Q(1)):
            assert clamped_biharmonic_green_kernel(2 * x, 2 * t, length=2) == (
                8 * clamped_biharmonic_green_kernel(x, t)
            )


def test_green_positivity_symmetry_on_dense_and_random_points():
    rng = Random(84)
    points = [Q(i, 30) for i in range(31)]
    points += [Q(rng.randrange(10000), 10000) for _ in range(30)]
    for x in points:
        for t in points:
            value = clamped_biharmonic_green_kernel(x, t)
            assert value >= 0
            assert value == clamped_biharmonic_green_kernel(t, x)
            assert value <= Q(1, 192)
    assert clamped_biharmonic_green_kernel(Q(1, 2), Q(1, 2)) == Q(1, 192)


def test_exact_gate_does_not_discharge_its_external_bounds():
    gate = clamped_cubic_contraction(1, 32, 32, Q(1, 8))
    assert gate.passed
    assert gate.defect == Q(1, 12)
    assert gate.contraction == Q(81, 256)
    assert gate.self_map_margin == Q(13, 6144)
    payload = gate.to_payload()
    assert payload["status"] == "PASS"
    assert set(payload["external_premises"].values()) == {"UNVERIFIED"}
    assert not payload["pde_existence_claim"]
    assert not payload["theorem_prover_verified"]


def test_false_residual_bound_is_only_a_conditional_gate():
    gate = clamped_cubic_contraction(0, 0, 0, Q(1, 1000000))
    assert gate.passed
    assert not gate.to_payload()["pde_existence_claim"]


def test_failed_contraction_or_self_map_refuses():
    assert not clamped_cubic_contraction(1, 32, 32, Q(1, 1000000)).passed
    assert not clamped_cubic_contraction(1, 0, 1000, Q(1, 8)).passed


@pytest.mark.parametrize(
    "args",
    [
        (True, 1, 1, Q(1, 8)),
        (1.0, 1, 1, Q(1, 8)),
        (1, float("nan"), 1, Q(1, 8)),
    ],
)
def test_nonrational_inputs_refused(args):
    with pytest.raises(TypeError):
        clamped_cubic_contraction(*args)


@pytest.mark.parametrize(
    "args", [(-1, 1, 1, Q(1, 8)), (1, -1, 1, Q(1, 8)), (1, 1, -1, Q(1, 8)), (1, 1, 1, 0)]
)
def test_wrong_sign_inputs_refused(args):
    with pytest.raises(ValueError):
        clamped_cubic_contraction(*args)


def test_domain_refusals():
    with pytest.raises(ValueError):
        clamped_biharmonic_inverse_norm(0)
    with pytest.raises(ValueError):
        clamped_biharmonic_green_kernel(-1, 0)
    with pytest.raises(ValueError):
        clamped_biharmonic_green_kernel(1, 0, length=Q(1, 2))
