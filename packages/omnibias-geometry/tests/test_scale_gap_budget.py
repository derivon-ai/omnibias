# SPDX-License-Identifier: Apache-2.0
"""Exact finite algebra, scale normalization and premise-refusal regressions."""
from __future__ import annotations

import random
from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof import certificate as certificate_module
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.scale_gap_budget import (
    SCOPE,
    replay_scale_gap_budget_certificate,
    scale_gap_budget,
)


def _steps(n: int) -> list[list[Q]]:
    return [[Q(1, 2), Q(1, 16) * Q(1, 2)**i, Q(1, 4) * Q(1, 2)**i, Q(1)] for i in range(n)]


def test_nonzero_cross_couplings_and_geometric_floor() -> None:
    result = scale_gap_budget(1, 2, _steps(8), geometric_envelope=(Q(1, 8), Q(1, 2)))
    w = result["witness"]
    assert result["status"] == "PASS"
    assert replay_scale_gap_budget_certificate(result["certificate"])
    assert w["geometric_all_terms_sum"] == "1/4"
    assert w["conditional_all_stage_physical_floor"] == "3/2"
    assert Q(w["conditional_finite_physical_gap_floor"]) > Q(3, 2)
    assert w["spacing_sequence"][-1] == "1/256"
    assert Q(w["computed_lattice_gap_sequence"][-1]) * 256 == Q(w["conditional_finite_physical_gap_floor"])
    assert all(row["schur_matrix_psd_verified"] for row in w["steps"])
    assert all(result[key] is False for key in SCOPE)
    assert result["theorem_prover_verified"] is False
    assert "theorem_prover_verified" not in result["certificate"]["honesty"]


@pytest.mark.parametrize("n", [0, 1, 2, 7, 20])
def test_exact_scale_cancellation_and_product_bound(n: int) -> None:
    result = scale_gap_budget(Q(3, 5), Q(7, 2), _steps(n))
    witness = result["witness"]
    product, total = Q(1), Q(0)
    for i, (_, epsilon, beta, h) in enumerate(_steps(n), 1):
        loss = epsilon + beta**2 / h
        product *= 1 - loss
        total += loss
        assert product >= 1 - total
        assert Q(witness["retained_product_sequence"][i]) == product
        assert Q(witness["computed_lattice_gap_sequence"][i]) == Q(7, 2) * Q(1, 2)**i * product
    assert Q(witness["conditional_finite_physical_gap_floor"]) == Q(35, 6) * product


def test_scaling_spacings_and_gaps_together_preserves_physical_result() -> None:
    original = scale_gap_budget(Q(3, 5), Q(7, 2), _steps(4))["witness"]
    changed = scale_gap_budget(Q(33, 35), Q(11, 2), _steps(4))["witness"]
    assert original["conditional_finite_physical_gap_floor"] == changed["conditional_finite_physical_gap_floor"]
    other_ratios = [[Q(3, 4), *row[1:]] for row in _steps(4)]
    refined = scale_gap_budget(Q(3, 5), Q(7, 2), other_ratios)["witness"]
    assert original["spacing_sequence"] != refined["spacing_sequence"]
    assert original["conditional_finite_physical_gap_floor"] == refined["conditional_finite_physical_gap_floor"]


def test_exact_schur_square_identity_on_grid_and_random_arguments() -> None:
    rng = random.Random(160129)
    points = [(Q(x, 3), Q(y, 4)) for x in range(-3, 4) for y in range(-3, 4)]
    points += [(Q(rng.randrange(-20, 21), 7), Q(rng.randrange(-20, 21), 11)) for _ in range(50)]
    for epsilon, beta, h in ((Q(0), Q(0), Q(1)), (Q(1, 8), Q(1, 3), Q(5, 4)),
                             (Q(2), Q(3), Q(1, 7))):
        delta = epsilon + beta**2 / h
        for x, y in points:
            remainder = beta**2 / h * x*x - 2*beta*x*y + (h+delta)*y*y
            square = (beta*x-h*y)**2 + (h*epsilon+beta**2)*y*y
            assert h*remainder == square >= 0


def test_uniform_relative_loss_is_not_a_uniform_physical_gap() -> None:
    floors = []
    for n in (1, 8, 32):
        result = scale_gap_budget(1, 1, [[Q(1, 2), Q(1, 2), Q(0), Q(1)]] * n)
        assert result["status"] == "PASS"  # only a positive finite prefix
        w = result["witness"]
        assert Q(w["conditional_finite_lattice_gap_floor"]) == Q(1, 4)**n
        assert Q(w["conditional_finite_physical_gap_floor"]) == Q(1, 2)**n
        assert w["conditional_all_stage_physical_floor"] is None
        assert result["all_scale_envelope_verified"] is False
        floors.append(Q(w["conditional_finite_physical_gap_floor"]))
    assert floors[2] < floors[1] < floors[0]


def test_passing_prefix_does_not_certify_an_unseen_successor() -> None:
    initial = scale_gap_budget(1, 2, _steps(3), geometric_envelope=(Q(1, 8), Q(1, 2)))
    assert initial["status"] == "PASS"
    assert not initial["all_scale_envelope_verified"]
    bad_next = [*_steps(3), [Q(1, 2), Q(1, 4), Q(0), Q(1)]]
    result = scale_gap_budget(1, 2, bad_next, geometric_envelope=(Q(1, 8), Q(1, 2)))
    assert result["status"] == "INCONCLUSIVE"
    assert result["witness"]["finite_chain_budget_verified"] is True
    assert result["witness"]["geometric_prefix_verified"] is False
    assert result["witness"]["conditional_all_stage_physical_floor"] is None
    assert replay_scale_gap_budget_certificate(result["certificate"])


@pytest.mark.parametrize("loss", [Q(1), Q(3, 2), Q(2)])
def test_nonpositive_retained_fraction_is_inconclusive(loss: Q) -> None:
    result = scale_gap_budget(1, 1, [[Q(1, 2), loss, Q(0), Q(1)]] * 2)
    assert result["status"] == "INCONCLUSIVE"
    assert not result["witness"]["finite_chain_budget_verified"]
    assert result["witness"]["conditional_finite_physical_gap_floor"] is None
    assert replay_scale_gap_budget_certificate(result["certificate"])


def test_geometric_total_boundary_and_zero_envelope() -> None:
    at_boundary = scale_gap_budget(1, 2, [], geometric_envelope=(Q(1, 2), Q(1, 2)))
    assert at_boundary["status"] == "INCONCLUSIVE"
    assert at_boundary["witness"]["geometric_prefix_verified"] is True
    zero = scale_gap_budget(1, 2, [[Q(1, 2), Q(0), Q(0), Q(1)]], geometric_envelope=(0, 0))
    assert zero["status"] == "PASS"
    assert zero["witness"]["conditional_all_stage_physical_floor"] == "2"
    assert not zero["all_scale_envelope_verified"]


@pytest.mark.parametrize("damage", ["spacing", "steps", "product", "floor", "physical", "future", "formal", "parent", "extra", "type"])
def test_canonical_replay_rejects_resealed_tampering(damage: str) -> None:
    cert = deepcopy(scale_gap_budget(1, 2, _steps(3), geometric_envelope=(Q(1, 8), Q(1, 2)))["certificate"])
    w = cert["payload"]["witness"]
    if damage == "spacing":
        w["inputs"]["initial_spacing"] = "2"
    elif damage == "steps":
        w["inputs"]["steps"][0][2] = "0"
    elif damage == "product":
        w["retained_product_sequence"][-1] = "1"
    elif damage == "floor":
        w["conditional_all_stage_physical_floor"] = "100"
    elif damage == "physical":
        cert["honesty"]["physical_gap_verified"] = True
    elif damage == "future":
        cert["honesty"]["all_scale_envelope_verified"] = True
    elif damage == "formal":
        cert["honesty"]["theorem_prover_verified"] = True
    elif damage == "parent":
        cert["honesty"]["yang_mills_mass_gap_claim"] = True
    elif damage == "extra":
        w["unverified_lemma_assumed"] = True
    else:
        cert["payload"]["type"] = "physical_rg_solution"
    assert not replay_scale_gap_budget_certificate(seal_certificate(cert))


@pytest.mark.parametrize("value", [None, True, [], (), 1, "certificate", {}])
def test_malformed_replay_is_false(value: Any) -> None:
    assert replay_scale_gap_budget_certificate(value) is False


@pytest.mark.parametrize("value", [True, False, 0.5, "1/2", None])
def test_inexact_public_inputs_are_rejected(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        scale_gap_budget(value, 1, [])
    with pytest.raises((TypeError, ValueError)):
        scale_gap_budget(1, value, [])
    with pytest.raises((TypeError, ValueError)):
        scale_gap_budget(1, 1, [[1, 0, value, 1]])
    with pytest.raises((TypeError, ValueError)):
        scale_gap_budget(1, 1, [], geometric_envelope=(value, Q(1, 2)))


@pytest.mark.parametrize("steps", [None, "steps", [[1, 2]], [[0, 0, 0, 1]], [[1, -1, 0, 1]], [[1, 0, -1, 1]], [[1, 0, 0, 0]]])
def test_malformed_step_shape_and_signs_are_rejected(steps: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        scale_gap_budget(1, 1, steps)


@pytest.mark.parametrize("envelope", [(1,), (0, 1), (-1, 0), (0, -1), "envelope"])
def test_malformed_envelope_is_rejected(envelope: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        scale_gap_budget(1, 1, [], geometric_envelope=envelope)


def test_rational_certificate_does_not_inherit_sticky_transcendental_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(certificate_module, "libm_fallback_used", lambda: True)
    monkeypatch.setattr(certificate_module, "backend_name", lambda: "libm_fallback")
    result = scale_gap_budget(1, 2, _steps(2))
    assert result["certificate"]["meta"]["transcend_backend"] == "not_used"
    assert result["certificate"]["meta"]["analytic_implication"] == "docs/api/gauge-scale-gap-budget.md"
    assert replay_scale_gap_budget_certificate(result["certificate"])
