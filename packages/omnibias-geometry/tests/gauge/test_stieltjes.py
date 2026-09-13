# SPDX-License-Identifier: Apache-2.0
"""Independent positive-measure checks and tamper tests for rational boxes."""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction as Q
from random import Random

import pytest
from omnibias.core.proof.certificate import (
    schema_errors_v1,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.geometry.gauge._core.stieltjes import (
    replay_stieltjes_pair_certificate,
    stieltjes_pair_box,
)


def _mixture(s: Q, atoms: list[tuple[Q, Q]]) -> Q:
    return sum((weight / (s + mass_squared) for mass_squared, weight in atoms), Q(0))


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1, -2, -1, 2, 0, 1), "negative_first_value"),
        ((1, 0, 1, 2, -2, -1), "negative_second_value"),
        ((1, 1, 2, 2, 3, 4), "increasing_D"),
        ((1, 4, 5, 2, 1, 1), "decreasing_sD"),
    ],
)
def test_strict_box_exclusion(args, expected):
    result = stieltjes_pair_box(*args)
    assert result["status"] == "INCOMPATIBLE"
    assert result["finite_box_exclusion_verified"]
    assert expected in {row["code"] for row in result["obstructions"]}
    assert all(Q(row["strict_margin"]) > 0 for row in result["obstructions"])
    assert result["conditional_support_floor_upper"]["mass_squared_upper"] is None
    assert replay_stieltjes_pair_certificate(result["certificate"])


@pytest.mark.parametrize(
    "args",
    [(0, 0, 0, 1, 0, 0), (1, 1, 1, 2, 1, 1), (1, 2, 2, 2, 1, 1), (0, -1, 1, 1, -1, 1)],
)
def test_non_strict_boundaries_never_promote(args):
    result = stieltjes_pair_box(*args)
    assert result["status"] == "INCONCLUSIVE"
    assert not result["finite_box_exclusion_verified"]
    assert replay_stieltjes_pair_certificate(result["certificate"])
    # The constant positive point pair above cannot actually be the stated
    # model: every nonzero measure is strictly decreasing. Necessary checks
    # deliberately remain incomplete rather than asserting existence.
    assert not result["positive_spectral_representation_claim"]
    assert not result["spectral_gap_claim"]


def test_single_atom_recovers_support_as_an_upper_bound():
    # rho = 3 delta_4, hence D(1)=3/5, D(2)=1/2.
    result = stieltjes_pair_box(1, Q(3, 5), Q(3, 5), 2, Q(1, 2), Q(1, 2))
    upper = result["conditional_support_floor_upper"]
    assert upper["status"] == "CONDITIONAL"
    assert upper["mass_squared_upper"] == "4"
    assert upper["direction"] == "UPPER"
    assert result["status"] == "INCONCLUSIVE"
    assert not result["mass_gap_lower_bound_claim"]


def test_massless_atom_refutes_positive_gap_reading():
    atoms = [(Q(0), Q(1)), (Q(4), Q(3))]
    d1, d2 = (_mixture(Q(s), atoms) for s in (1, 2))
    result = stieltjes_pair_box(1, d1, d1, 2, d2, d2)
    upper = Q(result["conditional_support_floor_upper"]["mass_squared_upper"])
    assert upper > 0  # A positive upper bound coexists with zero support floor.
    assert min(t for t, _ in atoms) == 0
    assert not result["spectral_gap_claim"]


def test_zero_momentum_with_finite_values():
    atoms = [(Q(1), Q(2)), (Q(4), Q(3))]
    d0, d2 = _mixture(Q(0), atoms), _mixture(Q(2), atoms)
    result = stieltjes_pair_box(0, d0, d0, 2, d2, d2)
    assert result["status"] == "INCONCLUSIVE"
    assert Q(result["conditional_support_floor_upper"]["mass_squared_upper"]) >= 1


def test_direct_mixtures_grid_and_random_cover_true_support():
    rng = Random(7341)
    for i in range(160):
        atoms = [
            (Q(rng.randrange(0, 40), 7), Q(rng.randrange(1, 30), 11)) for _ in range(1 + i % 6)
        ]
        s1 = Q(1 + i % 10, 10)
        s2 = s1 + Q(1 + i % 7, 9)
        d1, d2 = _mixture(s1, atoms), _mixture(s2, atoms)
        # Both a deterministic grid of relative widths and random widths.
        width_fraction = Q(i % 21, 100) if i < 80 else Q(rng.randrange(0, 20), 100)
        width = min(d2, d1 - d2) * width_fraction
        result = stieltjes_pair_box(s1, d1 - width, d1 + width, s2, d2 - width, d2 + width)
        assert result["status"] == "INCONCLUSIVE"
        upper = Q(result["conditional_support_floor_upper"]["mass_squared_upper"])
        assert upper >= min(t for t, _ in atoms)
        exact_average = (s2 * d2 - s1 * d1) / (d1 - d2)
        assert upper >= exact_average


def test_support_upper_encloses_grid_and_random_rational_box_points():
    s1, lo1, hi1, s2, lo2, hi2 = map(Q, (1, 3, 4, 3, 2, Q(5, 2)))
    result = stieltjes_pair_box(s1, lo1, hi1, s2, lo2, hi2)
    upper = Q(result["conditional_support_floor_upper"]["mass_squared_upper"])
    fractions = [Q(i, 40) for i in range(41)]
    rng = Random(422)
    fractions += [Q(rng.randrange(10001), 10000) for _ in range(80)]
    for f in fractions:
        a = lo1 + f * (hi1 - lo1)
        for g in fractions:
            b = lo2 + g * (hi2 - lo2)
            assert (s2 * b - s1 * a) / (a - b) <= upper


def test_exclusion_excludes_every_grid_and_random_box_point():
    rng = Random(615)
    for _ in range(100):
        s1, s2 = Q(rng.randrange(1, 5)), Q(rng.randrange(5, 12))
        lo1, lo2 = Q(rng.randrange(-10, 15), 3), Q(rng.randrange(-10, 15), 3)
        hi1 = lo1 + Q(rng.randrange(0, 8), 3)
        hi2 = lo2 + Q(rng.randrange(0, 8), 3)
        result = stieltjes_pair_box(s1, lo1, hi1, s2, lo2, hi2)
        if result["status"] != "INCOMPATIBLE":
            continue
        fractions = [Q(i, 8) for i in range(9)]
        fractions += [Q(rng.randrange(1001), 1000) for _ in range(8)]
        for f in fractions:
            for g in fractions:
                a, b = lo1 + f * (hi1 - lo1), lo2 + g * (hi2 - lo2)
                assert a < 0 or b < 0 or a < b or s1 * a > s2 * b


@pytest.mark.parametrize(
    "bad",
    [True, False, 1.0, float("nan"), float("inf"), "1/2", Decimal("0.5"), None, complex(1, 0), [1]],
)
@pytest.mark.parametrize("position", range(6))
def test_refuse_inexact_or_unadvertised_types(bad, position):
    args = [1, 2, 3, 2, 1, 2]
    args[position] = bad
    with pytest.raises(TypeError):
        stieltjes_pair_box(*args)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 1, 2, 2, 1, 2),
        (2, 1, 2, 2, 1, 2),
        (3, 1, 2, 2, 1, 2),
        (1, 2, 1, 2, 1, 2),
        (1, 1, 2, 2, 2, 1),
    ],
)
def test_refuse_domain_and_box_inversion(args):
    with pytest.raises(ValueError):
        stieltjes_pair_box(*args)


def test_roundtrip_and_exact_flags():
    result = stieltjes_pair_box(1, 2, 3, 2, 1, 2)
    certificate = json.loads(json.dumps(result["certificate"]))
    assert schema_errors_v1(certificate) == []
    assert verify_certificate_digest(certificate)
    assert replay_stieltjes_pair_certificate(certificate)
    assert result["verification_kind"] == "EXACT_RATIONAL"
    assert set(result["external_premises"].values()) == {"UNVERIFIED"}
    for key in (
        "physical_applicability_verified",
        "supplied_boxes_verified",
        "theorem_prover_verified",
        "mathlib_verified",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "spectral_gap_claim",
    ):
        assert result[key] is False


@pytest.mark.parametrize(
    "target",
    ["status", "claim", "scope", "honesty", "premise", "upper", "formal", "extra", "float_input"],
)
def test_rehashed_promotion_and_claim_tampering_is_rejected(target):
    certificate = deepcopy(stieltjes_pair_box(1, 3, 4, 3, 2, Q(5, 2))["certificate"])
    report = certificate["payload"]["report"]
    if target == "status":
        report["status"] = "PROVED"
    elif target == "claim":
        certificate["claim"] = "physical mass gap proved"
    elif target == "scope":
        certificate["meta"]["scope"] = "continuum Yang-Mills"
    elif target == "honesty":
        certificate["honesty"]["yang_mills_mass_gap_claim"] = True
    elif target == "premise":
        report["external_premises"]["supplied_boxes_cover_true_values"] = "VERIFIED"
    elif target == "upper":
        report["conditional_support_floor_upper"]["direction"] = "LOWER"
    elif target == "formal":
        report["theorem_prover_verified"] = True
    elif target == "extra":
        certificate["new_physical_claim"] = True
    elif target == "float_input":
        report["inputs"]["s1"] = 1.0
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay_stieltjes_pair_certificate(certificate)


@pytest.mark.parametrize(
    "bad", [None, [], 0, {}, {"digest": "bad"}, {"payload": None}, {"payload": {"type": "other"}}]
)
def test_replay_refuses_malformed_envelopes(bad):
    assert not replay_stieltjes_pair_certificate(bad)
