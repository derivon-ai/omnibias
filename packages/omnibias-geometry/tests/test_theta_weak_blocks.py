# SPDX-License-Identifier: Apache-2.0
"""Original-link frames, complete angular budgets and attached theta evidence."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import theta_weak_blocks as theta
from omnibias.geometry.gauge.transfer import weak_plaquette

Quat = tuple[Q, Q, Q, Q]
Word = tuple[tuple[int, bool], ...]
_IDENTITY: Quat = (Q(1), Q(0), Q(0), Q(0))
_X: Word = ((5, True), (0, True), (4, False), (2, False))
_Y: Word = ((5, True), (1, False), (6, False), (3, True))


def _mul(a: Quat, b: Quat) -> Quat:
    return (
        a[0] * b[0] - sum((a[i] * b[i] for i in range(1, 4)), Q(0)),
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def _inv(a: Quat) -> Quat:
    return a[0], -a[1], -a[2], -a[3]


def _stereo(values: Sequence[Q]) -> Quat:
    s = sum((v * v for v in values), Q(0))
    return (
        (1 - s) / (1 + s),
        2 * values[0] / (1 + s),
        2 * values[1] / (1 + s),
        2 * values[2] / (1 + s),
    )


def _dot(a: Sequence[Q], b: Sequence[Q]) -> Q:
    return sum((x * y for x, y in zip(a, b, strict=True)), Q(0))


def _basis(axis: int) -> Quat:
    return Q(0), Q(axis == 0, 2), Q(axis == 1, 2), Q(axis == 2, 2)


def _word(links: Sequence[Quat], word: Word) -> Quat:
    value = _IDENTITY
    for edge, inverse in word:
        value = _mul(value, _inv(links[edge]) if inverse else links[edge])
    return value


def _word_derivative(links: Sequence[Quat], word: Word, edge: int, axis: int) -> Quat:
    if edge not in [i for i, _ in word]:
        return Q(0), Q(0), Q(0), Q(0)
    value = _IDENTITY
    for i, inverse in word:
        factor = _mul(links[i], _basis(axis)) if i == edge else links[i]
        value = _mul(value, _inv(factor) if inverse else factor)
    return value


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return theta.su2_theta_weak_block_gaps()


def test_default_exact_bounds_sectors_and_canonical_replay(result: dict[str, Any]) -> None:
    a = result["arithmetic"]
    assert result["status"] == "PASS"
    assert a["ground_energy_upper"] == "6"
    assert a["full_reduced_gap_lower"] == "2/5"
    assert a["conditional_A_quantum_gap_lower"] == "2/15"
    assert a["conditional_B_quantum_gap_lower"] == "4/25"
    assert a["conditional_A_poincare_gap_lower"] == "256/15"
    assert a["conditional_B_poincare_gap_lower"] == "512/25"
    assert a["forest_maximal_correlation_lower"] == "253/256"
    assert Q(a["radial_branch_slack"]) == Q(779, 121) - Q(39, 2048) - Q(32, 5) > 0
    assert all(a["gates"].values())
    assert result["certificate"]["payload"] == {
        k: v for k, v in result.items() if k != "certificate"
    }
    assert result["actual_full_reduced_gap_verified_in_written_analysis"]
    assert result["actual_conditional_block_gaps_verified_in_written_analysis"]
    assert result["all_boundary_flux_sectors_retained"]
    assert theta.replay_su2_theta_weak_block_certificate(result["certificate"])
    for flag in (
        "source_gap_used_as_premise",
        "unrestricted_original_seven_link_gap_verified",
        "ambient_exterior_uniformity_verified",
        "uniform_in_volume_claim",
        "all_scale_refinement_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "analytic_proof_formally_verified",
        "theorem_prover_verified",
        "mathlib_verified",
        "correlation_lower_is_overlap_upper",
        "physical_projection_angle_lower_verified",
    ):
        assert result[flag] is False


@pytest.mark.parametrize("kappa", [Q(1, 64), Q(1, 333), Q(1, 2**180)])
def test_positive_interval_and_inverse_coupling_poincare_units(kappa: Q) -> None:
    row = theta.su2_theta_weak_block_gaps(kappa)
    a = row["arithmetic"]
    assert row["status"] == "PASS"
    assert kappa * Q(a["conditional_A_poincare_gap_lower"]) / 2 == Q(2, 15)
    assert kappa * Q(a["conditional_B_poincare_gap_lower"]) / 2 == Q(4, 25)
    assert theta.replay_su2_theta_weak_block_certificate(row["certificate"])


@pytest.mark.parametrize("kappa", [Q(1, 64) + Q(1, 10**40), Q(1, 32), Q(4, 3), Q(9)])
def test_outside_gap_range_retains_only_the_earned_correlation(kappa: Q) -> None:
    row = theta.su2_theta_weak_block_gaps(kappa)
    assert row["status"] == "INCONCLUSIVE"
    assert not row["actual_full_reduced_gap_verified_in_written_analysis"]
    assert not row["actual_conditional_block_gaps_verified_in_written_analysis"]
    assert row["actual_theta_vacuum_identified_in_written_analysis"]
    assert row["actual_forest_correlation_lower_verified_in_written_analysis"]
    assert Q(row["arithmetic"]["forest_maximal_correlation_lower"]) == max(Q(0), 1 - 3 * kappa / 4)
    for key in (
        "full_reduced_gap_lower",
        "physical_theta_gap_lower",
        "conditional_A_quantum_gap_lower",
        "conditional_B_quantum_gap_lower",
        "conditional_A_poincare_gap_lower",
        "conditional_B_poincare_gap_lower",
    ):
        assert row["arithmetic"][key] is None
    assert theta.replay_su2_theta_weak_block_certificate(row["certificate"])


@pytest.mark.parametrize("value", [True, False, 0, -1, Q(-1, 9), 0.01, "1/64", None])
def test_strict_input_guards(value: object) -> None:
    call: Callable[..., object] = theta.su2_theta_weak_block_gaps
    with pytest.raises((TypeError, ValueError)):
        call(value)


@pytest.mark.parametrize(
    "damage", ["bound", "scope", "geometry", "units", "source", "status", "meta"]
)
def test_resealed_outer_and_nested_forgery_rejected(damage: str, result: dict[str, Any]) -> None:
    cert = deepcopy(result["certificate"])
    p = cert["payload"]
    if damage == "bound":
        p["arithmetic"]["full_reduced_gap_lower"] = "1"
    elif damage == "scope":
        p["ambient_exterior_uniformity_verified"] = True
    elif damage == "geometry":
        p["blocks"]["internal_A_vertices"] = []
    elif damage == "units":
        p["arithmetic"]["conditional_A_poincare_gap_lower"] = "2/15"
    elif damage == "status":
        p["status"] = "INCONCLUSIVE"
    elif damage == "meta":
        cert["meta"]["transcend_backend"] = "unchecked"
    else:
        source = p["plaquette_trial_source_certificate"]
        source["payload"]["witness"]["arithmetic"]["ground_energy_upper"] = "0"
        p["plaquette_trial_source_certificate"] = seal_certificate(source)
    forged = seal_certificate(cert)
    assert verify_certificate_digest(forged)
    assert not theta.replay_su2_theta_weak_block_certificate(forged)


def test_dynamic_source_failure_cannot_earn_any_actual_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = weak_plaquette.su2_weak_plaquette_gap(Q(1, 64))
    damaged = deepcopy(source)
    damaged["certificate"]["payload"]["witness"]["arithmetic"]["ground_energy_upper"] = "0"
    damaged["certificate"] = seal_certificate(damaged["certificate"])
    original = weak_plaquette.su2_weak_plaquette_gap
    calls = 0

    def producer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return damaged if calls == 1 else original(*args, **kwargs)

    monkeypatch.setattr(weak_plaquette, "su2_weak_plaquette_gap", producer)
    row = theta.su2_theta_weak_block_gaps()
    assert row["status"] == "INCONCLUSIVE"
    assert not row["actual_theta_vacuum_identified_in_written_analysis"]
    assert not row["actual_forest_correlation_lower_verified_in_written_analysis"]
    assert row["arithmetic"]["forest_maximal_correlation_lower"] is None


def test_detached_source_summaries_and_source_gap_are_not_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = weak_plaquette.su2_weak_plaquette_gap(Q(1, 64))
    source["gap_lower"] = "999"
    source["status"] = "INCONCLUSIVE"
    source["witness"] = {"arithmetic": {"ground_energy_upper": "-999"}}
    original = weak_plaquette.su2_weak_plaquette_gap
    # Replay must rebuild through the genuine implementation, not this stub.
    calls = 0

    def producer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source if calls == 1 else original(*args, **kwargs)

    monkeypatch.setattr(weak_plaquette, "su2_weak_plaquette_gap", producer)
    row = theta.su2_theta_weak_block_gaps()
    assert row["status"] == "PASS" and row["arithmetic"]["ground_energy_upper"] == "6"


@pytest.mark.parametrize("bad", [None, True, [], {}, "certificate"])
def test_malformed_replay(bad: object) -> None:
    replay: Callable[..., bool] = theta.replay_su2_theta_weak_block_certificate
    assert not replay(bad)


@pytest.mark.parametrize("seed", range(6))
def test_original_seven_edge_derivatives_and_conditional_forms(seed: int) -> None:
    rng = Random(981 + seed)
    links = [_stereo([Q(rng.randint(-5, 5), 7) for _ in range(3)]) for _ in range(7)]
    x, y = _word(links, _X), _word(links, _Y)
    gx = tuple(Q(rng.randint(-5, 5), 3) for _ in range(4))
    gy = tuple(Q(rng.randint(-5, 5), 3) for _ in range(4))
    rows = [
        sum(
            (
                (
                    _dot(gx, _word_derivative(links, _X, edge, axis))
                    + _dot(gy, _word_derivative(links, _Y, edge, axis))
                )
                ** 2
                for axis in range(3)
            ),
            Q(0),
        )
        for edge in range(7)
    ]
    cx = sum((_dot(gx, _mul(x, _basis(axis))) ** 2 for axis in range(3)), Q(0))
    cy = sum((_dot(gy, _mul(y, _basis(axis))) ** 2 for axis in range(3)), Q(0))
    diag = sum(
        (
            (_dot(gx, _mul(_basis(axis), x)) + _dot(gy, _mul(_basis(axis), y))) ** 2
            for axis in range(3)
        ),
        Q(0),
    )
    ca = sum((rows[i] for i in [0, 1, 5]), Q(0))
    cb = sum((rows[i] for i in [2, 3, 4, 6]), Q(0))
    assert ca == cx + cy + diag
    assert cb == 2 * (cx + cy)
    assert sum(rows) == 3 * (cx + cy) + diag
    assert 0 <= diag <= 2 * (cx + cy)
    assert ca >= sum(rows) / 3 and cb >= 2 * sum(rows) / 5


def test_finite_internal_and_boundary_gauge_actions() -> None:
    links = [_stereo([Q(i + 1, 11), Q(i - 2, 13), Q(1, 7)]) for i in range(7)]
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    old_x, old_y = _word(links, _X), _word(links, _Y)
    h = _stereo([Q(1, 3), Q(2, 5), Q(-1, 7)])
    for vertex in range(6):
        transformed = [
            _mul(_mul(h if a == vertex else _IDENTITY, u), _inv(h) if b == vertex else _IDENTITY)
            for (a, b), u in zip(edges, links, strict=True)
        ]
        expected_x = _mul(_mul(h, old_x), _inv(h)) if vertex == 4 else old_x
        expected_y = _mul(_mul(h, old_y), _inv(h)) if vertex == 4 else old_y
        assert _word(transformed, _X) == expected_x
        assert _word(transformed, _Y) == expected_y


def test_character_covariance_identity_by_exact_degree_two_haar_cubature() -> None:
    nodes: list[Quat] = []
    for axis, sign in product(range(4), [-1, 1]):
        nodes.append(
            (
                Q(sign if axis == 0 else 0),
                Q(sign if axis == 1 else 0),
                Q(sign if axis == 2 else 0),
                Q(sign if axis == 3 else 0),
            )
        )
    assert sum((2 * h[0] for h in nodes), Q(0)) == 0
    assert sum(((2 * h[0]) ** 2 for h in nodes), Q(0)) / 8 == 1
    for i in range(-8, 9):
        u = _stereo([Q(i, 11), Q(2, 9), Q(-1, 7)])
        p = _stereo([Q(1, 5), Q(i, 13), Q(3, 11)])
        integral = sum((4 * _mul(u, _inv(h))[0] * _mul(h, p)[0] for h in nodes), Q(0)) / 8
        assert integral == _mul(u, p)[0]


def test_barta_radial_derivatives_completion_and_all_angular_budget() -> None:
    for angular, t, c in product(range(8), [Q(1, 7), Q(2), Q(11)], [Q(i, 9) for i in range(-8, 9)]):
        first = t - angular * c / (1 - c * c)
        second = -angular * (1 + c * c) / (1 - c * c) ** 2
        quotient = (
            -(1 - c * c) * (second + first * first)
            + 3 * c * first
            + angular * (angular + 1) / (1 - c * c)
        )
        assert quotient == angular * (angular + 2) + (2 * angular + 3) * t * c - t * t * (1 - c * c)
        a = Q(3, 8)
        b, d = (2 * angular + 3) * a * t, a * t * t
        local = a * angular * (angular + 2) + b * c + d * (1 - c) ** 2
        lower = a * angular * (angular + 2) + b - b * b / (4 * d)
        assert local - lower == d * ((1 - c) - b / (2 * d)) ** 2 >= 0
        assert a * (angular * (angular + 2) - Q((2 * angular + 3) ** 2, 4)) == -Q(
            12 * angular + 27, 32
        )
    rng = Random(119)
    kappas = [Q(i, 64 * 256) for i in range(1, 257)]
    kappas += [Q(rng.randint(1, 2**30), 64 * 2**30) for _ in range(128)]
    for kappa in kappas:
        assert Q(779, 121) - 39 * kappa / 32 > Q(32, 5)
        assert Q(76, 11) - 33 * kappa / 16 > Q(32, 5)
