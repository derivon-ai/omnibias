# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent compact quaternion identities, exact budgets and source replay."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from math import comb
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import compact_commutator as compact

Quaternion = tuple[Q, Q, Q, Q]
Vector = tuple[Q, Q, Q]


def _mul(a: Quaternion, b: Quaternion) -> Quaternion:
    return (
        a[0] * b[0] - sum((a[i] * b[i] for i in range(1, 4)), Q(0)),
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def _inv(a: Quaternion) -> Quaternion:
    return (a[0], -a[1], -a[2], -a[3])


def _stereo(v: Vector) -> Quaternion:
    s = sum((x * x for x in v), Q(0))
    return ((1 - s) / (1 + s), 2 * v[0] / (1 + s), 2 * v[1] / (1 + s), 2 * v[2] / (1 + s))


def _cross(a: Sequence[Q], b: Sequence[Q]) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _action(a: Quaternion, b: Quaternion) -> Q:
    word = _mul(_mul(_mul(a, b), _inv(a)), _inv(b))
    return 2 - 2 * word[0]


@pytest.fixture(scope="module")
def certificate_result() -> dict[str, Any]:
    return compact.su2_compact_commutator_gap()


def test_default_exact_bounds_and_sealed_sector_scopes(certificate_result: dict[str, Any]) -> None:
    row = certificate_result
    a = row["arithmetic"]
    assert row["status"] == "PASS"
    assert a["kappa_one_third_enclosure"] == ["1/4096", "1/4096"]
    assert a["center_even_gap_lower"] == "1/204800"
    assert a["other_center_sector_energy_difference_upper"] == str(Q(91, 2**24))
    assert a["other_center_sector_count"] == 7
    assert Q(a["normalized_gap_lower"]) == Q(11389445042978907523, 230083594272878100480)
    assert Q(a["normalized_gap_slack"]) > 0
    assert Q(a["flux_coefficient_budget_upper"]) == Q(18173, 200) < 91
    assert all(a["gates"].values())
    assert row["certificate"]["payload"] == {
        key: value for key, value in row.items() if key != "certificate"
    }
    assert row["actual_compact_center_even_gap_verified_in_written_analysis"]
    assert row["actual_seven_flux_sector_upper_bounds_verified_in_written_analysis"]
    for key in (
        "unrestricted_gap_verified",
        "unrestricted_kappa_one_third_gap_verified",
        "exponential_tunneling_rate_verified",
        "arbitrary_exterior_conditional_verified",
        "uniform_in_volume_claim",
        "all_scale_refinement_claim",
        "continuum_claim",
        "yang_mills_mass_gap_claim",
        "analytic_proof_formally_verified",
        "theorem_prover_verified",
        "mathlib_verified",
    ):
        assert row[key] is False
    assert compact.replay_compact_commutator_certificate(row["certificate"])


@pytest.mark.parametrize("kappa", [Q(1, 2**36), Q(1, 3 * 2**40), Q(1, 2**999)])
def test_positive_range_and_relative_root_floor(kappa: Q) -> None:
    row = compact.su2_compact_commutator_gap(kappa)
    a = row["arithmetic"]
    lower, upper = (Q(v) for v in a["kappa_one_third_enclosure"])
    assert row["status"] == "PASS"
    assert 0 < lower**3 <= kappa <= upper**3
    assert 0 < Q(a["center_even_gap_lower"]) <= lower / 50
    assert Q(a["other_center_sector_energy_difference_upper"]) >= 91 * upper**2
    assert compact.replay_compact_commutator_certificate(row["certificate"])


@pytest.mark.parametrize("kappa", [Q(1, 2**36) + Q(1, 10**40), Q(1, 2**35), Q(1)])
def test_outside_range_is_inconclusive_not_an_unrestricted_gap(kappa: Q) -> None:
    row = compact.su2_compact_commutator_gap(kappa)
    a = row["arithmetic"]
    assert row["status"] == "INCONCLUSIVE"
    assert a["center_even_gap_lower"] is None
    assert a["other_center_sector_energy_difference_upper"] is None
    assert not row["actual_compact_center_even_gap_verified_in_written_analysis"]
    assert not row["actual_seven_flux_sector_upper_bounds_verified_in_written_analysis"]
    assert compact.replay_compact_commutator_certificate(row["certificate"])


@pytest.mark.parametrize("value", [True, 0.0, 0.001, "1/100", None, 0, -1, Q(-1, 8)])
def test_invalid_couplings(value: object) -> None:
    call: Callable[..., object] = compact.su2_compact_commutator_gap
    with pytest.raises((TypeError, ValueError)):
        call(value)


def test_failed_dynamic_matrix_source_cannot_earn_compact_claim(
    monkeypatch: pytest.MonkeyPatch, certificate_result: dict[str, Any]
) -> None:
    source = certificate_result["matrix_source_certificate"]
    monkeypatch.setattr(compact, "_matrix_source", lambda: (source, False, Q(1701, 200), Q(43, 5)))
    row = compact.su2_compact_commutator_gap()
    assert row["status"] == "INCONCLUSIVE"
    assert not row["arithmetic"]["gates"]["canonical_matrix_source"]
    assert not row["actual_compact_center_even_gap_verified_in_written_analysis"]
    assert not row["actual_seven_flux_sector_upper_bounds_verified_in_written_analysis"]


@pytest.mark.parametrize(
    "place", ["sector", "bound", "scope", "source", "scalar_source", "meta", "status"]
)
def test_rehashed_outer_and_nested_forgery_rejected(
    place: str, certificate_result: dict[str, Any]
) -> None:
    cert = deepcopy(certificate_result["certificate"])
    payload = cert["payload"]
    if place == "sector":
        payload["gap_sector"] = "all physical sectors"
    elif place == "bound":
        payload["arithmetic"]["center_even_gap_lower"] = "1"
    elif place == "scope":
        payload["unrestricted_gap_verified"] = True
    elif place == "status":
        payload["status"] = "INCONCLUSIVE"
    elif place == "meta":
        cert["meta"]["no_homogeneous_subspace_of_larger_lattice_assumed"] = False
    else:
        matrix = payload["matrix_source_certificate"]
        if place == "source":
            matrix["payload"]["arithmetic"]["first_singlet_excited_energy_lower"] = "100"
        else:
            scalar = matrix["payload"]["scalar_source"]["certificate"]
            scalar["payload"]["status"] = "INCONCLUSIVE"
            matrix["payload"]["scalar_source"]["certificate"] = make_certificate(
                claim=scalar["claim"], payload=scalar["payload"], meta=scalar["meta"]
            )
        payload["matrix_source_certificate"] = make_certificate(
            claim=matrix["claim"], payload=matrix["payload"], meta=matrix["meta"]
        )
    forged = make_certificate(claim=cert["claim"], payload=payload, meta=cert["meta"])
    assert verify_certificate_digest(forged)
    assert not compact.replay_compact_commutator_certificate(forged)


def test_malformed_replay_refused(certificate_result: dict[str, Any]) -> None:
    assert not compact.replay_compact_commutator_certificate({})
    cert = deepcopy(certificate_result["certificate"])
    del cert["payload"]["inputs"]["kappa"]
    assert not compact.replay_compact_commutator_certificate(cert)
    cert = make_certificate(claim=cert["claim"], payload=cert["payload"], meta=cert["meta"])
    assert not compact.replay_compact_commutator_certificate(cert)


def test_integer_cube_root_exact_grid_and_power_boundaries() -> None:
    for n in [*range(4096), *(2 ** (3 * p) + j for p in [30, 80, 111] for j in [-1, 0, 1])]:
        root = compact._integer_cube_root(n)
        assert root**3 <= n < (root + 1) ** 3
    for n in [-1, True]:
        with pytest.raises(ValueError):
            compact._integer_cube_root(n)


def test_cube_root_enclosure_independent_exact_grid_and_random() -> None:
    rng = Random(9238)
    values = [Q(a, b) for a in range(1, 12) for b in range(1, 12)]
    values += [
        Q(rng.randint(1, 1000), rng.randint(1, 1000)) * Q(2) ** rng.randint(-1000, 1000)
        for _ in range(128)
    ]
    for value in values:
        lo, hi = compact._cube_root_enclosure(value)
        assert 0 < lo <= hi and lo**3 <= value <= hi**3
        assert hi - lo <= lo / 2**80
    for exponent in [-600, -36, 0, 33, 300]:
        value = Q(2) ** exponent
        assert compact._cube_root_enclosure(value) == (Q(2) ** (exponent // 3),) * 2


@pytest.mark.parametrize("sample", range(8))
def test_exact_nonabelian_commutator_identity_and_center_invariance(sample: int) -> None:
    rng = Random(3281 + sample)
    a = _stereo((Q(rng.randint(-5, 5), 7), Q(rng.randint(-5, 5), 7), Q(rng.randint(-5, 5), 7)))
    b = _stereo((Q(rng.randint(-5, 5), 9), Q(rng.randint(-5, 5), 9), Q(rng.randint(-5, 5), 9)))
    cross = _cross(a[1:], b[1:])
    expected = 4 * sum((z * z for z in cross), Q(0))
    for sign_a, sign_b in product([-1, 1], repeat=2):
        aa = (sign_a * a[0], sign_a * a[1], sign_a * a[2], sign_a * a[3])
        bb = (sign_b * b[0], sign_b * b[1], sign_b * b[2], sign_b * b[3])
        assert _action(aa, bb) == expected
    assert _action(a, a) == 0


def test_original_quaternion_generator_metric_and_radial_ims_factor() -> None:
    for n in range(-8, 9):
        u = _stereo((Q(n, 13), Q(2, 11), Q(-1, 7)))
        directions = []
        for axis in range(3):
            tangent = (Q(0), Q(int(axis == 0), 2), Q(int(axis == 1), 2), Q(int(axis == 2), 2))
            directions.append(_mul(u, tangent)[1:])
        actual = [
            [sum((d[i] * d[j] for d in directions), Q(0)) for j in range(3)] for i in range(3)
        ]
        expected = [[(int(i == j) - u[i + 1] * u[j + 1]) / 4 for j in range(3)] for i in range(3)]
        assert actual == expected
        s2 = sum((u[i] ** 2 for i in range(1, 4)), Q(0))
        gradient = [sum((2 * u[i + 1] * d[i] for i in range(3)), Q(0)) for d in directions]
        assert sum((v * v for v in gradient), Q(0)) == s2 * (1 - s2)
        assert s2 * (1 - s2) / (4 * s2) <= Q(1, 4)


@pytest.mark.parametrize("degree", range(1, 8))
def test_conditional_radial_casimir_from_ambient_homogeneous_laplacian(degree: int) -> None:
    for v in [Q(1, 7), Q(2, 5), Q(3, 4)]:
        q1, q2 = v, Q(1, 3)
        z = q1**2 + q2**2
        ambient_laplacian = Q(0)
        for k in range(degree + 1):
            a, b = 2 * k, 2 * (degree - k)
            if a >= 2:
                ambient_laplacian += comb(degree, k) * a * (a - 1) * q1 ** (a - 2) * q2**b
            if b >= 2:
                ambient_laplacian += comb(degree, k) * b * (b - 1) * q1**a * q2 ** (b - 2)
        ambient = (-ambient_laplacian + 2 * degree * (2 * degree + 2) * z**degree) / 4
        radial = -(degree**2) * z ** (degree - 1) + degree * (degree + 1) * z**degree
        assert ambient == radial


def test_conditional_positive_test_function_square_and_kinetic_allocation() -> None:
    for a, eta, z in product([Q(1, 8), Q(2)], [Q(0), Q(1, 3), Q(7)], [Q(0), Q(1, 4), Q(1)]):
        local = a * (-(eta**2) * z * (1 - z) + eta * (1 - 2 * z)) + a * eta**2 * z
        assert local == a * eta + a * ((eta * z - 1) ** 2 - 1)
        assert local >= a * eta - a
    # Six ordered pieces allocate half the kinetic energy and every potential.
    assert Q(1, 4) + 2 * Q(1, 8) == Q(1, 2)
    assert 2 * Q(1) == 2
    assert 6 * Q(1, 8) == Q(3, 4)


def test_center_characters_count_one_good_copy_per_sector() -> None:
    signs = list(product([-1, 1], repeat=3))
    characters = list(product([0, 1], repeat=3))
    values = [[s[0] ** p[0] * s[1] ** p[1] * s[2] ** p[2] for s in signs] for p in characters]
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            assert sum(a * b for a, b in zip(left, right, strict=True)) == (8 if i == j else 0)
    assert values[0] == [1] * 8
    assert len(values[1:]) == 7


def test_gaussian_moments_scaling_and_cutoff_coefficients() -> None:
    a = Q(5, 4)
    second, fourth = Q(1) / (2 * a), Q(3) / (4 * a**2)
    r2 = 9 * second
    r4 = 9 * fourth + 2 * comb(9, 2) * second**2
    assert r2 == Q(18, 5) and r4 == Q(99) / (4 * a**2)
    potential = 3 * (9 * second**2 - 3 * second**2)
    assert 9 * a / 2 + potential == Q(1701, 200)
    assert a**2 * r4 / 4 + Q(10, 16) == Q(109, 16)
    for sigma in [Q(1, 4096), Q(1, 30000), Q(1, 2**50)]:
        kappa, scale = sigma**3, sigma / 2
        assert kappa / (8 * scale**2) == scale
        assert 8 * scale**4 / kappa == scale


def test_uniform_budget_monotonicity_grid_random_and_flux_estimates() -> None:
    rng = Random(75390)
    taus = [Q(i, 64 * 100) for i in range(1, 101)]
    taus += [Q(rng.randint(1, 10000), 64 * 10000) for _ in range(100)]
    lower, upper = [], []
    for tau in sorted(taus):
        x = tau**2
        f, w = (1 - 4 * x) ** 3, 1 / (1 - 4 * x) ** 2
        lo = Q(43, 5) * f - Q(5, 8) * x
        hi = w * (Q(1701, 200) + Q(109, 16) * x) / (1 - Q(9, 10) * x)
        assert lo - hi > Q(1, 25)
        assert tau - Q(3, 4) * tau**6 >= Q(43, 10) * tau**2
        assert w - 1 <= 9 * x and 1 - f <= 12 * x
        assert w <= 2 and 1 - Q(1701, 200) * tau / 2 >= Q(1, 2)
        assert (21 * Q(1701, 200) + Q(25, 8)) / 2 == Q(18173, 200) < 91
        lower.append(lo)
        upper.append(hi)
    assert lower == sorted(lower, reverse=True)
    assert upper == sorted(upper)
