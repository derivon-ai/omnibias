# SPDX-License-Identifier: Apache-2.0
"""Independent finite algebra and scope tests for actual Wilson energy bounds."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.wilson_large_field import (
    replay_su2_wilson_large_field_certificate,
    su2_wilson_large_field,
)


@pytest.mark.parametrize("coupling", (Q(1, 1000), Q(1, 100), Q(1, 4), Q(4, 3), Q(2), Q(15)))
def test_exact_energy_and_mean_are_actual_uniform_family_bounds(coupling: Q) -> None:
    report = su2_wilson_large_field(coupling)
    a = report["witness"]["arithmetic"]
    mean = min(Q(3, 2) * coupling, Q(2))
    assert Q(a["trial_parameter_t"]) == 4 / coupling
    assert Q(a["ground_energy_per_plaquette_upper"]) == min(Q(3), 4 / coupling)
    assert Q(a["plaquette_action_mean_upper"]) == mean
    assert Q(a["plaquette_action_form_energy_upper"]) == coupling * mean * (4 - mean) / 2
    assert report["actual_finite_volume_vacuum_verified"]
    assert report["volume_uniform_plaquette_mean_bound_verified"]
    assert replay_su2_wilson_large_field_certificate(report["certificate"])
    assert report["witness"]["family"]["side_length_quantifier"] == (
        "every integer L>=3 with 3*L^3>=marked_count"
    )
    for flag in (
        "uniform_over_all_exteriors_verified", "conditional_large_field_bound_verified",
        "exponential_polymer_bound_verified", "cluster_independence_claim",
        "spectral_gap_claim", "nonvacuum_variance_lower_bound_verified",
        "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
        "yang_mills_claim", "yang_mills_mass_gap_claim",
        "theorem_prover_verified", "mathlib_verified",
    ):
        assert report[flag] is False


def test_weak_coupling_exact_example_and_localization_coefficients() -> None:
    report = su2_wilson_large_field(
        Q(1, 100), threshold=Q(1, 2), marked_count=8, required_bad_count=8,
    )
    a = report["witness"]["arithmetic"]
    assert a["plaquette_action_mean_upper"] == "3/200"
    assert a["expected_bad_count_upper"] == "6/25"
    assert a["large_field_probability_upper"] == "3/100"
    assert a["plaquette_action_gradient_square_mean_upper"] == "2391/40000"
    assert a["plaquette_action_form_energy_upper"] == "2391/8000000"
    assert a["marked_average_action_form_energy_upper"] == "2391/16000000"
    assert a["marked_edge_incidence_upper"] == 4
    assert report["probability_bound_nontrivial"]


@pytest.mark.parametrize("count", (1, 2, 7, 100))
def test_all_marked_bad_is_not_exponentiated(count: int) -> None:
    report = su2_wilson_large_field(
        Q(1, 100), threshold=Q(1, 2), marked_count=count, required_bad_count=count,
    )
    assert report["large_field_probability_upper"] == "3/100"
    if count > 1:
        assert Q(report["large_field_probability_upper"]) > Q(3, 100) ** count


def test_vacuous_probability_is_reported_honestly() -> None:
    report = su2_wilson_large_field(10, threshold=1, marked_count=8, required_bad_count=1)
    assert report["status"] == "PASS"
    assert report["large_field_probability_upper"] == "1"
    assert report["probability_bound_nontrivial"] is False
    assert report["witness"]["arithmetic"]["expected_bad_count_upper"] == "8"


@pytest.mark.parametrize("count", (1, 2, 3, 4, 8, 100))
def test_marked_average_retains_actual_incidence_factor(count: int) -> None:
    report = su2_wilson_large_field(Q(1, 3), marked_count=count)
    a = report["witness"]["arithmetic"]
    assert a["marked_edge_incidence_upper"] == min(4, count)
    assert Q(a["marked_average_action_form_energy_upper"]) == (
        Q(a["plaquette_action_form_energy_upper"]) * min(4, count) / count
    )


def test_first_moments_allow_a_perfectly_correlated_probability_fixture() -> None:
    # A finite probability model, not the actual quantum vacuum.
    coupling, threshold, count = Q(1, 100), Q(1, 2), 8
    mean = Q(3, 2) * coupling
    all_bad_probability = mean / 4
    report = su2_wilson_large_field(
        coupling, threshold=threshold, marked_count=count, required_bad_count=count,
    )
    assert 4 * all_bad_probability == mean
    assert all_bad_probability <= Q(report["large_field_probability_upper"])
    assert all_bad_probability > Q(report["large_field_probability_upper"]) ** count


@pytest.mark.parametrize("c", [Q(k, 8) for k in range(1, 65)])
def test_exact_barrier_contact_identity(c: Q) -> None:
    barrier = 1 - Q(3, 2) / c
    contact = 1 - barrier**2 - 3 * barrier / c - Q(3, 2) / c**2
    assert contact == Q(3, 4) / c**2 > 0


def _one_link_mean_enclosure(c: Q) -> tuple[Q, Q]:
    # Positive Haar-moment series for Z=E exp(c*q0) and Z'.
    # Ratios decrease, so the two omitted tails are bounded geometrically.
    order = 32
    term, z, derivative = Q(1), Q(1), Q(0)
    for k in range(1, order + 1):
        term *= c**2 / (4 * k * (k + 1))
        z += term
        derivative += 2 * k * term / c
    next_term = term * c**2 / (4 * (order + 1) * (order + 2))
    z_ratio = c**2 / (4 * (order + 2) * (order + 3))
    d_ratio = c**2 / (4 * (order + 1) * (order + 3))
    assert 0 < z_ratio < 1 and 0 < d_ratio < 1
    z_upper = z + next_term / (1 - z_ratio)
    d_upper = derivative + 2 * (order + 1) * next_term / c / (1 - d_ratio)
    return derivative / z_upper, d_upper / z


def test_dense_and_seeded_one_link_integral_enclosures_check_trial_bounds() -> None:
    rng = Random(9122026)
    values = [Q(k, 8) for k in range(1, 65)]
    values += [Q(rng.randrange(1, 129), 16) for _ in range(24)]
    for c in values:
        lower, upper = _one_link_mean_enclosure(c)
        assert 0 < lower <= upper < 1
        assert lower >= max(Q(0), 1 - Q(3, 2) / c)
        # For the certified choice t=4/kappa, c=16/kappa.
        coupling = 16 / c
        trial_upper = Q(3, 2) * upper + 4 / coupling * (1 - lower**4)
        assert trial_upper <= 3


Quaternion = tuple[Q, Q, Q, Q]


def _mul(a: Quaternion, b: Quaternion) -> Quaternion:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    )


def _conjugate(q: Quaternion) -> Quaternion:
    return q[0], -q[1], -q[2], -q[3]


def _unit(v: tuple[Q, Q, Q]) -> Quaternion:
    r = sum((x * x for x in v), Q(0))
    return (1 - r) / (1 + r), 2 * v[0] / (1 + r), 2 * v[1] / (1 + r), 2 * v[2] / (1 + r)


def _word(factors: list[Quaternion]) -> Quaternion:
    value: Quaternion = (Q(1), Q(0), Q(0), Q(0))
    for factor in factors:
        value = _mul(value, factor)
    return value


def test_exact_quaternion_grid_and_random_gradients_obey_original_metric() -> None:
    rng = Random(20260913)
    grid = [_unit((Q(x), Q(y), Q(z))) for x, y, z in product((-1, 0, 1), repeat=3)]
    packs = [[grid[(i + j * 5) % len(grid)] for j in range(4)] for i in range(len(grid))]
    packs += [
        [_unit(tuple(Q(rng.randrange(-3, 4), 3) for _ in range(3))) for _ in range(4)]  # type: ignore[arg-type]
        for _ in range(24)
    ]
    generators: tuple[Quaternion, ...] = (
        (Q(0), Q(1, 2), Q(0), Q(0)),
        (Q(0), Q(0), Q(1, 2), Q(0)),
        (Q(0), Q(0), Q(0), Q(1, 2)),
    )
    for links in packs:
        factors = [links[0], links[1], _conjugate(links[2]), _conjugate(links[3])]
        holonomy = _word(factors)
        action = 2 - 2 * holonomy[0]
        total = Q(0)
        for edge in range(4):
            edge_square = Q(0)
            for generator in generators:
                derivative = _mul(generator, links[edge])
                if edge >= 2:
                    derivative = _conjugate(derivative)
                varied = factors.copy()
                varied[edge] = derivative
                derivative_action = -2 * _word(varied)[0]
                edge_square += derivative_action**2
            assert edge_square == action - action**2 / 4
            total += edge_square
        assert total == 4 * action - action**2


@pytest.mark.parametrize("side", (3, 4, 5))
def test_periodic_cubic_geometry_has_distinct_edges_and_incidence_four(side: int) -> None:
    incidence: Counter[tuple[int, int, int, int]] = Counter()
    count = 0
    for vertex in product(range(side), repeat=3):
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = list(vertex), list(vertex)
                vi[i] = (vi[i] + 1) % side
                vj[j] = (vj[j] + 1) % side
                edges = [
                    (vertex[0], vertex[1], vertex[2], i),
                    (vi[0], vi[1], vi[2], j),
                    (vj[0], vj[1], vj[2], i),
                    (vertex[0], vertex[1], vertex[2], j),
                ]
                assert len(set(edges)) == 4
                incidence.update(edges)
                count += 1
    assert count == len(incidence) == 3 * side**3
    assert set(incidence.values()) == {4}


Edge = tuple[int, int, int, int]
Face = tuple[tuple[Edge, bool], ...]
Vector = tuple[Q, Q, Q]


def _periodic_faces(side: int) -> list[Face]:
    faces: list[Face] = []
    for vertex in product(range(side), repeat=3):
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = list(vertex), list(vertex)
                vi[i] = (vi[i] + 1) % side
                vj[j] = (vj[j] + 1) % side
                faces.append((
                    ((vertex[0], vertex[1], vertex[2], i), False),
                    ((vi[0], vi[1], vi[2], j), False),
                    ((vj[0], vj[1], vj[2], i), True),
                    ((vertex[0], vertex[1], vertex[2], j), True),
                ))
    return faces


def _face_action_gradient(
    face: Face, links: dict[Edge, Quaternion],
) -> tuple[Q, dict[Edge, Vector]]:
    factors = [_conjugate(links[e]) if inverse else links[e] for e, inverse in face]
    action = 2 - 2 * _word(factors)[0]
    generators: tuple[Quaternion, ...] = (
        (Q(0), Q(1, 2), Q(0), Q(0)),
        (Q(0), Q(0), Q(1, 2), Q(0)),
        (Q(0), Q(0), Q(0), Q(1, 2)),
    )
    gradient: dict[Edge, Vector] = {}
    for position, (edge, inverse) in enumerate(face):
        derivatives: list[Q] = []
        for generator in generators:
            derivative = _mul(generator, links[edge])
            varied = factors.copy()
            varied[position] = _conjugate(derivative) if inverse else derivative
            derivatives.append(-2 * _word(varied)[0])
        gradient[edge] = derivatives[0], derivatives[1], derivatives[2]
    return action, gradient


@pytest.mark.parametrize("count", (1, 2, 4, 6, 12))
def test_shared_link_pointwise_sum_and_universal_ims_majorant(count: int) -> None:
    # Independent original-link differentiation; no vacuum/certificate bound
    # supplies the pointwise inequality, including on four-face shared edges.
    all_faces = _periodic_faces(3)
    anchor = (0, 0, 0, 0)
    star = [face for face in all_faces if any(e == anchor for e, _ in face)]
    assert len(star) == 4
    marked = (star + [face for face in all_faces if face not in star])[:count]
    incidence = Counter(e for face in marked for e, _ in face)
    degree = max(incidence.values())
    assert degree == min(4, count)
    edges = sorted(incidence)
    rng = Random(20260914 + count)
    for sample in range(8):
        links: dict[Edge, Quaternion] = {}
        for i, edge in enumerate(edges):
            if sample < 3:
                coordinates = (
                    Q((i + sample) % 3 - 1, 32),
                    Q((2 * i + sample) % 3 - 1, 32),
                    Q((i + 2 * sample) % 3 - 1, 32),
                )
            else:
                coordinates = (
                    Q(rng.randrange(-3, 4), 32),
                    Q(rng.randrange(-3, 4), 32),
                    Q(rng.randrange(-3, 4), 32),
                )
            links[edge] = _unit(coordinates)
        total_action, separate_squares = Q(0), Q(0)
        sums = {e: [Q(0), Q(0), Q(0)] for e in edges}
        for face in marked:
            action, gradient = _face_action_gradient(face, links)
            total_action += action
            square = sum((x * x for vector in gradient.values() for x in vector), Q(0))
            assert square == 4 * action - action**2
            separate_squares += square
            for edge, vector in gradient.items():
                for axis in range(3):
                    sums[edge][axis] += vector[axis]
        total_square = sum((x * x for vector in sums.values() for x in vector), Q(0))
        assert 0 <= total_square <= degree * separate_squares <= 4 * degree * total_action
        if total_action == 0:
            assert total_square == 0
            continue
        threshold = Q(3, 2) * total_action
        assert threshold / 2 < total_action < threshold
        lipschitz, coupling = Q(22, 7) / threshold, Q(3, 5)
        # Local exact circle algebra keeps both partition derivatives. The
        # rational 22/(7*s) is their slope majorant, not an evaluation of pi.
        cosine, sine = Q(3, 5), Q(4, 5)
        partition_square = (cosine**2 + sine**2) * lipschitz**2 * total_square
        assert partition_square == lipschitz**2 * total_square
        assert coupling * partition_square / 2 <= 968 * coupling * degree / (49 * threshold)
        assert 2 * total_action / coupling >= threshold / coupling


@pytest.mark.parametrize("side", (1, 2, 7, 128, 1024))
def test_growing_block_universal_ims_and_bad_potential_scaling(side: int) -> None:
    b = Q(side)
    coupling, epsilon = b**-30, b**-12
    threshold = (7 * epsilon / (44 * b))**2
    universal = 968 * coupling * 4 / (49 * threshold)
    bad_floor = threshold / coupling
    assert threshold == Q(7, 44)**2 * b**-26
    assert universal == Q(3872, 49) * Q(44, 7)**2 / b**4
    assert bad_floor == Q(7, 44)**2 * b**4
    assert universal * bad_floor == Q(3872, 49)


@pytest.mark.parametrize("kwargs", (
    {"kappa": 0}, {"kappa": -1}, {"kappa": 1, "threshold": 0},
    {"kappa": 1, "threshold": 5}, {"kappa": 1, "marked_count": 0},
    {"kappa": 1, "required_bad_count": 0},
    {"kappa": 1, "marked_count": 2, "required_bad_count": 3},
))
def test_invalid_exact_domains_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        su2_wilson_large_field(**kwargs)


@pytest.mark.parametrize("key", ("kappa", "threshold", "marked_count", "required_bad_count"))
@pytest.mark.parametrize("bad", (True, 1.0, "1", None))
def test_inexact_or_ambiguous_inputs_refused(key: str, bad: Any) -> None:
    kwargs: dict[str, Any] = {"kappa": Q(1, 2), key: bad}
    with pytest.raises(TypeError):
        su2_wilson_large_field(**kwargs)


@pytest.mark.parametrize("change", (
    "family", "mean", "probability", "form", "normalization", "conditional",
    "polymer", "gap", "continuum", "formal", "schema", "claim",
))
def test_rehashed_arithmetic_or_scope_tampering_is_rejected(change: str) -> None:
    certificate = deepcopy(su2_wilson_large_field(Q(1, 100))["certificate"])
    witness = certificate["payload"]["witness"]
    if change == "family":
        witness["family"]["boundary"] = "arbitrary fixed exterior"
    elif change == "mean":
        witness["arithmetic"]["plaquette_action_mean_upper"] = "0"
    elif change == "probability":
        witness["arithmetic"]["large_field_probability_upper"] = "0"
    elif change == "form":
        witness["arithmetic"]["marked_average_action_form_energy_upper"] = "0"
    elif change == "normalization":
        witness["normalization"] = "SU2 fundamental Casimir1"
    elif change in ("conditional", "polymer", "gap", "continuum", "formal"):
        key = {
            "conditional": "uniform_over_all_exteriors_verified",
            "polymer": "exponential_polymer_bound_verified",
            "gap": "spectral_gap_claim",
            "continuum": "continuum_claim",
            "formal": "theorem_prover_verified",
        }[change]
        certificate["honesty"][key] = True
    elif change == "schema":
        certificate["payload"]["type"] = "unsupported"
    else:
        certificate["claim"] = "continuum Yang-Mills large-field control"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay_su2_wilson_large_field_certificate(certificate)


@pytest.mark.parametrize("bad", (None, [], {}, {"payload": {}}, {"digest": "bad"}))
def test_malformed_certificate_refused(bad: Any) -> None:
    assert not replay_su2_wilson_large_field_certificate(bad)
