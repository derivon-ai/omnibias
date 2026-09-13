# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent curl, trial-form and tamper regressions; floats are diagnostics."""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from typing import Any

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from omnibias.core.proof.certificate import make_certificate
from omnibias.geometry.gauge.transfer.periodic_corner import (
    periodic_corner_box_lower_budget,
    periodic_corner_box_surface,
    periodic_corner_energy_budget,
    replay_periodic_corner_certificate,
    su2_periodic_corner_harmonic,
    su2_periodic_upper_density,
)


def _curl(side: int) -> np.ndarray:
    vertices = list(product(range(side), repeat=3))
    edge = {(x, axis): 3 * i + axis for i, x in enumerate(vertices) for axis in range(3)}
    result = np.zeros((3 * len(vertices), 3 * len(vertices)))
    for vertex_index, x in enumerate(vertices):
        for face_index, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            xi = tuple((v + (axis == i)) % side for axis, v in enumerate(x))
            xj = tuple((v + (axis == j)) % side for axis, v in enumerate(x))
            row = result[3 * vertex_index + face_index]
            row[edge[x, i]] += 1
            row[edge[xi, j]] += 1
            row[edge[xj, i]] -= 1
            row[edge[x, j]] -= 1
    return result


def _fourier(side: int) -> tuple[np.ndarray, np.ndarray]:
    momenta = 2 * np.pi * np.array(list(product(range(side), repeat=3))) / side
    z = np.exp(1j * momenta) - 1
    lam = np.sum(abs(z) ** 2, axis=1)
    coherent = abs(z[:, 2] - z[:, 1]) ** 2
    coherent += abs(z[:, 0] - z[:, 2]) ** 2 + abs(z[:, 1] - z[:, 0]) ** 2
    p = np.zeros(len(lam))
    nonzero = lam > 1e-20
    p[nonzero] = coherent[nonzero] / lam[nonzero]
    return lam, p


@pytest.mark.parametrize("side", [3, 4])
@pytest.mark.parametrize("tilt", [0, Q(1, 8), Q(1, 4)])
def test_original_link_tilt_spectrum_matches_two_frequency_symbol(side: int, tilt: Q) -> None:
    curl = _curl(side)
    face_metric = np.eye(3 * side**3) + float(tilt) * np.kron(
        np.eye(side**3), (np.eye(3) - np.ones((3, 3))) / 4
    )
    actual = np.linalg.eigvalsh(curl.T @ face_metric @ curl)
    lam, p = _fourier(side)
    expected = np.sort(
        np.concatenate(
            [np.zeros(side**3), lam * (1 + float(tilt) / 4), lam * (1 + float(tilt) * (1 - p) / 4)]
        )
    )
    assert np.max(abs(actual - expected)) < 4e-14
    assert np.count_nonzero(abs(actual) < 1e-12) == side**3 + 2


@pytest.mark.parametrize("side", [3, 4, 5, 7, 12])
def test_discrete_moments_gaussian_ratio_and_secant(side: int) -> None:
    lam, p = _fourier(side)
    momenta = 2 * np.pi * np.array(list(product(range(side), repeat=3))) / side
    q = 4 * np.sin(momenta / 2) ** 2
    assert np.allclose(q.mean(axis=0), 2, rtol=0, atol=2e-14)
    assert abs(np.mean(q[:, 0] * q[:, 1]) - 4) < 2e-14
    positive = lam > 0
    d = 2 * np.mean(np.sqrt(lam)) / 3
    c = np.sum(q[positive, 0] * q[positive, 1] / np.sqrt(lam[positive])) / (4 * side**3)
    assert 1 - 2 * c / d < 2 / 3
    assert 9 * c / 4 > 9 / 14
    e0 = 3 * np.mean(np.sqrt(lam))
    e1 = 1.5 * np.mean(np.sqrt(lam) * (np.sqrt(1 + 1 / 16) + np.sqrt(1 + (1 - p) / 16)))
    assert 4 * (e1 - e0) > float(Q(29, 112))


def test_rank_three_resolvent_uses_coherent_original_rows() -> None:
    side = 3
    curl = _curl(side)
    k = curl.T @ curl
    corner = curl[:3]
    qs = 4 * np.sin(np.pi * np.array(list(product(range(side), repeat=3))) / side) ** 2
    lam = qs.sum(axis=1)
    for s in [0.01, 0.125, 1.0, 4.0, 64.0]:
        original = corner @ np.linalg.solve(k + s * np.eye(len(k)), corner.T)
        d = 2 * np.mean(lam / (s + lam)) / 3
        c = np.mean(qs[:, 0] * qs[:, 1] / (s + lam)) / 4
        expected = (d + c) * np.eye(3) - c * np.ones((3, 3))
        assert np.max(abs(original - expected)) < 2e-13
        determinant = np.linalg.det(np.eye(3) + (np.eye(3) - np.ones((3, 3))) @ original / 4)
        assert determinant > 1


def test_exact_global_polynomial_and_curvature_controls() -> None:
    # Derive the rational determinant rather than trusting a sampled grid.
    import sympy as sp

    s = sp.Symbol("s")
    f = (1 - 2 / (s + 6) + 1 / (s + 12)) * (1 + 1 / (s + 6) + 1 / (4 * (s + 12))) ** 2
    numerator = sp.cancel((f - 1) * 16 * (s + 6) ** 3 * (s + 12) ** 3)
    assert sp.Poly(numerator, s).all_coeffs() == [24, 993, 15509, 111654, 353484, 344088]
    interval_numerator = sp.cancel(
        (f - sp.Rational(1083, 1024)) * 1024 * (s + 6) ** 3 * (s + 12) ** 3
    )
    assert (
        sp.expand(
            interval_numerator
            - s * (6106752 + 2099232 * s + 189704 * s**2 - 6540 * s**3 - 1650 * s**4 - 59 * s**5)
        )
        == 0
    )
    assert 6540 * 4 + 1650 * 4**2 + 59 * 4**3 < 189704
    assert Q(15, 128) ** 2 * 6 * Q(8, 7) ** 3 == Q(675, 5488) < Q(1, 4)
    assert Q(3) * Q(7, 22) * Q(59, 1083) == Q(413, 7942)


@pytest.mark.parametrize("seed", range(6))
def test_nonproduct_cosine_trial_oscillator_identity_and_local_bounds(seed: int) -> None:
    """Independent high-order quadrature diagnostic, not the analytic proof."""
    rng = np.random.default_rng(521 + seed)
    angle = rng.uniform(-1, 1)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    omega = rotation @ np.diag(rng.uniform(0.8, 1.5, size=2)) @ rotation.T
    kappa, eta, radius = 0.4, 0.2, 0.8
    k = omega @ omega - eta * np.eye(2)
    nodes, weights = leggauss(90)
    points = radius * np.array(list(product(nodes, repeat=2)))
    quadrature = radius**2 * np.array([a * b for a, b in product(weights, repeat=2)])
    a = np.pi / (2 * radius)
    cosine = np.cos(a * points)
    density = np.exp(-np.einsum("ni,ij,nj->n", points, omega, points) / kappa) * np.prod(
        cosine**2, axis=1
    )
    probability = quadrature * density
    probability /= probability.sum()
    tangent = np.tan(a * points)
    tangent_mean = probability @ tangent**2
    assert np.max(tangent_mean) <= 1 + 1e-12
    assert np.max(abs(probability @ points)) < 1e-14
    sigma2 = kappa / (2 * np.linalg.eigvalsh(omega).min())
    assert np.max(probability @ points**4) <= 5 * sigma2**2
    gradient = -(points @ omega) / kappa - a * tangent
    kinetic = kappa * np.sum(probability * np.sum(gradient**2, axis=1)) / 2
    potential = np.sum(probability * np.einsum("ni,ij,nj->n", points, k, points)) / (2 * kappa)
    expected = np.trace(omega) / 2 - eta * np.sum(probability * np.sum(points**2, axis=1)) / (
        2 * kappa
    )
    expected += kappa * a**2 * np.sum(tangent_mean) / 2
    assert abs(kinetic + potential - expected) < 2e-12


def test_even_trial_cancels_nonabelian_cubic_and_keeps_quartic_remainder() -> None:
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

    def action(vectors: np.ndarray) -> float:
        result = np.eye(2, dtype=complex)
        for v in vectors:
            radius = np.linalg.norm(v)
            matrix = (
                np.eye(2)
                if radius == 0
                else np.cos(radius / 2) * np.eye(2)
                + 1j * np.sin(radius / 2) * np.einsum("i,ijk->jk", v / radius, sigma)
            )
            result = result @ matrix
        return float(2 - np.trace(result).real)

    rng = np.random.default_rng(78127)
    samples = [rng.uniform(-0.4, 0.4, size=(4, 3)) for _ in range(40)]
    samples += [
        np.array([[x, 0, 0], [0, x, 0], [0, 0, x], [x, x, 0]]) for x in np.linspace(-0.5, 0.5, 21)
    ]
    for vectors in samples:
        quadratic = np.sum(vectors.sum(axis=0) ** 2) / 4
        cubic = (
            -sum(
                np.dot(vectors[i], np.cross(vectors[j], vectors[k]))
                for i in range(4)
                for j in range(i + 1, 4)
                for k in range(j + 1, 4)
            )
            / 4
        )
        error = np.sum(np.linalg.norm(vectors, axis=1)) ** 4 / 192
        assert abs(action(vectors) - quadratic - cubic) <= error + 2e-15
        assert abs((action(vectors) + action(-vectors)) / 2 - quadratic) <= error + 2e-15


def test_upper_density_constants_include_positive_metric_excess() -> None:
    assert (Q(64, 49) - 1) > 0
    assert 42 + Q(64, 49) * Q(9, 2) == Q(2346, 49)
    assert Q(16, 49) * Q(15, 2) + Q(64, 49) * Q(9, 8) * Q(22, 7) ** 2 == Q(40728, 2401)
    assert Q(2346, 49) + Q(40728, 2401) < 65
    for kappa in np.geomspace(1e-20, 1, 61):
        radius = np.sqrt(3) * kappa**0.25
        tangential = (radius / (2 * np.sin(radius / 2))) ** 2
        bound = (1 - kappa**0.5 / 8) ** -2
        assert 1 - 1e-15 <= tangential <= bound + 1e-15


@pytest.mark.parametrize("tau", [Q(1), Q(1, 8192), Q(1, 2**110)])
def test_actual_upper_replay_including_extremely_small_coupling(tau: Q) -> None:
    row = su2_periodic_upper_density(tau)
    assert row["status"] == "PASS"
    assert row["arithmetic"]["density_error_upper"] == str(65 * tau)
    assert row["arithmetic"]["kappa_upper"] == str(tau**3)
    assert row["actual_nonlinear_upper_verified_in_written_analysis"]
    assert row["certificate"]["payload"]["actual_nonlinear_upper_verified_in_written_analysis"]
    assert not row["actual_periodic_tilted_lower_density_verified"]
    assert not row["actual_periodic_vacuum_decorrelation_verified"]
    assert replay_periodic_corner_certificate(row["certificate"])


@pytest.mark.parametrize("errors", [(Q(1, 64), Q(1, 64)), (Q(0), Q(1, 32)), (Q(1, 16), Q(0))])
def test_conditional_budget_never_promotes_supplied_errors(errors: tuple[Q, Q]) -> None:
    row = periodic_corner_energy_budget(*errors)
    assert (row["status"] == "PASS") == (sum(errors) <= Q(1, 32))
    assert not row["actual_nonlinear_upper_verified_in_written_analysis"]
    assert not row["actual_periodic_vacuum_decorrelation_verified"]
    assert not row["arithmetic"]["input_error_hypotheses_verified"]
    assert Q(row["arithmetic"]["conditional_action_difference_over_kappa_lower"]) == 2 * (
        Q(29, 112) - 4 * sum(errors)
    )
    assert replay_periodic_corner_certificate(row["certificate"])


@pytest.mark.parametrize("value", [True, False, 0.25, "1/4", None, [], {}, (1, 2)])
def test_exact_input_guards(value: Any) -> None:
    with pytest.raises(TypeError):
        su2_periodic_upper_density(value)
    with pytest.raises(TypeError):
        periodic_corner_energy_budget(value, Q(0))


@pytest.mark.parametrize("value", [Q(0), Q(-1), Q(1001, 1000)])
def test_upper_domain_guards(value: Q) -> None:
    with pytest.raises(ValueError):
        su2_periodic_upper_density(value)


@pytest.mark.parametrize("value", [None, [], "bad", 1, True, (), {}])
def test_malformed_certificates_refuse(value: Any) -> None:
    assert not replay_periodic_corner_certificate(value)


def _reseal(cert: dict[str, Any]) -> dict[str, Any]:
    return make_certificate(claim=cert["claim"], payload=cert["payload"], meta=cert["meta"])


@pytest.mark.parametrize(
    "change",
    [
        "density",
        "metric",
        "zero_modes",
        "parent",
        "lower",
        "type",
        "input_bool",
        "input_spelling",
        "extra_input",
    ],
)
def test_resealed_mutations_are_rejected(change: str) -> None:
    cert = deepcopy(su2_periodic_upper_density()["certificate"])
    payload = cert["payload"]
    if change == "density":
        payload["arithmetic"]["density_error_upper"] = "0"
    elif change == "metric":
        payload["arithmetic"]["metric_multiplier_upper"] = "1"
    elif change == "zero_modes":
        payload["arithmetic"]["all_gauge_and_toron_coordinates_retained"] = False
    elif change == "parent":
        payload["yang_mills_mass_gap_claim"] = True
    elif change == "lower":
        payload["actual_periodic_tilted_lower_density_verified"] = True
    elif change == "type":
        payload["type"] = "other"
    elif change == "input_bool":
        payload["inputs"]["tau_upper"] = True
    elif change == "input_spelling":
        payload["inputs"]["tau_upper"] = "2/16384"
    else:
        payload["inputs"]["finite_volume"] = 3
    assert not replay_periodic_corner_certificate(_reseal(cert))


def test_harmonic_certificate_is_not_a_toron_uniform_or_nonlinear_claim() -> None:
    row = su2_periodic_corner_harmonic()
    assert row["harmonic_bound_verified_in_written_analysis"]
    assert row["arithmetic"]["reference_only"]
    assert not row["arithmetic"]["uniform_over_flat_backgrounds_verified"]
    assert not row["actual_periodic_upper_density_verified"]
    assert not row["physical_gap_claim"]
    assert replay_periodic_corner_certificate(row["certificate"])


def test_negative_budget_errors_are_refused() -> None:
    with pytest.raises(ValueError):
        periodic_corner_energy_budget(Q(-1), Q(0))


@pytest.mark.parametrize("side,ell", [(3, 2), (4, 2), (5, 2)])
def test_original_edge_box_replacement_is_positive_and_has_surface_rank(
    side: int, ell: int
) -> None:
    curl = _curl(side)
    vertices = list(product(range(side), repeat=3))
    quotient, remainder = divmod(side, ell)
    lengths = [ell + remainder, *[ell] * (quotient - 1)]
    ends: dict[int, int] = {}
    start = 0
    for length in lengths:
        ends.update({i: start + length for i in range(start, start + length)})
        start += length
    internal = np.array([x[axis] + 1 < ends[x[axis]] for x in vertices for axis in range(3)])
    t = 0.25
    face_metric = np.eye(3) + t * (np.eye(3) - np.ones((3, 3))) / 4
    full = np.zeros_like(curl)
    split = np.zeros_like(curl)
    crossings = replacements = 0
    for i in range(side**3):
        rows = curl[3 * i : 3 * i + 3]
        full += rows.T @ face_metric @ rows
        selected = np.array([np.all(internal[np.flatnonzero(row)]) for row in rows])
        if np.all(selected):
            split += rows.T @ face_metric @ rows
        else:
            crossings += 1
            replacements += int(np.count_nonzero(selected))
            split += (1 - t / 2) * rows.T @ np.diag(selected.astype(float)) @ rows
    difference_eigenvalues = np.linalg.eigvalsh(full - split)
    assert difference_eigenvalues.min() > -2e-14
    assert np.count_nonzero(difference_eigenvalues > 1e-10) <= 3 * crossings
    contained = sum((a - 1) * (b - 1) * (c - 1) for a, b, c in product(lengths, repeat=3))
    assert crossings == side**3 - contained
    assert replacements > 0
    assert Q(crossings, side**3) <= Q(3, ell)
    full_e = 1.5 * np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(full), 0)))
    split_e = 1.5 * np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(split), 0)))
    assert (full_e - split_e) / side**3 <= float(Q(54, ell))
    assert replay_periodic_corner_certificate(periodic_corner_box_surface(ell)["certificate"])


@pytest.mark.parametrize("ell,error", [(8192, Q(1, 128)), (4096, Q(1, 128)), (2, Q(0))])
def test_surface_and_conditional_lower_keep_the_missing_premise(ell: int, error: Q) -> None:
    surface = periodic_corner_box_surface(ell)
    assert Q(surface["arithmetic"]["surface_density_error_upper"]) == Q(54, ell)
    row = periodic_corner_box_lower_budget(ell, error)
    assert (row["status"] == "PASS") == (error + Q(54, ell) <= Q(1, 64))
    assert not row["actual_periodic_tilted_lower_density_verified"]
    assert not row["arithmetic"]["nonlinear_box_lower_premise_verified"]
    assert replay_periodic_corner_certificate(row["certificate"])


@pytest.mark.parametrize("bad", [True, 2.0, "2", None, 1, 0, -1])
def test_box_domain_and_type_guards(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        periodic_corner_box_surface(bad)


def test_resealed_surface_normalization_and_conditional_scope_attacks() -> None:
    for key, value in [("rank_per_crossing_cell_upper", 1), ("surface_density_error_upper", "0")]:
        cert = deepcopy(periodic_corner_box_surface()["certificate"])
        cert["payload"]["arithmetic"][key] = value
        assert not replay_periodic_corner_certificate(_reseal(cert))
    cert = deepcopy(periodic_corner_box_lower_budget()["certificate"])
    cert["payload"]["actual_periodic_tilted_lower_density_verified"] = True
    assert not replay_periodic_corner_certificate(_reseal(cert))
