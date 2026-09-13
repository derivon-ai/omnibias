# SPDX-License-Identifier: Apache-2.0
"""Analytic-budget checks, normalized-kernel controls and adversarial replay."""

import json
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, exp, pi, sin, sqrt
from random import Random
from typing import Any

import numpy as np
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import theta_kernel_threshold as module


def test_explicit_interval_includes_boundary_without_materializing_power() -> None:
    row = module.su2_theta_kernel_contraction()
    assert row["status"] == "PASS"
    assert row["inputs"]["dyadic_exponent"] == 17_022_271
    assert row["actual_radial_maximal_correlation_upper"] == "143/480"
    assert row["actual_full_cycle_maximal_correlation_upper"] == "95/224"
    assert row["arithmetic"]["coupling_exponent_margin"] == 0
    assert row["explicit_coupling_threshold_verified_in_written_analysis"]
    assert module.replay_su2_theta_kernel_contraction_certificate(row["certificate"])
    tiny = module.su2_theta_kernel_contraction(dyadic_exponent=10**30)
    assert tiny["status"] == "PASS"
    assert len(json.dumps(tiny)) < 100_000
    assert not tiny["coupling_family"]["floating_threshold_materialized"]


@pytest.mark.parametrize("n", (1, 19, 20, 17_022_270))
def test_interval_too_large_is_inconclusive(n: int) -> None:
    row = module.su2_theta_kernel_contraction(dyadic_exponent=n)
    assert row["status"] == "INCONCLUSIVE"
    assert row["actual_radial_maximal_correlation_upper"] is None
    assert not row["actual_radial_contraction_verified_in_written_analysis"]
    assert module.replay_su2_theta_kernel_contraction_certificate(row["certificate"])


@pytest.mark.parametrize("bad", (True, False, None, 1.0, Q(3), "17022271", [], {}))
def test_nondyadic_exponent_types_refused(bad: Any) -> None:
    with pytest.raises(TypeError):
        module.su2_theta_kernel_contraction(dyadic_exponent=bad)


@pytest.mark.parametrize("bad", (0, -1))
def test_nonpositive_exponent_refused(bad: int) -> None:
    with pytest.raises(ValueError):
        module.su2_theta_kernel_contraction(dyadic_exponent=bad)


@pytest.mark.parametrize("damage", ("interval", "bound", "source", "tail", "formal", "continuum"))
def test_resealed_forgery_cannot_change_interval_or_analytic_scope(damage: str) -> None:
    cert = deepcopy(module.su2_theta_kernel_contraction()["certificate"])
    p = cert["payload"]
    if damage == "interval":
        p["inputs"]["dyadic_exponent"] = 20
    elif damage == "bound":
        p["actual_radial_maximal_correlation_upper"] = "0"
    elif damage == "source":
        p["quasimode_source_certificate"]["payload"]["arithmetic"][
            "residual_l2_coefficient_upper"
        ] = "0"
    elif damage == "tail":
        p["normalized_tail_source_certificate"]["payload"]["tail_region_has_zero_haar_measure"] = (
            True
        )
    elif damage == "formal":
        p["mathlib_verified"] = True
    else:
        p["continuum_claim"] = True
    assert not module.replay_su2_theta_kernel_contraction_certificate(seal_certificate(cert))


@pytest.mark.parametrize("bad", (None, True, [], {}, "certificate"))
def test_malformed_certificate_refused(bad: Any) -> None:
    assert not module.replay_su2_theta_kernel_contraction_certificate(bad)


def test_detached_quasimode_summary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibias.geometry.gauge.transfer import theta_quasimode as source

    original = source.su2_theta_quasimode_error

    def altered(kappa: Q) -> dict[str, Any]:
        row = deepcopy(original(kappa))
        row["arithmetic"]["residual_l2_coefficient_upper"] = "0"
        return row

    monkeypatch.setattr(source, "su2_theta_quasimode_error", altered)
    row = module.su2_theta_kernel_contraction()
    assert row["status"] == "INCONCLUSIVE"
    assert not row["explicit_coupling_threshold_verified_in_written_analysis"]


def test_exact_budget_chain_and_nonempty_tail() -> None:
    row = module.su2_theta_kernel_contraction()
    a = row["arithmetic"]
    assert a["log_amplification"] == 4_255_560
    assert 4 * a["log_amplification"] + 21 + 10 == 17_022_271
    assert 2 * a["dyadic_amplification_exponent"] + 21 - 17_022_271 == -4
    assert 664 + 2**20 == 1_049_240 < 2**21
    assert Q(43, 2880) < Q(1, 64)
    assert 32_768 * (Q(24, 9) + Q(768, 64)) < 2**19
    assert Q(81_640) - Q(2048**2, 45) == -Q(520_504, 45)
    assert 26_000_000 < 2**25
    assert 520_504 // 45 - 25 > 12
    tail = row["normalized_tail_source_certificate"]["payload"]
    assert not tail["tail_region_has_zero_haar_measure"]
    assert Q(tail["inputs"]["kappa"]) * Q(tail["inputs"]["radius"]) == Q(2, 5)
    assert Q(1, 60) + Q(9, 32) == Q(143, 480) < Q(3, 10)
    assert Q(1, 7) + Q(9, 32) == Q(95, 224) < Q(1, 2)
    for field in (
        "ambient_exterior_uniformity_verified",
        "uniform_in_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_mass_gap_claim",
        "analytic_proof_formally_verified",
        "mathlib_verified",
    ):
        assert not row[field]


def test_exponential_coordinate_phase_and_jacobian_grid_and_random_samples() -> None:
    rng = Random(49201)
    points = [
        (pi * i / 10, pi * j / 10, c) for i in range(11) for j in range(11) for c in (-1, 0, 1)
    ]
    points.extend((rng.uniform(0, pi), rng.uniform(0, pi), rng.uniform(-1, 1)) for _ in range(128))
    a, b = 1 / sqrt(3) + 1 / sqrt(5), 2 * (1 / sqrt(5) - 1 / sqrt(3))
    kappa = 1 / 64
    for x, y, c in points:
        r2 = 4 * (x * x + y * y) / kappa
        action = 4 - 2 * cos(x) - 2 * cos(y)
        phase = a * action + b * sin(x) * sin(y) * c
        quadratic = 2 * a * (x * x + y * y) / kappa + 2 * b * x * y * c / kappa
        assert abs(2 * phase / kappa - quadratic) <= kappa * r2 * r2 / 64 + 1e-10
        jx = (sin(x) / x) ** 2 if x else 1
        jy = (sin(y) / y) ** 2 if y else 1
        assert 0 <= 1 - jx * jy <= kappa * r2 / 12 + 1e-12
        assert min(2 * phase / kappa, quadratic) >= r2 / 8 - 1e-10


def test_compact_normalized_kernel_l1_bridge_on_varied_positive_densities() -> None:
    rng = np.random.default_rng(17022271)
    for epsilon in [*np.linspace(0, 1, 41), *rng.uniform(0, 1, 64)]:
        q = rng.lognormal(0, 1.5, size=(7, 9))
        q /= q.sum()
        other = rng.lognormal(0, 2, size=q.shape)
        other /= other.sum()
        p = (1 - epsilon) * q + epsilon * other
        px, py, qx, qy = p.sum(1), p.sum(0), q.sum(1), q.sum(0)
        tau = float(abs(p - q).sum())
        # A strict sub-box tests that the own marginals are taken globally.
        ix, iy = slice(1, 5), slice(2, 8)
        upper = float(max(p[ix, iy].max(), q[ix, iy].max()))
        lower = float(min(px[ix].min(), py[iy].min(), qx[ix].min(), qy[iy].min()))
        kp, kq = p / np.sqrt(px[:, None] * py), q / np.sqrt(qx[:, None] * qy)
        error = float(np.linalg.norm((kp - kq)[ix, iy]))
        assert error <= 3 * sqrt(upper * tau) / lower + 1e-14
        center_error = np.linalg.norm(np.sqrt(px[:, None] * py) - np.sqrt(qx[:, None] * qy))
        assert center_error <= 2 * sqrt(tau) + 1e-14


def test_gaussian_reference_kernel_precision_and_marginal_budget() -> None:
    variance = (sqrt(3) + sqrt(5)) / 4
    precision = np.linalg.inv(np.array([[4.0, 1.0], [1.0, 4.0]]))
    values, vectors = np.linalg.eigh(precision)
    b = (vectors * np.sqrt(values)) @ vectors.T
    assert 19 / 20 < variance < 1
    assert np.linalg.eigvalsh(2 * b - np.eye(2) / (2 * variance))[0] > 1 / 3
    assert 625 * 16**2 * Q(22, 7) ** 4 < 2**24
    for r in np.linspace(0, 5, 101):
        actual = (2 * pi * variance) ** (-1.5) * exp(-r * r / (2 * variance))
        assert actual >= exp(-5 - r * r)
