# SPDX-License-Identifier: Apache-2.0
"""Global quaternion differentiation, Haar budgets and canonical source guards."""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from math import factorial
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import theta_quasimode as quasi
from omnibias.geometry.gauge.transfer import theta_weak_blocks as weak

Quat = tuple[Q, Q, Q, Q]


def _mul(x: Quat, y: Quat) -> Quat:
    return (
        x[0] * y[0] - sum((x[i] * y[i] for i in range(1, 4)), Q(0)),
        x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
        x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
        x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0],
    )


def _stereo(v: tuple[Q, Q, Q]) -> Quat:
    r = sum((z * z for z in v), Q(0))
    return (1 - r) / (1 + r), 2 * v[0] / (1 + r), 2 * v[1] / (1 + r), 2 * v[2] / (1 + r)


def _generator(axis: int) -> Quat:
    return Q(0), Q(axis == 0, 2), Q(axis == 1, 2), Q(axis == 2, 2)


def _dot(x: Quat, y: Quat) -> Q:
    return sum((x[i] * y[i] for i in range(1, 4)), Q(0))


def _phase_derivatives(
    x: Quat, y: Quat, dx: Quat, dy: Quat, ddx: Quat, ddy: Quat, a: Q, b: Q
) -> tuple[Q, Q]:
    first = -2 * a * (dx[0] + dy[0]) + b * (_dot(dx, y) + _dot(x, dy))
    second = -2 * a * (ddx[0] + ddy[0]) + b * (_dot(ddx, y) + 2 * _dot(dx, dy) + _dot(x, ddy))
    return first, second


_POINTS: tuple[Quat, ...] = (
    (Q(1), Q(0), Q(0), Q(0)),
    (Q(-1), Q(0), Q(0), Q(0)),
    (Q(0), Q(1), Q(0), Q(0)),
    _stereo((Q(1, 3), Q(-2, 5), Q(1, 7))),
    _stereo((Q(2), Q(1, 2), Q(-1))),
)


@pytest.mark.parametrize("a,b", [(Q(1), Q(0)), (Q(3, 2), Q(-1, 4)), (Q(0), Q(2))])
def test_independent_left_quaternion_first_and_second_derivatives(a: Q, b: Q) -> None:
    zero: Quat = (Q(0), Q(0), Q(0), Q(0))
    for x, y in product(_POINTS, repeat=2):
        gamma, casimir = Q(0), Q(0)
        for axis in range(3):
            dx, dy = _mul(_generator(axis), x), _mul(_generator(axis), y)
            ddx, ddy = _mul(_generator(axis), dx), _mul(_generator(axis), dy)
            for weight, jx, jy, jjx, jjy in (
                (3, dx, zero, ddx, zero),
                (3, zero, dy, zero, ddy),
                (1, dx, dy, ddx, ddy),
            ):
                first, second = _phase_derivatives(x, y, jx, jy, jjx, jjy, a, b)
                gamma += weight * first * first
                casimir -= weight * second
        s, t, w = x[0], y[0], _dot(x, y)
        u, v, action = 1 - s * s, 1 - t * t, 4 - 2 * s - 2 * t
        sum_gradient = (a * a + b * b / 4) * (u + v) + a * b * (s + t) * w - b * b * w * w / 2
        diagonal = (a + b * t / 2) ** 2 * u + (a + b * s / 2) ** 2 * v
        diagonal += 2 * (a + b * t / 2) * (a + b * s / 2) * w
        assert gamma == 3 * sum_gradient + diagonal
        assert casimir == a * (3 * action - 12) + b * (5 * w - 3 * s * t / 2)


def test_exact_oscillator_cancellation_in_quadratic_radical_field() -> None:
    # Pairs represent c+d/sqrt(15), so no numerical radical is sampled.
    aa, ab, bb = (Q(8, 15), Q(2)), (Q(-4, 15), Q(0)), (Q(32, 15), Q(-8))
    assert tuple(4 * aa[i] + ab[i] + bb[i] for i in range(2)) == (Q(4), Q(0))
    assert tuple(2 * aa[i] + 8 * ab[i] + bb[i] / 2 for i in range(2)) == (Q(0), Q(0))
    for x, y in product(_POINTS, repeat=2):
        s, t, w = x[0], y[0], _dot(x, y)
        u, v = 1 - s * s, 1 - t * t
        for i in range(2):
            exact = 4 * aa[i] * (u + v) + bb[i] * (3 * (u + v) + t * t * u + s * s * v) / 4
            exact += ab[i] * (t * u + s * v + 4 * (s + t) * w)
            exact += (2 * aa[i] + bb[i] * s * t / 2) * w - 3 * bb[i] * w * w / 2
            simplified = (4 * (u + v) if i == 0 else Q(0)) + ab[i] * (
                (t - 1) * u + (s - 1) * v + 4 * (s + t - 2) * w
            )
            simplified += bb[i] / 4 * (-2 * u * v - 6 * w * w + 2 * (s * t - 1) * w)
            assert exact == simplified


def test_haar_moments_and_residual_square_budget() -> None:
    moments = [Q(512, 3) * factorial(m + 2) * Q(9, 16) ** (m + 3) for m in (2, 3, 4)]
    assert moments == [Q(59049, 256), Q(2657205, 4096), Q(71744535, 32768)]
    square = Q(25, 4) * moments[0] + 5 * moments[1] + moments[2]
    assert square == Q(225271935, 32768) < 83**2
    assert 1152 * Q(22, 7) < 4096
    assert 664 + 2**20 < 2**21


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return quasi.su2_theta_quasimode_error()


def test_default_source_arithmetic_replay_and_scope(report: dict[str, Any]) -> None:
    assert report["status"] == "PASS"
    a = report["arithmetic"]
    assert a["residual_l2_upper"] == "83/64"
    assert a["vacuum_l2_upper"] == "83/16"
    assert a["vacuum_density_l1_upper"] == "83/8"
    assert Q(a["source_first_reduced_excitation_lower"]) > Q(32, 5)
    assert a["residual_l2_squared_coefficient_upper"] == "225271935/32768"
    assert all(a["gates"].values())
    assert report["actual_vacuum_density_l1_error_verified_in_written_analysis"] is True
    assert report["certificate"]["payload"] == {
        k: v for k, v in report.items() if k != "certificate"
    }
    assert quasi.replay_su2_theta_quasimode_certificate(report["certificate"])
    for flag in (
        "density_error_is_normalized_kernel_error",
        "explicit_maximal_correlation_threshold_verified",
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
    ):
        assert report[flag] is False


@pytest.mark.parametrize("kappa", [Q(1, 1024), Q(1, 2**40), Q(3, 4096)])
def test_exact_requested_error_scales_linearly(kappa: Q) -> None:
    r = quasi.su2_theta_quasimode_error(kappa)
    assert r["status"] == "PASS"
    assert Q(r["arithmetic"]["vacuum_density_l1_upper"]) == 664 * kappa
    assert quasi.replay_su2_theta_quasimode_certificate(r["certificate"])


@pytest.mark.parametrize("kappa", [Q(1, 63), Q(1, 4), 1, 5])
def test_outside_interval_is_honest_inconclusive(kappa: int | Q) -> None:
    r = quasi.su2_theta_quasimode_error(kappa)
    assert r["status"] == "INCONCLUSIVE"
    assert r["arithmetic"]["residual_l2_upper"] is None
    assert r["arithmetic"]["vacuum_l2_upper"] is None
    assert r["arithmetic"]["vacuum_density_l1_upper"] is None
    assert not r["actual_quasimode_error_verified_in_written_analysis"]
    assert not r["actual_vacuum_l2_error_verified_in_written_analysis"]
    assert quasi.replay_su2_theta_quasimode_certificate(r["certificate"])


@pytest.mark.parametrize("bad", [True, False, 0.01, "1/64", None, complex(1)])
def test_inexact_or_malformed_input_refused(bad: Any) -> None:
    with pytest.raises(TypeError):
        quasi.su2_theta_quasimode_error(bad)


@pytest.mark.parametrize("bad", [0, -1, Q(-1, 64)])
def test_nonpositive_refused(bad: int | Q) -> None:
    with pytest.raises(ValueError):
        quasi.su2_theta_quasimode_error(bad)


@pytest.mark.parametrize("field", ["arithmetic", "source", "scope", "input", "phase", "metadata"])
def test_resealed_tampering_is_not_canonical(report: dict[str, Any], field: str) -> None:
    certificate = deepcopy(report["certificate"])
    p = certificate["payload"]
    if field == "arithmetic":
        p["arithmetic"]["residual_l2_coefficient_upper"] = "1"
    elif field == "source":
        source = p["theta_weak_source_certificate"]
        source["payload"]["arithmetic"]["first_reduced_excitation_lower"] = "100"
        p["theta_weak_source_certificate"] = seal_certificate(source)
    elif field == "scope":
        p["uniform_in_volume_claim"] = True
    elif field == "input":
        p["inputs"]["kappa"] = "1/128"
    elif field == "phase":
        p["phase"] = "F=S"
    else:
        certificate["meta"]["analytic_implication"] = "different.md"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not quasi.replay_su2_theta_quasimode_certificate(certificate)


@pytest.mark.parametrize("damage", ["energy", "scope", "coupling", "missing_certificate"])
def test_bad_nested_source_cannot_earn_actual_error(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original = weak.su2_theta_weak_block_gaps()
    bad = deepcopy(original)
    if damage == "missing_certificate":
        bad["certificate"] = None
    else:
        p = bad["certificate"]["payload"]
        if damage == "energy":
            p["arithmetic"]["first_reduced_excitation_lower"] = "1"
        elif damage == "scope":
            p["actual_theta_vacuum_identified_in_written_analysis"] = False
        else:
            p["inputs"]["kappa"] = "1/128"
        bad["certificate"] = seal_certificate(bad["certificate"])

    def producer(kappa: int | Q = Q(1, 64)) -> dict[str, Any]:
        del kappa
        return bad

    monkeypatch.setattr(weak, "su2_theta_weak_block_gaps", producer)
    result = quasi.su2_theta_quasimode_error()
    assert result["status"] == "INCONCLUSIVE"
    assert result["arithmetic"]["vacuum_density_l1_upper"] is None
    assert not result["actual_vacuum_l2_error_verified_in_written_analysis"]


@pytest.mark.parametrize("bad", [{}, {"payload": {}}, {"payload": {"type": "wrong"}}])
def test_malformed_replay_refused(bad: dict[str, Any]) -> None:
    assert not quasi.replay_su2_theta_quasimode_certificate(bad)
