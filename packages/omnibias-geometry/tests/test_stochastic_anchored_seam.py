# SPDX-License-Identifier: Apache-2.0
"""Independent original-link, Haar, free-word and gauge controls for the seam."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from random import Random
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.stochastic import anchored_seam as api
from omnibias.geometry.gauge.stochastic.lattice import Quaternion, _product
from omnibias.geometry.gauge.stochastic.lattice import quaternion_inverse as inv
from omnibias.geometry.gauge.stochastic.lattice import quaternion_product as mul

IDENTITY: Quaternion = (Q(1), Q(0), Q(0), Q(0))


def _rational_unit(rng: Random) -> Quaternion:
    v = [Q(rng.randint(-4, 4), 5) for _ in range(3)]
    square = sum((x * x for x in v), Q(0))
    return (
        (1 - square) / (1 + square),
        2 * v[0] / (1 + square),
        2 * v[1] / (1 + square),
        2 * v[2] / (1 + square),
    )


def _field(seed: int) -> list[Quaternion]:
    rng = Random(seed)
    return [_rational_unit(rng) for _ in range(12)]


def _word(field: list[Quaternion], word: list[int]) -> Quaternion:
    return _product([field[e - 1] if e > 0 else inv(field[-e - 1]) for e in word])


def _mp(q: Q | str | int) -> Any:
    value = Q(q)
    return mp.mpf(value.numerator) / value.denominator


def _extend(
    old: list[Quaternion], y1: Quaternion, y2: Quaternion, h: Quaternion = IDENTITY
) -> list[Quaternion]:
    graph = api.anchored_three_face_geometry()
    anchors = [_word(old, word) for word in graph["anchors_to_old_root"]]
    out = list(old)
    for edge, anchor, y in zip(graph["added_links"], anchors, (IDENTITY, y1, y2), strict=True):
        out[edge - 1] = mul(mul(anchor, y), h)
    return out


def test_free_word_geometry_and_original_link_accounting() -> None:
    graph = api.anchored_three_face_geometry()
    assert len(graph["old_edges"]) == 9 and len(graph["new_edges"]) == 3
    assert graph["new_vertex"] == 7
    assert graph["reduced_old_face_identities"] == graph["old_faces"]
    assert graph["old_relative_incidences"] == 8
    assert graph["new_relative_incidences"] == 4
    assert sorted(graph["original_link_relative_incidences"]) == [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ]
    assert all(len(set(map(abs, word))) == 6 for word in graph["relative_cycles"])


@pytest.mark.parametrize("seed", [0, 19, 93, 127, 519, 8192])
def test_noncommuting_old_new_faces_and_relative_variables(seed: int) -> None:
    old, rng = _field(seed), Random(seed + 937)
    y1, y2, h = (_rational_unit(rng) for _ in range(3))
    field = _extend(old, y1, y2, h)
    graph = api.anchored_three_face_geometry()
    assert [_word(field, word) for word in graph["relative_cycles"]] == [y1, y2]
    old_faces = [_word(field, word) for word in graph["old_faces"]]
    expected = [
        mul(old_faces[0], y1),
        mul(mul(inv(y1), old_faces[1]), y2),
        mul(inv(y2), old_faces[2]),
    ]
    assert [2 * _word(field, word)[0] for word in graph["new_faces"]] == [
        2 * x[0] for x in expected
    ]
    zero_error = _extend(old, IDENTITY, IDENTITY, h)
    assert [2 * _word(zero_error, word)[0] for word in graph["new_faces"]] == [
        2 * x[0] for x in old_faces
    ]


@pytest.mark.parametrize("seed", [5, 19, 7541])
def test_full_vertex_gauge_covariance_exact(seed: int) -> None:
    field, rng = _field(seed), Random(seed + 1)
    gauges = [_rational_unit(rng) for _ in range(8)]
    edges = api.anchored_three_face_geometry()["edges"]
    transformed = [
        mul(mul(gauges[u], q), inv(gauges[v])) for (u, v), q in zip(edges, field, strict=True)
    ]
    before, after = (
        api.anchored_seam_point(x)["witness"]["arithmetic"] for x in (field, transformed)
    )
    for key in (
        "density",
        "relative_traces",
        "old_face_traces",
        "new_face_traces",
        "old_log_amplitude_gradient_squared",
        "new_log_amplitude_gradient_squared",
    ):
        assert before[key] == after[key]


@pytest.mark.parametrize("seed", [8, 273, 5192])
def test_all_twelve_original_link_derivatives_independent_group_curves(seed: int) -> None:
    field = _field(seed)
    row = api.anchored_seam_point(field)["witness"]["arithmetic"]
    density, h = Q(row["density"]), Q(1, 10**7)
    for edge in range(12):
        for axis in range(3):
            values = []
            for sign in (1, -1):
                # This exact rational SU2 curve has velocity4*T_axis at0;
                # the factor16 also includes d(log sqrt(rho))=d rho/(2rho).
                rotation = [
                    (1 - h * h) / (1 + h * h),
                    *[2 * sign * h / (1 + h * h) if j == axis else Q(0) for j in range(3)],
                ]
                perturbed = list(field)
                perturbed[edge] = mul(rotation, field[edge])
                values.append(
                    Q(api.anchored_seam_point(perturbed)["witness"]["arithmetic"]["density"])
                )
            oracle = (values[0] - values[1]) / (16 * h * density)
            actual = Q(row["log_amplitude_gradient"][edge][axis])
            assert abs(actual - oracle) < 100 * h * h


def _axes(w: Q, radius: Q) -> list[Quaternion]:
    return [
        (
            w,
            sign * radius if axis == 0 else Q(0),
            sign * radius if axis == 1 else Q(0),
            sign * radius if axis == 2 else Q(0),
        )
        for axis in range(3)
        for sign in (-1, 1)
    ]


@pytest.mark.parametrize("seed", [0, 413])
def test_exact_orientation_cubature_cancels_shared_row_scores(seed: int) -> None:
    old, r = _field(seed), Q(1, 3)
    c = r / (1 + r * r)
    w1, w2 = Q(3, 5), Q(5, 13)
    single = [c * c * (1 - w * w) / (1 + 2 * c * w) ** 2 for w in (w1, w2)]
    averages = [Q(0), Q(0)]
    expected_traces = [Q(0), Q(0), Q(0)]
    for y1, y2 in product(_axes(w1, Q(4, 5)), _axes(w2, Q(12, 13))):
        row = api.anchored_seam_point(_extend(old, y1, y2), tilt=r)["witness"]["arithmetic"]
        averages[0] += Q(row["old_log_amplitude_gradient_squared"]) / 36
        averages[1] += Q(row["new_log_amplitude_gradient_squared"]) / 36
        expected_traces = [
            a + Q(b) / 36 for a, b in zip(expected_traces, row["new_face_traces"], strict=True)
        ]
    assert averages == [sum(single), sum(single) / 2]
    graph = api.anchored_three_face_geometry()
    old_traces = [2 * _word(old, word)[0] for word in graph["old_faces"]]
    assert expected_traces == [w1 * old_traces[0], w1 * w2 * old_traces[1], w2 * old_traces[2]]


@pytest.mark.parametrize("tilt", [Q(-2, 3), Q(0), Q(1, 3), Q(4, 5)])
def test_integrated_profile_against_independent_haar_quadrature(tilt: Q) -> None:
    old_traces = [Q(2), Q(-7, 9), Q(-2)]
    row = api.anchored_seam_control(old_traces, tilt=tilt)["witness"]["arithmetic"]
    with mp.workdps(60):
        c = _mp(tilt) / (1 + _mp(tilt) ** 2)

        def haar(fun: Any) -> Any:
            return mp.quad(
                lambda u: 2 / mp.pi * mp.sin(u) ** 2 * (1 + 2 * c * mp.cos(u)) * fun(u),
                [0, mp.pi / 2, mp.pi],
            )

        assert abs(haar(lambda u: 1) - 1) < mp.mpf("1e-55")
        mean_half_trace = haar(mp.cos)
        fisher = haar(lambda u: c * c * mp.sin(u) ** 2 / (1 + 2 * c * mp.cos(u)) ** 2)
        assert abs(fisher - _mp(row["single_increment_fisher"])) < mp.mpf("1e-55")
        dampings = [mean_half_trace, mean_half_trace**2, mean_half_trace]
        for actual, damping, trace in zip(
            row["expected_added_face_actions"], dampings, old_traces, strict=True
        ):
            assert abs(_mp(actual) - (2 - damping * _mp(trace))) < mp.mpf("1e-55")


@pytest.mark.parametrize("kappa", [Q(1, 10**20), Q(1, 64), Q(1), Q(10000)])
@pytest.mark.parametrize("scale", [Q(1, 4), Q(1, 2), Q(1), Q(4)])
def test_all_coupling_exact_gate(kappa: Q, scale: Q) -> None:
    report = api.su2_anchored_three_face_seam(kappa, heat_time_scale=scale)
    a = report["witness"]["arithmetic"]
    assert Q(a["constant_upper"]) == 12 * scale + Q(9, 4) / scale
    if scale == Q(1, 2):
        assert a["constant_upper"] == "21/2"
    assert a["action_feedback_coefficient"] == "1"
    assert Q(a["total_fisher_upper"]) == Q(9, 2) / (kappa * scale)
    assert report["actual_anchored_form_comparison_verified"] is True
    assert report["strict_action_contraction_verified"] is False
    assert api.replay_anchored_seam_certificate(report["certificate"])


def test_heat_magnetic_dampings_and_uniform_excess_grid_and_random() -> None:
    rng = Random(72945)
    with mp.workdps(65):
        times = [Q(1, 2**n) for n in range(0, 20)] + [
            Q(rng.randint(1, 10000), 100) for _ in range(24)
        ]
        for t in times:
            random_traces = tuple(Q(rng.randint(-200, 200), 100) for _ in range(3))
            packs = [(Q(2), Q(2), Q(2)), (Q(-2), Q(-2), Q(-2)), random_traces]
            for traces in packs:
                excess = mp.fsum(
                    _mp(chi) * (1 - mp.exp(-_mp(multiplier * t)))
                    for chi, multiplier in zip(traces, (Q(3, 4), Q(3, 2), Q(3, 4)), strict=True)
                )
                assert excess <= 6 * _mp(t)


@pytest.mark.parametrize("kind", ["heat", "control", "point"])
@pytest.mark.parametrize(
    "attack", ["arithmetic", "geometry", "scope", "input", "proof", "bool_type"]
)
def test_canonical_rehash_attacks(kind: str, attack: str) -> None:
    if kind == "heat":
        report = api.su2_anchored_three_face_seam(Q(1, 64))
    elif kind == "control":
        report = api.anchored_seam_control([1, 0, -1])
    else:
        report = api.anchored_seam_point(_field(293))
    cert = deepcopy(report["certificate"])
    witness = cert["payload"]["witness"]
    if attack == "arithmetic":
        witness["arithmetic"]["conditional_normalizer"] = "2"
    elif attack == "geometry":
        witness["geometry"]["relative_cycles"][0][0] *= -1
    elif attack == "scope":
        cert["honesty"]["strict_action_contraction_verified"] = True
    elif attack == "input":
        witness["inputs"]["kappa" if kind == "heat" else "tilt"] = "1/17"
    elif attack == "proof":
        witness["proof_register"] = "Lean proved continuum YM"
    else:
        cert["honesty"]["finite_gate_verified"] = 1
    assert not api.replay_anchored_seam_certificate(seal_certificate(cert))


def test_nested_profile_source_rehash_attack() -> None:
    cert = deepcopy(api.anchored_seam_control([2, 1, 0])["certificate"])
    a = cert["payload"]["witness"]["arithmetic"]
    source = a["profile_source_certificate"]
    source["payload"]["arithmetic"]["one_face_fisher"] = "0"
    a["profile_source_certificate"] = seal_certificate(source)
    assert not api.replay_anchored_seam_certificate(seal_certificate(cert))


@pytest.mark.parametrize("value", [True, False, 0, -1, 0.5, "1/2"])
def test_exact_positive_coupling_and_scale_guards(value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        api.su2_anchored_three_face_seam(value)
    with pytest.raises((ValueError, TypeError)):
        api.su2_anchored_three_face_seam(1, heat_time_scale=value)


@pytest.mark.parametrize("value", [None, [], (), "cert", True, 4])
def test_replay_nonmapping_guard(value: Any) -> None:
    assert api.replay_anchored_seam_certificate(value) is False


def test_control_scope_and_bad_fields() -> None:
    for report in (api.anchored_seam_control([0, 0, 0]), api.anchored_seam_point(_field(9))):
        assert report["actual_anchored_form_comparison_verified"] is False
        assert report["actual_conditional_vacuum_identified"] is False
        assert report["continuum_claim"] is False
        assert report["yang_mills_mass_gap_claim"] is False
        assert report["mathlib_verified"] is False
        assert api.replay_anchored_seam_certificate(report["certificate"])
    with pytest.raises(ValueError):
        api.anchored_seam_control([0, 0, 3])
    with pytest.raises(ValueError):
        api.anchored_seam_control([0, 0])
    with pytest.raises(ValueError):
        api.anchored_seam_point([IDENTITY] * 11)
    with pytest.raises(ValueError):
        api.anchored_seam_point([IDENTITY] * 11 + [(Q(1), Q(1), Q(0), Q(0))])
