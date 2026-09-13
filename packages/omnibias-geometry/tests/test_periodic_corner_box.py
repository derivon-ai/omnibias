# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact arithmetic and independent finite geometry controls for the box theorem.

Finite algebra regressions support, but do not replace, the analytic proof.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product
from random import Random

import pytest
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.periodic_corner_box import (
    _ceil_log2,
    replay_su2_periodic_corner_box_lower_certificate,
    su2_periodic_corner_box_lower,
)

Vertex = tuple[int, int, int]
Edge = tuple[Vertex, Vertex]
Quaternion = tuple[Q, Q, Q, Q]
IDENTITY: Quaternion = (Q(1), Q(0), Q(0), Q(0))


def _rank(rows: Sequence[Sequence[Q]]) -> int:
    work = [list(row) for row in rows]
    pivot = 0
    for column in range(len(work[0]) if work else 0):
        selected = next((i for i in range(pivot, len(work)) if work[i][column]), None)
        if selected is None:
            continue
        work[pivot], work[selected] = work[selected], work[pivot]
        scale = work[pivot][column]
        work[pivot] = [entry / scale for entry in work[pivot]]
        for i in range(len(work)):
            if i != pivot:
                factor = work[i][column]
                work[i] = [x - factor * y for x, y in zip(work[i], work[pivot], strict=True)]
        pivot += 1
        if pivot == len(work):
            break
    return pivot


def _det(rows: Sequence[Sequence[Q]]) -> Q:
    if not rows:
        return Q(1)
    if len(rows) == 1:
        return rows[0][0]
    return sum(
        (
            (-1) ** j
            * rows[0][j]
            * _det([[entry for k, entry in enumerate(row) if k != j] for row in rows[1:]])
            for j in range(len(rows))
        ),
        Q(0),
    )


def _step(vertex: Vertex, axis: int) -> Vertex:
    return (vertex[0] + int(axis == 0), vertex[1] + int(axis == 1), vertex[2] + int(axis == 2))


def _geometry(sides: Vertex) -> tuple[list[Vertex], list[Edge], list[list[Vertex]]]:
    vertices = [
        (x, y, z) for x in range(sides[0]) for y in range(sides[1]) for z in range(sides[2])
    ]
    edges = [
        (v, _step(v, axis)) for v in vertices for axis in range(3) if v[axis] + 1 < sides[axis]
    ]
    faces = []
    for v in vertices:
        for a, b in combinations(range(3), 2):
            if v[a] + 1 < sides[a] and v[b] + 1 < sides[b]:
                faces.append([v, _step(v, a), _step(_step(v, a), b), _step(v, b)])
    return vertices, edges, faces


def _face_row(face: list[Vertex], edges: list[Edge]) -> list[Q]:
    row = [Q(0)] * len(edges)
    for a, b in zip(face, [*face[1:], face[0]], strict=True):
        if (a, b) in edges:
            row[edges.index((a, b))] += 1
        else:
            row[edges.index((b, a))] -= 1
    return row


def _mul(a: Quaternion, b: Quaternion) -> Quaternion:
    return (
        a[0] * b[0] - sum((a[i] * b[i] for i in range(1, 4)), Q(0)),
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def _stereo(a: Sequence[Q]) -> Quaternion:
    norm = sum((x * x for x in a), Q(0))
    return (
        (1 - norm) / (1 + norm),
        2 * a[0] / (1 + norm),
        2 * a[1] / (1 + norm),
        2 * a[2] / (1 + norm),
    )


def test_default_exact_endpoint_and_written_scope() -> None:
    row = su2_periodic_corner_box_lower()
    a = row["arithmetic"]
    assert row["status"] == "PASS"
    assert (a["chart_radius_bits"], a["raw_default_kappa_exponent"], a["kappa_exponent"]) == (
        107,
        328,
        330,
    )
    assert Q(a["kappa_upper"]) == Q(1, 2**330)
    assert Q(a["tau_upper"]) ** 3 == Q(a["kappa_upper"])
    assert a["box_vertex_side_interval"] == [8192, 16383]
    assert a["tilt_interval"] == ["0", "1/4"]
    assert Q(a["surface_density_error_upper"]) == Q(27, 4096)
    assert Q(a["nonlinear_density_error_upper"]) < Q(1, 128)
    assert Q(a["total_density_error_upper"]) == Q(a["nonlinear_density_error_upper"]) + Q(27, 4096)
    assert Q(a["total_density_error_upper"]) < Q(1, 64)
    assert all(a["gates"].values())
    assert row["actual_nonlinear_lower_verified_in_written_analysis"]
    assert row["certificate"]["payload"] == {k: v for k, v in row.items() if k != "certificate"}
    for key in (
        "analytic_proof_formally_verified",
        "arbitrary_exterior_conditional_verified",
        "physical_gap_claim",
        "infinite_volume_reconstruction_claim",
        "all_scale_refinement_claim",
        "continuum_claim",
        "yang_mills_mass_gap_claim",
        "theorem_prover_verified",
        "mathlib_verified",
    ):
        assert row[key] is False
    assert replay_su2_periodic_corner_box_lower_certificate(row["certificate"])


def test_budget_reconstructed_without_calling_production_helpers() -> None:
    a = su2_periodic_corner_box_lower()["arithmetic"]
    ell = 8192
    r = Q(1, 2**107)
    filling = 96 * ell**4
    metric = 32 * (24 * ell**3) ** 2
    potential = 2048 * filling
    f = (1 - max(metric, potential) * r) / (1 + 4 * r * r)
    cutoff = r * r / filling
    expected = 18 * (1 - f) + 20 * Q(1, 2**330) / (cutoff * ell**3)
    assert Q(a["nonlinear_density_error_upper"]) == expected
    assert Q(a["bad_localized_energy_lower"]) >= 18 * 8 * ell**3
    assert metric * r <= Q(1, 16384) and potential * r <= Q(1, 16384)
    assert Q(a["ims_error_per_box_upper"]) == 20 * Q(1, 2**330) / cutoff


@pytest.mark.parametrize("side", [3, 8, 4096, 8192, 16384])
def test_family_surface_gate_and_inconclusive_replay(side: int) -> None:
    row = su2_periodic_corner_box_lower(side)
    assert (row["status"] == "PASS") == (Q(54, side) <= Q(1, 128))
    assert row["actual_nonlinear_lower_verified_in_written_analysis"] == (row["status"] == "PASS")
    assert replay_su2_periodic_corner_box_lower_certificate(row["certificate"])


def test_requested_exponents_are_checked_by_the_budget_not_default_order() -> None:
    rows = [su2_periodic_corner_box_lower(kappa_exponent=p) for p in range(318, 340, 3)]
    errors = [Q(row["arithmetic"]["nonlinear_density_error_upper"]) for row in rows]
    assert errors == sorted(errors, reverse=True)
    assert rows[0]["status"] == "INCONCLUSIVE"
    assert rows[-1]["status"] == "PASS"
    assert any(
        row["status"] == "PASS" and row["arithmetic"]["kappa_exponent"] < 330 for row in rows
    )
    for row in rows:
        assert replay_su2_periodic_corner_box_lower_certificate(row["certificate"])
        assert not row["physical_gap_claim"]


@pytest.mark.parametrize("value", [True, 3.0, Q(3), "8192", None, 2, 2**20 + 1])
def test_invalid_box_inputs(value: object) -> None:
    call: Callable[..., object] = su2_periodic_corner_box_lower
    with pytest.raises((TypeError, ValueError)):
        call(value)


@pytest.mark.parametrize("value", [True, 330.0, Q(330), "330", 0, -3, 4, 10002])
def test_invalid_exponents(value: object) -> None:
    call: Callable[..., object] = su2_periodic_corner_box_lower
    with pytest.raises((TypeError, ValueError)):
        call(kappa_exponent=value)


@pytest.mark.parametrize("place", ["claim", "meta", "scope", "status", "arithmetic", "input"])
def test_canonical_replay_rejects_resealed_tampering(place: str) -> None:
    cert = deepcopy(su2_periodic_corner_box_lower()["certificate"])
    if place == "claim":
        cert["claim"] = "a spectral gap"
    elif place == "meta":
        cert["meta"]["no_toron_minimum_premise"] = False
    elif place == "scope":
        cert["payload"]["physical_gap_claim"] = True
    elif place == "status":
        cert["payload"]["status"] = "INCONCLUSIVE"
    elif place == "arithmetic":
        cert["payload"]["arithmetic"]["nonlinear_density_error_upper"] = "0"
    else:
        cert["payload"]["inputs"]["box_side"] = 4096
    forged = make_certificate(claim=cert["claim"], payload=cert["payload"], meta=cert["meta"])
    assert verify_certificate_digest(forged)
    assert not replay_su2_periodic_corner_box_lower_certificate(forged)


def test_replay_rejects_malformed_and_unsealed_data() -> None:
    assert not replay_su2_periodic_corner_box_lower_certificate({})
    cert = deepcopy(su2_periodic_corner_box_lower()["certificate"])
    del cert["payload"]["inputs"]["kappa_exponent"]
    assert not replay_su2_periodic_corner_box_lower_certificate(cert)
    forged = make_certificate(claim=cert["claim"], payload=cert["payload"], meta=cert["meta"])
    assert not replay_su2_periodic_corner_box_lower_certificate(forged)


def test_integer_logarithm_is_exact_at_power_boundaries() -> None:
    for n in [*range(1, 512), *(2**k + j for k in [32, 107, 328, 1024] for j in [-1, 0, 1])]:
        exponent = _ceil_log2(n)
        assert n <= 2**exponent
        assert exponent == 0 or 2 ** (exponent - 1) < n
    for invalid in [0, -1, True]:
        with pytest.raises(ValueError):
            _ceil_log2(invalid)


@pytest.mark.parametrize("tilt", [Q(0), Q(1, 16), Q(1, 4)])
def test_every_crossing_cell_replacement_is_psd_and_rank_at_most_three(tilt: Q) -> None:
    for retained in product([False, True], repeat=3):
        matrix = [
            [
                (Q(1) if i == j else -tilt / 4) - ((1 - tilt / 2) if i == j and retained[i] else 0)
                for j in range(3)
            ]
            for i in range(3)
        ]
        for count in range(1, 4):
            for indices in combinations(range(3), count):
                assert _det([[matrix[i][j] for j in indices] for i in indices]) >= 0
        assert _rank(matrix) <= 3


@pytest.mark.parametrize("sides", [(2, 2, 2), (3, 3, 3), (3, 4, 5)])
def test_comb_tree_and_retained_face_coverage(sides: Vertex) -> None:
    vertices, edges, faces = _geometry(sides)
    tree = [
        (a, b)
        for a, b in edges
        if a[0] != b[0] or (a[1] != b[1] and a[0] == 0) or (a[2] != b[2] and a[0] == a[1] == 0)
    ]
    reached = {(0, 0, 0)}
    while True:
        added = {b for a, b in tree if a in reached} | {a for a, b in tree if b in reached}
        if added <= reached:
            break
        reached |= added
    assert reached == set(vertices)
    assert len(tree) == len(vertices) - 1
    complete = {v for v in vertices if all(v[i] + 1 < sides[i] for i in range(3))}
    assert len(complete) == (sides[0] - 1) * (sides[1] - 1) * (sides[2] - 1)
    exact_faces = [face for face in faces if face[0] in complete]
    replacement_faces = [face for face in faces if face[0] not in complete]
    assert len(exact_faces) + len(replacement_faces) == len(faces)
    assert replacement_faces
    incidence = {edge: sum(edge[0] in face and edge[1] in face for face in faces) for edge in edges}
    assert max(incidence.values()) <= 4
    assert _rank([_face_row(face, edges) for face in faces]) == len(edges) - len(vertices) + 1
    if sides == (2, 2, 2):
        assert len(exact_faces) == 3
        assert _rank([_face_row(face, edges) for face in exact_faces]) == 3
        assert len(edges) - len(vertices) + 1 == 5


@pytest.mark.parametrize("ell", [3, 4, 8])
def test_partition_including_single_interval_removes_periodic_seam(ell: int) -> None:
    for n in range(ell, 6 * ell):
        quotient, remainder = divmod(n, ell)
        lengths = [ell + remainder] + [ell] * (quotient - 1)
        assert sum(lengths) == n and all(ell <= side <= 2 * ell - 1 for side in lengths)
        internal_coordinate_links = sum(side - 1 for side in lengths)
        assert internal_coordinate_links == n - quotient < n
        complete_cells = internal_coordinate_links**3
        crossing = n**3 - complete_cells
        assert Q(crossing, n**3) <= Q(3, ell)
        if quotient == 1:
            assert complete_cells == (n - 1) ** 3  # Wrapping edges are cut even in one box.


@pytest.mark.parametrize("length", range(7))
def test_six_letter_action_remainder_and_haar_density_exact_rational(length: int) -> None:
    rng = Random(982 + length)
    radius = Q(1, 32)
    for _ in range(12):
        factors = [_stereo([Q(rng.randint(-3, 3), 4096) for _ in range(3)]) for _ in range(length)]
        z2 = sum((u[i] ** 2 for u in factors for i in range(1, 4)), Q(0))
        assert z2 <= radius**2
        density = Q(1)
        actual = IDENTITY
        linear = [Q(0)] * 3
        for u in factors:
            assert sum((v * v for v in u), Q(0)) == 1 and u[0] > 0
            density /= u[0]
            sign = rng.choice([-1, 1])
            v = (u[0], sign * u[1], sign * u[2], sign * u[3])
            actual = _mul(actual, v)
            linear = [linear[i] + v[i + 1] for i in range(3)]
        assert 1 <= density <= 1 + 4 * radius**2
        action = 2 - 2 * actual[0]
        quadratic = sum((x * x for x in linear), Q(0))
        assert abs(action - quadratic) <= 192 * radius * z2
