# SPDX-License-Identifier: Apache-2.0
"""Exact original-strip derivative, partition and source-dependent tail checks."""

from collections.abc import Sequence
from copy import deepcopy
from fractions import Fraction as Q
from importlib import import_module
from math import acos, cos, pi, sin, sqrt
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import strip_kernel_tail as module
from omnibias.geometry.gauge.transfer import weak_plaquette

Quat = tuple[Q, Q, Q, Q]
ID: Quat = (Q(1), Q(0), Q(0), Q(0))
ZERO: Quat = (Q(0), Q(0), Q(0), Q(0))


def mul(a: Quat, b: Quat) -> Quat:
    return (
        a[0] * b[0] - sum((a[i] * b[i] for i in range(1, 4)), Q(0)),
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
    )


def inv(a: Quat) -> Quat:
    return a[0], -a[1], -a[2], -a[3]


def stereo(v: Sequence[Q]) -> Quat:
    s = sum((x * x for x in v), Q(0))
    return (1 - s) / (1 + s), 2 * v[0] / (1 + s), 2 * v[1] / (1 + s), 2 * v[2] / (1 + s)


def word(links: Sequence[Quat], path: Sequence[int], edge: int = -1, axis: int = 0) -> Quat:
    if edge >= 0 and all(abs(i) - 1 != edge for i in path):
        return ZERO
    generator: Quat = (Q(0), Q(axis == 0, 2), Q(axis == 1, 2), Q(axis == 2, 2))
    result = ID
    for signed in path:
        i = abs(signed) - 1
        factor = mul(links[i], generator) if i == edge else links[i]
        result = mul(result, inv(factor) if signed < 0 else factor)
    return result


def independent_graph(n: int) -> tuple[list[tuple[int, int]], list[list[int]]]:
    # Vertex IDs bottom0..n and top(n+1)..(2n+1), independent coordinate construction.
    vertices = {(x, y): x + y * (n + 1) for y in range(2) for x in range(n + 1)}
    edges = [(vertices[x, y], vertices[x + 1, y]) for y in range(2) for x in range(n)]
    edges += [(vertices[x, 0], vertices[x, 1]) for x in range(n + 1)]
    lookup = {e: i + 1 for i, e in enumerate(edges)}
    faces = []
    for x in range(n):
        loop = [(x, 0), (x + 1, 0), (x + 1, 1), (x, 1), (x, 0)]
        face = []
        for v, w in zip(loop, loop[1:], strict=False):
            e = (vertices[v], vertices[w])
            face.append(lookup[e] if e in lookup else -lookup[e[1], e[0]])
        faces.append(face)
    return edges, faces


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return module.su2_strip_normalized_kernel_tail(Q(1, 2**48), 4, 500000000, target=Q(1, 1024))


def test_actual_source_nonempty_target_and_scope(result: dict[str, Any]) -> None:
    assert result["status"] == result["tail_target_status"] == "PASS"
    assert not result["tail_region_has_zero_haar_measure"]
    assert result["arithmetic"]["ground_energy_upper"] == "12"
    assert result["arithmetic"]["squared_hs_tail_exponent"] == "-3468160/33"
    assert result["arithmetic"]["dyadic_upper_negative_integer_exponent"] == 105095
    assert result["arithmetic"]["target_required_dyadic_exponent"] == 10
    assert result["retained_cycles"] == [1, 2] and result["eliminated_cycles"] == [3, 4]
    assert module.replay_su2_strip_kernel_tail_certificate(result["certificate"])
    assert weak_plaquette.replay_su2_weak_plaquette_gap_certificate(
        result["plaquette_trial_source_certificate"]
    )
    assert result["certificate"]["payload"] == {
        k: v for k, v in result.items() if k != "certificate"
    }
    for flag in (
        "actual_normalized_kernel_tail_verified_in_written_analysis",
        "actual_product_trial_energy_verified_in_written_analysis",
        "all_reduced_modes_included",
    ):
        assert result[flag] is True
    for flag in (
        "source_gap_used_as_premise",
        "full_reduced_metric_bounded_by_path_matrix_claim",
        "gaussian_kernel_comparison_verified",
        "maximal_correlation_upper_verified",
        "explicit_correlation_coupling_threshold_verified",
        "ambient_exterior_uniformity_verified",
        "uniform_in_volume_claim",
        "continuum_claim",
        "yang_mills_claim",
        "mathlib_verified",
        "theorem_prover_verified",
    ):
        assert result[flag] is False


def test_partition_canonicalization_and_target_separation() -> None:
    a = module.su2_strip_normalized_kernel_tail(
        Q(1, 64), 5, 1, retained_cycles=[4, 1], target=Q(1, 1024)
    )
    b = module.su2_strip_normalized_kernel_tail(
        Q(1, 64), 5, 1, retained_cycles=(1, 4), target=Q(1, 1024)
    )
    assert a["certificate"] == b["certificate"]
    assert a["status"] == "PASS" and a["tail_target_status"] == "INCONCLUSIVE"
    assert (
        module.su2_strip_normalized_kernel_tail(Q(1, 64), 2, 1)["tail_target_status"]
        == "NOT_REQUESTED"
    )
    outside = module.su2_strip_normalized_kernel_tail(Q(1, 32), 2, 1000000, target=1)
    assert outside["status"] == outside["tail_target_status"] == "INCONCLUSIVE"
    assert outside["squared_hs_tail_upper"] is None


def test_compact_empty_equality_and_extreme_rational_representation() -> None:
    for radius in (2048, 2049):
        row = module.su2_strip_normalized_kernel_tail(Q(1, 64), 4, radius, target=Q(1, 2**4096))
        assert row["effective_squared_hs_tail_upper"] == {"exact_rational": "0"}
        assert row["tail_target_status"] == "PASS"
    row = module.su2_strip_normalized_kernel_tail(Q(1, 2**4096), 3, 2**4096, target=Q(1, 2**4096))
    assert not row["tail_region_has_zero_haar_measure"]
    assert row["tail_target_status"] == "PASS"
    assert row["arithmetic"]["dyadic_upper_negative_integer_exponent"].bit_length() > 4000
    assert module.replay_su2_strip_kernel_tail_certificate(row["certificate"])


@pytest.mark.parametrize("field", ("kappa", "radius", "target"))
@pytest.mark.parametrize("bad", (True, False, 0.25, "1/64", 0, -1))
def test_exact_positive_input_guards(field: str, bad: Any) -> None:
    args: dict[str, Any] = {"kappa": Q(1, 64), "n_plaquettes": 2, "radius": 1, "target": 1}
    args[field] = bad
    with pytest.raises((TypeError, ValueError)):
        module.su2_strip_normalized_kernel_tail(**args)


@pytest.mark.parametrize("bad", (None, True, False, 0, 1, -2, 2.0, Q(2), "2"))
def test_size_guards(bad: Any) -> None:
    with pytest.raises(ValueError):
        module.su2_strip_normalized_kernel_tail(Q(1, 64), bad, 1)


@pytest.mark.parametrize("bad", ([], [1, 2], [1, 1], [0], [3], [True], [1.0], "1", {1}))
def test_partition_guards(bad: Any) -> None:
    with pytest.raises(ValueError):
        module.su2_strip_normalized_kernel_tail(Q(1, 64), 2, 1, retained_cycles=bad)


@pytest.mark.parametrize("bad", (None, [], (), "certificate", 1, True))
def test_replay_malformed_guard(bad: Any) -> None:
    assert not module.replay_su2_strip_kernel_tail_certificate(bad)


def test_rehashed_arithmetic_geometry_scope_and_source_attacks(result: dict[str, Any]) -> None:
    for kind in ("coefficient", "tree", "word", "partition", "formal", "source"):
        cert = deepcopy(result["certificate"])
        p = cert["payload"]
        if kind == "coefficient":
            p["arithmetic"]["normalized_kernel_decay_coefficient"] = "1"
        elif kind == "tree":
            p["geometry"]["tree_link_ids"][0] = 1
        elif kind == "word":
            p["geometry"]["plaquette_words"][0][0] *= -1
        elif kind == "partition":
            p["retained_cycles"] = [1]
        elif kind == "formal":
            p["analytic_proof_formally_verified"] = True
        else:
            attached = p["plaquette_trial_source_certificate"]
            attached["payload"]["witness"]["arithmetic"]["ground_energy_upper"] = "0"
            p["plaquette_trial_source_certificate"] = seal_certificate(attached)
        assert not module.replay_su2_strip_kernel_tail_certificate(seal_certificate(cert))


def test_source_failure_and_sealed_consumption(monkeypatch: pytest.MonkeyPatch) -> None:
    original = weak_plaquette.su2_weak_plaquette_gap

    def detached(kappa: int | Q, **kwargs: Any) -> dict[str, Any]:
        row = original(kappa, **kwargs)
        row["unrelated_detached_gap"] = "1000000"
        return row

    monkeypatch.setattr(weak_plaquette, "su2_weak_plaquette_gap", detached)
    assert (
        module.su2_strip_normalized_kernel_tail(Q(1, 64), 3, 1)["arithmetic"]["ground_energy_upper"]
        == "9"
    )
    monkeypatch.setattr(
        weak_plaquette, "replay_su2_weak_plaquette_gap_certificate", lambda _: False
    )
    bad = module.su2_strip_normalized_kernel_tail(Q(1, 64), 3, 1)
    assert bad["status"] == "INCONCLUSIVE"
    assert not bad["actual_normalized_kernel_tail_verified_in_written_analysis"]


@pytest.mark.parametrize("n", (2, 3, 5, 8))
def test_independent_original_incidence_tree_and_harmonic_matrix(n: int) -> None:
    edges, faces = independent_graph(n)
    row = module.su2_strip_normalized_kernel_tail(Q(1, 64), n, 1)
    assert row["geometry"]["oriented_edges"] == [list(e) for e in edges]
    assert row["geometry"]["plaquette_words"] == faces
    boundary = [
        [Q((e + 1 in face) - (-(e + 1) in face)) for e in range(len(edges))] for face in faces
    ]
    for i in range(n):
        for j in range(n):
            gram = sum((a * b for a, b in zip(boundary[i], boundary[j], strict=True)), Q(0))
            assert gram == (4 if i == j else -1 if abs(i - j) == 1 else 0)
    chords = [stereo((Q(i + 1, n + 2), Q(1, 3), Q(-1, 5))) for i in range(n)]
    links = chords + [ID] * (2 * n + 1)
    assert [word(links, face) for face in faces] == chords
    # Exact spanning-tree connectivity; n independent chords, no prefix product.
    reached = {n + 1}
    while True:
        new = (
            reached
            | {v for u, v in edges[n:] if u in reached}
            | {u for u, v in edges[n:] if v in reached}
        )
        if new == reached:
            break
        reached = new
    assert len(reached) == 2 * n + 2 and len(edges[n:]) == 2 * n + 1


@pytest.mark.parametrize("n", (2, 3, 5))
def test_exact_all_original_generator_rows_grid_and_random(n: int) -> None:
    edges, faces = independent_graph(n)
    rng = Random(440 + n)
    packs = [
        [stereo((Q((i + j) % 3 - 1, 2), Q(j, 3), Q(-i % 3, 4))) for i in range(len(edges))]
        for j in range(3)
    ]
    packs += [
        [stereo([Q(rng.randrange(-3, 4), 5) for _ in range(3)]) for _ in edges] for _ in range(5)
    ]
    for links in packs:
        traces = [2 * word(links, face)[0] for face in faces]
        action = [2 - t for t in traces]
        rows = [
            [[-2 * word(links, face, e, a)[0] for a in range(3)] for e in range(len(edges))]
            for face in faces
        ]
        for i in range(n):
            squared = sum((v * v for row in rows[i] for v in row), Q(0))
            assert squared == 4 * (1 - traces[i] ** 2 / 4)
            for j in range(i + 1, n):
                cross = sum(
                    (rows[i][e][a] * rows[j][e][a] for e in range(len(edges)) for a in range(3)),
                    Q(0),
                )
                if j > i + 1:
                    assert cross == 0
                else:
                    # Shared-edge Cauchy has no four-edge multiplicity.
                    assert cross * cross <= (1 - traces[i] ** 2 / 4) * (1 - traces[j] ** 2 / 4)
        total = sum(
            (
                sum((rows[i][e][a] for i in range(n)), Q(0)) ** 2
                for e in range(len(edges))
                for a in range(3)
            ),
            Q(0),
        )
        assert total <= 6 * sum(action, Q(0))
        assert 3 * sum(traces, Q(0)) == 6 * n - 3 * sum(action, Q(0))
        # Diagnostic original-link phase derivatives; no reduced-metric guess.
        factors = [1 / sqrt(float((1 + trace / 2) / 2)) for trace in traces]
        phase_gamma = sum(
            sum(factors[i] * float(rows[i][e][a]) for i in range(n)) ** 2
            for e in range(len(edges))
            for a in range(3)
        )
        assert 2 * float(sum(action, Q(0))) <= phase_gamma + 1e-10
        assert phase_gamma <= 6 * float(sum(action, Q(0))) + 1e-10


def test_numeric_diagnostics_phase_bounds_radial_path_and_haar_integrals() -> None:
    # Floating checks are diagnostics; the source proof uses exact identities/inequalities.
    mp = import_module("mpmath")
    rng = Random(719)
    angles = [pi * i / 400 for i in range(401)] + [rng.random() * pi for _ in range(200)]
    eps = Q(1, 64)
    for theta in angles:
        f = 8 * (1 - cos(theta / 2))
        smooth = 8 * (sqrt(1 + float(eps) ** 2) - sqrt(cos(theta / 2) ** 2 + float(eps) ** 2))
        action = 2 - 2 * cos(theta)
        assert 32 * f / 33 <= smooth + 2e-14
        assert smooth <= f + 2e-14 and f <= 2 * action + 2e-14
        assert 2 * theta <= pi * sqrt(max(0, action)) + 2e-13
    for k in [Q(i, 6400) for i in range(1, 101, 5)] + [
        Q(rng.randrange(1, 101), 6400) for _ in range(10)
    ]:
        with mp.workdps(45):
            coupling = mp.mpf(k.numerator) / k.denominator

            def integral(c: Any, coupling: Any = coupling) -> Any:
                return mp.quad(
                    lambda theta: (
                        2
                        / mp.pi
                        * mp.sin(theta) ** 2
                        * mp.exp(-c * 8 * (1 - mp.cos(theta / 2)) / coupling)
                    ),
                    [0, mp.sqrt(coupling), mp.pi / 2, mp.pi],
                )

            assert integral(3) >= coupling ** mp.mpf("1.5") / 200
            assert integral(mp.mpf(17) / 330) <= 466 * coupling ** mp.mpf("1.5")


def test_symbolic_cutoff_and_all_n_arithmetic() -> None:
    sp = import_module("sympy")
    s, n, k, b, gamma = sp.symbols("s n k b gamma", positive=True)
    zeta = 1 - s / (b * k)
    eta = zeta**2
    delta = sp.diff(eta, s) * (6 * n - 3 * s) + sp.diff(eta, s, 2) * gamma
    grad = sp.diff(eta, s) ** 2 * gamma
    assert (
        sp.simplify(
            -delta
            + 2 * grad / eta
            - 2 * zeta * (6 * n - 3 * s) / (b * k)
            - 6 * gamma / (b * k) ** 2
        )
        == 0
    )
    assert Q(4, 5) * 4 * Q(32, 33) - 3 == Q(17, 165)
    assert Q(121, 98) * Q(330, 17) ** 2 < 466
    assert Q(8, 3) ** 12 > 93200
    # The growth is genuine in this bound, not a volume-uniform claim.
    previous = Q(0)
    for size in (2, 3, 10, 100):
        row = module.su2_strip_normalized_kernel_tail(Q(1, 64), size, 1)
        exponent = Q(row["arithmetic"]["squared_hs_tail_exponent"])
        assert exponent > previous
        previous = exponent


@pytest.mark.parametrize("n", (2, 3, 5))
def test_exact_tree_gauge_and_product_trial_cross_cancellation(n: int) -> None:
    edges, faces = independent_graph(n)
    rng = Random(781 + n)
    links = [stereo([Q(rng.randrange(-3, 4), 4) for _ in range(3)]) for _ in edges]
    gauges = [ID] * (2 * n + 2)
    for i in range(n):
        gauges[n + i + 2] = mul(gauges[n + i + 1], links[n + i])
    for i in range(n + 1):
        gauges[i] = mul(gauges[n + i + 1], inv(links[2 * n + i]))
    fixed = [
        mul(mul(gauges[u], link), inv(gauges[v])) for (u, v), link in zip(edges, links, strict=True)
    ]
    assert fixed[n:] == [ID] * (2 * n + 1)
    for i in range(n):
        assert word(fixed, faces[i]) == fixed[i]
        assert fixed[i][0] == word(links, faces[i])[0]
    # Averaging each central score over U and U^{-1} already cancels its
    # cross expectation, before integration over the class angles. A central
    # trial's value and radial derivative factor are equal on the pair.
    base = [stereo((Q(i + 1, n + 1), Q(1, 3), Q(1, 5))) for i in range(n)]
    base += [ID] * (2 * n + 1)
    for i in range(n - 1):
        crossings = []
        for flip_i in (False, True):
            for flip_j in (False, True):
                pack = base.copy()
                pack[i] = inv(pack[i]) if flip_i else pack[i]
                pack[i + 1] = inv(pack[i + 1]) if flip_j else pack[i + 1]
                crossings.append(
                    sum(
                        (
                            4 * word(pack, faces[i], e, a)[0] * word(pack, faces[i + 1], e, a)[0]
                            for e in range(len(edges))
                            for a in range(3)
                        ),
                        Q(0),
                    )
                )
        assert any(crossings)
        assert sum(crossings, Q(0)) == 0


def test_two_cycle_orientation_matches_theta_up_to_inversion() -> None:
    # diag(1,-1) conjugates [[4,-1],[-1,4]] to the theta [[4,1],[1,4]].
    matrix = ((4, -1), (-1, 4))
    signs = (1, -1)
    assert [[signs[i] * matrix[i][j] * signs[j] for j in range(2)] for i in range(2)] == [
        [4, 1],
        [1, 4],
    ]
    coarse = module.su2_strip_normalized_kernel_tail(Q(1, 2**48), 2, 1)
    assert Q(coarse["arithmetic"]["squared_hs_tail_exponent"]) > 81640
