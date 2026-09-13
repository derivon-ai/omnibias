# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent original-link and local-remainder controls for the corner proof."""

import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
from random import Random

import numpy as np
import pytest
from omnibias.core.proof.certificate import make_certificate
from omnibias.geometry.gauge.stochastic.lattice import Quaternion
from omnibias.geometry.gauge.stochastic.lattice import quaternion_inverse as inv
from omnibias.geometry.gauge.stochastic.lattice import quaternion_product as mul
from omnibias.geometry.gauge.transfer.corner_vacuum import (
    corner_geometry,
    corner_reduced_metric,
    replay_corner_vacuum_certificate,
    search_corner_vacuum_interval,
    su2_corner_vacuum_bound,
)

IDENTITY = (Q(1), Q(0), Q(0), Q(0))


def _stereo(v: Sequence[Q]) -> tuple[Q, ...]:
    n = sum((x * x for x in v), Q(0))
    return ((1 - n) / (1 + n), *(2 * x / (1 + n) for x in v))


def _product(values: Iterable[Sequence[int | Q]]) -> Quaternion:
    result = IDENTITY
    for v in values:
        result = mul(result, v)
    return result


def _original_symbol(holonomies: Sequence[Sequence[Q]]) -> list[list[Q]]:
    geometry = corner_geometry()
    field: list[Sequence[Q]] = [IDENTITY] * 9
    for edge, value in zip(geometry["chord_edges"], holonomies, strict=True):
        field[edge - 1] = value
    rows = []
    for edge in range(9):
        for axis in range(3):
            tangent = [Q(0), *[Q(1, 2) if j == axis else Q(0) for j in range(3)]]
            row = []
            for face in geometry["faces"]:
                factors = [field[e - 1] if e > 0 else inv(field[-e - 1]) for e in face]
                derivative = [Q(0)] * 3
                for j, token in enumerate(face):
                    if abs(token) != edge + 1:
                        continue
                    inserted: Sequence[Q]
                    if token > 0:
                        inserted = mul(tangent, field[edge])
                    else:
                        inserted = tuple(-v for v in mul(inv(field[edge]), tangent))
                    derivative = list(_product([*factors[:j], inserted, *factors[j + 1 :]])[1:])
                row.extend(derivative)
            rows.append(row)
    return [[sum((row[i] * row[j] for row in rows), Q(0)) for j in range(9)] for i in range(9)]


@pytest.mark.parametrize("shift", range(8))
def test_reduced_metric_equals_all_original_nine_link_derivatives(shift: int) -> None:
    rng = Random(178 + shift)
    holonomies = [_stereo([Q(rng.randint(-3, 3), 5) for _ in range(3)]) for _ in range(3)]
    row = corner_reduced_metric(holonomies)
    actual = [[Q(v) for v in r] for r in row["arithmetic"]["metric"]]
    assert actual == _original_symbol(holonomies)
    assert Q(row["arithmetic"]["positive_potential_slack"]) >= 0
    assert replay_corner_vacuum_certificate(row["certificate"])


def test_flat_symbol_and_harmonic_reference_spectra() -> None:
    g = np.array(
        [
            [float(Q(v)) for v in row]
            for row in corner_reduced_metric([IDENTITY] * 3)["arithmetic"]["metric"]
        ]
    )
    expected = np.kron(5 * np.eye(3) - np.ones((3, 3)), np.eye(3)) / 4
    assert np.array_equal(g, expected)
    assert np.allclose(np.linalg.eigvalsh(g), [0.5] * 3 + [1.25] * 6)
    matrix = 5 * np.eye(3) - np.ones((3, 3))
    tilted_potential = 2.5 * np.eye(3) - 0.5 * np.ones((3, 3))
    frequencies = np.sqrt(np.linalg.eigvalsh(matrix @ tilted_potential / 2))
    assert np.allclose(frequencies, [1, 2.5, 2.5])
    assert abs(1.5 * sum(frequencies) - 9) < 1e-12


def test_local_metric_density_and_potential_bounds_grid_and_random() -> None:
    rng = Random(72213)
    samples = [[Q(k, 2**14), Q(0), Q(0)] * 3 for k in range(-8, 9)]
    samples += [[Q(rng.randint(-8, 8), 2**14) for _ in range(9)] for _ in range(32)]
    g0 = np.kron(5 * np.eye(3) - np.ones((3, 3)), np.eye(3)) / 4
    radius = 1 / 64
    for sample in samples:
        us = [_stereo(sample[3 * i : 3 * i + 3]) for i in range(3)]
        a = corner_reduced_metric(us)["arithmetic"]
        z = np.array([float(v) for u in us for v in u[1:]])
        z2 = float(z @ z)
        s = float(Q(a["old_action_sum"]))
        assert s <= radius**2
        assert z2 - 1e-15 <= s <= (1 + radius**2) * z2 + 1e-15
        density = np.prod([1 / float(u[0]) for u in us])
        assert 1 <= density <= 1 + 4 * radius**2
        g = np.array([[float(Q(v)) for v in row] for row in a["metric"]])
        assert np.linalg.eigvalsh(g - (1 - 64 * radius) * g0).min() >= -1e-14
        assert np.linalg.eigvalsh((1 + 64 * radius) * g0 - g).min() >= -1e-14
        boundary_quadratic = float(z.reshape(3, 3).sum(axis=0) @ z.reshape(3, 3).sum(axis=0))
        assert abs(float(Q(a["boundary_action"])) - boundary_quadratic) <= 32 * radius * z2 + 1e-15
        v0 = 2.5 * z2 - 0.5 * boundary_quadratic
        assert float(Q(a["tilted_potential"])) >= (1 - 18 * radius) * v0 - 1e-15


def test_first_passing_interval_and_genuine_energy_separation() -> None:
    result = search_corner_vacuum_interval()
    assert result["status"] == "PASS" and len(result["attempts"]) == 7
    row = result["accepted"]
    a = row["arithmetic"]
    assert row["inputs"]["tau_upper"] == "1/256"
    assert a["kappa_upper"] == str(Q(1, 2**40))
    assert Q(a["upper_remainder"]) <= Q(1, 16)
    assert Q(a["lower_remainder"]) <= Q(1, 16)
    assert Q(a["energy_separation_lower"]) > Q(19, 420)
    assert Q(a["observable_difference_over_kappa_upper"]) < -Q(19, 210)
    assert all(replay_corner_vacuum_certificate(x["certificate"]) for x in result["attempts"])
    assert not row["physical_gap_claim"] and not row["continuum_claim"]
    assert not a["exterior_coupled_vacuum_verified"] and not a["analytic_proof_formally_verified"]
    assert not search_corner_vacuum_interval(7)["accepted"]


def test_budget_monotonicity_grid_and_random_exact_points() -> None:
    rng = Random(61)
    taus = [Q(j, 256 * 32) for j in range(1, 33)]
    taus += [Q(rng.randint(1, 1000), 256000) for _ in range(32)]
    rows = [su2_corner_vacuum_bound(t)["arithmetic"] for t in sorted(taus)]
    upper = [Q(a["actual_ground_energy_upper"]) for a in rows]
    lower = [Q(a["actual_tilted_ground_energy_lower"]) for a in rows]
    assert upper == sorted(upper) and lower == sorted(lower, reverse=True)


@pytest.mark.parametrize("which", ["claim", "meta", "arithmetic"])
def test_resealed_forgery_rejected(which: str) -> None:
    cert = deepcopy(su2_corner_vacuum_bound()["certificate"])
    if which == "claim":
        cert["claim"] = "a mass gap"
    elif which == "meta":
        cert["meta"]["transcend_backend"] = "invented"
    else:
        cert["payload"]["arithmetic"]["exterior_coupled_vacuum_verified"] = True
    forged = make_certificate(claim=cert["claim"], payload=cert["payload"], meta=cert["meta"])
    assert not replay_corner_vacuum_certificate(forged)


@pytest.mark.parametrize("value", [True, 0.1, Q(0), Q(-1), Q(1, 2)])
def test_invalid_exact_domain_refused(value: int | float | Q) -> None:
    # This dynamic input boundary deliberately includes a statically invalid float.
    reject_input: Callable[..., object] = su2_corner_vacuum_bound
    pytest.raises((TypeError, ValueError), reject_input, value)


def test_failed_budget_cannot_earn_actual_energy_or_vacuum_gate() -> None:
    row = su2_corner_vacuum_bound(Q(1, 128))
    assert row["status"] == "INCONCLUSIVE"
    a = row["arithmetic"]
    assert a["actual_ground_energy_upper"] is None
    assert a["actual_tilted_ground_energy_lower"] is None
    assert not a["actual_corner_vacuum_decorrelation_verified"]


@pytest.mark.parametrize(
    "first,second",
    [
        ("omnibias.geometry.gauge.transfer", "omnibias.geometry.gauge.stochastic.lattice"),
        ("omnibias.geometry.gauge.stochastic.lattice", "omnibias.geometry.gauge.transfer"),
    ],
)
def test_fresh_process_import_order_does_not_reenter_partial_lattice_module(
    first: str,
    second: str,
    tmp_path: Path,
) -> None:
    # A fresh interpreter is essential: the test collector has already loaded
    # both packages, which would hide the original transfer/lattice cycle.
    script = f"""
import importlib
importlib.import_module({first!r})
importlib.import_module({second!r})
lattice = importlib.import_module("omnibias.geometry.gauge.stochastic.lattice")
corner = importlib.import_module("omnibias.geometry.gauge.transfer.corner_vacuum")
assert lattice._unit((1, 0, 0, 0)) == (1, 0, 0, 0)
row = corner.corner_reduced_metric([(1, 0, 0, 0)] * 3)
assert row["arithmetic"]["metric"][0][0] == "1"
assert corner.replay_corner_vacuum_certificate(row["certificate"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("seed", [0, 11, 4096])
def test_reduced_metric_transforms_as_tensor_under_exact_root_conjugation(seed: int) -> None:
    rng = Random(seed)
    holonomies = [_stereo([Q(rng.randint(-4, 4), 5) for _ in range(3)]) for _ in range(3)]
    gauge = _stereo([Q(rng.randint(-4, 4), 5) for _ in range(3)])
    conjugated = [mul(mul(gauge, u), inv(gauge)) for u in holonomies]
    before = corner_reduced_metric(holonomies)["arithmetic"]
    after = corner_reduced_metric(conjugated)["arithmetic"]
    for key in (
        "old_action_sum",
        "boundary_action",
        "tilted_potential",
        "positive_potential_slack",
    ):
        assert before[key] == after[key]

    # Derive the spatial adjoint rotation from exact quaternion conjugation,
    # independently of the production metric's left/right derivative rows.
    columns = [
        mul(mul(gauge, [Q(0), *[Q(int(j == axis)) for j in range(3)]]), inv(gauge))[1:]
        for axis in range(3)
    ]
    rotation = [[columns[j][i] for j in range(3)] for i in range(3)]
    assert [
        [sum(rotation[k][i] * rotation[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ] == [[Q(int(i == j)) for j in range(3)] for i in range(3)]
    old_metric = [[Q(value) for value in row] for row in before["metric"]]
    expected = [
        [
            sum(
                (
                    rotation[i % 3][a]
                    * old_metric[3 * (i // 3) + a][3 * (j // 3) + b]
                    * rotation[j % 3][b]
                    for a in range(3)
                    for b in range(3)
                ),
                Q(0),
            )
            for j in range(9)
        ]
        for i in range(9)
    ]
    assert [[Q(value) for value in row] for row in after["metric"]] == expected
