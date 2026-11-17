# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact margins, independent dense oracles, scope refusal and canonical replay."""

from __future__ import annotations

import ast
import re
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.core.verified import locality


def test_rectangular_weight_convention_and_exact_caps() -> None:
    row = locality.weighted_schur_bound([[1, -2, 3], [0, 4, -1]], [2, 3], [1, 2, 4])
    assert row["bounds"] == {
        "row_sums": ["11/2", "27/4"],
        "column_sums": ["2", "8", "9/4"],
        "row_upper": "27/4",
        "column_upper": "8",
        "l2_norm_squared_upper": "54",
    }
    transformed = (
        np.diag([2, 3]) @ np.array([[1, -2, 3], [0, 4, -1]]) @ np.diag([1, 0.5, 0.25])
    )
    assert np.linalg.norm(transformed, 2) ** 2 <= 54
    assert locality.replay_locality_certificate(row["certificate"])


@pytest.mark.parametrize("shape", ((1, 1), (1, 4), (4, 1), (3, 5), (5, 3)))
def test_independent_signed_dense_schur_oracle(shape: tuple[int, int]) -> None:
    m, n = shape
    matrix = [
        [Q((7 * i + 3 * j) % 11 - 5, 1 + (i + j) % 3) for j in range(n)]
        for i in range(m)
    ]
    rw = [Q(i + 1, 3) for i in range(m)]
    cw = [Q(j + 2, 5) for j in range(n)]
    row = locality.weighted_schur_bound(matrix, rw, cw)
    exact = [[abs(matrix[i][j] * rw[i] / cw[j]) for j in range(n)] for i in range(m)]
    row_max = max(map(sum, exact))
    col_max = max(sum(exact[i][j] for i in range(m)) for j in range(n))
    assert Q(row["bounds"]["l2_norm_squared_upper"]) == row_max * col_max
    transformed = np.array(
        [[float(matrix[i][j] * rw[i] / cw[j]) for j in range(n)] for i in range(m)]
    )
    assert np.linalg.norm(transformed, 2) ** 2 <= float(row_max * col_max) + 1e-12


def test_zero_schur_and_singleton_generator() -> None:
    assert (
        locality.weighted_schur_bound([[0, 0]])["bounds"]["l2_norm_squared_upper"]
        == "0"
    )
    row = locality.weighted_symmetric_generator_bounds([[Q(3, 2)]], [[0]], Q(4, 3))
    assert row["bounds"]["weighted_commutator_upper"] == "0"
    assert row["bounds"]["two_sided_hessian_resolvent_upper"] == "1/3"
    assert row["passed"]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: locality.weighted_schur_bound([[1, 2]]),
        lambda: locality.weighted_symmetric_generator_bounds(
            [[3, -1], [-1, 3]], [[0, 1], [1, 0]], Q(3, 2)
        ),
        lambda: locality.quadratic_ball_budget(Q(1, 16), 2, Q(1, 4), Q(1, 4)),
    ),
)
def test_report_and_sealed_payload_do_not_alias(factory: object) -> None:
    report = factory()
    original = deepcopy(report["certificate"])
    report["bounds"][next(iter(report["bounds"]))] = "outer mutation"
    assert report["certificate"] == original
    assert locality.replay_locality_certificate(report["certificate"])
    report["certificate"]["payload"]["inputs"][next(iter(report["inputs"]))] = (
        "inner mutation"
    )
    assert "inner mutation" not in report["inputs"].values()


@pytest.mark.parametrize("n", (2, 3, 5))
@pytest.mark.parametrize("weight", (Q(5, 4), Q(3, 2)))
def test_dense_semigroup_and_sylvester_oracles(n: int, weight: Q) -> None:
    matrix = [
        [4 if i == j else -1 if abs(i - j) == 1 else 0 for j in range(n)]
        for i in range(n)
    ]
    distances = [[abs(i - j) for j in range(n)] for i in range(n)]
    row = locality.weighted_symmetric_generator_bounds(matrix, distances, weight)
    assert row["passed"]
    bounds = row["bounds"]
    lam = Q(bounds["gap_floor"])
    eta = Q(bounds["weighted_commutator_upper"])
    assert lam == (3 if n == 2 else 2)
    assert eta == (weight - 1) * (1 if n == 2 else 2)
    assert Q(bounds["two_sided_hessian_decay_margin"]) == 2 * (lam - eta)
    dense = np.array(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    # Heights are distances from a chosen source, hence 1-Lipschitz.
    heights = np.array(distances[n // 2])
    d = np.diag(float(weight) ** heights)
    inverse_d = np.diag(float(weight) ** -heights)
    for t in (0.0, 0.01, 0.5, 2.0):
        semigroup = (eigenvectors * np.exp(-t * eigenvalues)) @ eigenvectors.T
        actual = np.linalg.norm(d @ semigroup @ inverse_d, 2)
        assert actual <= np.exp(-float(lam - eta) * t) + 2e-13
    sylvester = np.kron(dense, np.eye(n)) + np.kron(np.eye(n), dense)
    for transform, key in (
        (np.kron(d, d), "two_sided_hessian_resolvent_upper"),
        (np.kron(d, np.eye(n)), "one_sided_hessian_resolvent_upper"),
    ):
        actual = np.linalg.norm(
            transform @ np.linalg.inv(sylvester) @ np.linalg.inv(transform), 2
        )
        assert actual <= float(Q(bounds[key])) + 2e-13
    assert locality.replay_locality_certificate(row["certificate"])


def test_weighted_margin_failure_retains_only_earned_bounds() -> None:
    row = locality.weighted_symmetric_generator_bounds(
        [[2, -1], [-1, 2]], [[0, 1], [1, 0]], 2
    )
    assert row["status"] == "INCONCLUSIVE" and not row["passed"]
    assert row["finite_coercivity_verified"]
    assert not row["finite_weighted_decay_verified"]
    assert row["bounds"]["weighted_resolvent_upper"] is None
    assert row["bounds"]["two_sided_hessian_resolvent_upper"] is None
    assert row["bounds"]["one_sided_hessian_resolvent_upper"] == "1"
    assert locality.replay_locality_certificate(row["certificate"])


@pytest.mark.parametrize(
    "args,passed,mapped,contraction",
    (
        ((Q(1, 16), 2, Q(1, 4), Q(1, 4)), True, Q(5, 32), Q(1, 4)),
        ((Q(1, 2), 1, 0, Q(1, 2)), True, Q(1, 2), 0),
        ((0, 1, 1, Q(1, 2)), False, Q(1, 4), 1),
        ((2, 1, 0, 1), False, 2, 0),
        ((100, 0, 100, 1), True, 0, 0),
    ),
)
def test_ball_boundaries_and_external_hypotheses(
    args: tuple, passed: bool, mapped: Q, contraction: Q
) -> None:
    row = locality.quadratic_ball_budget(*args)
    assert row["passed"] is passed
    assert Q(row["bounds"]["mapped_radius_upper"]) == mapped
    assert Q(row["bounds"]["contraction_upper"]) == contraction
    assert set(row["external_premises"].values()) == {"UNVERIFIED"}
    assert not row["nonlinear_existence_verified"]
    assert not row["nonlinear_pde_closure_verified"]
    assert locality.replay_locality_certificate(row["certificate"])


@pytest.mark.parametrize(
    "matrix",
    ([], [[]], [[1], [1, 2]], [[1.0]], [[True]], [[float("nan")]], [[float("inf")]]),
)
def test_bad_schur_matrices_refused(matrix: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        locality.weighted_schur_bound(matrix)


@pytest.mark.parametrize(
    "rw,cw",
    (([0], [1]), ([-1], [1]), ([1, 2], [1]), ([1], []), ([True], [1]), ([1], [0.5])),
)
def test_bad_weights_refused(rw: list, cw: list) -> None:
    with pytest.raises((TypeError, ValueError)):
        locality.weighted_schur_bound([[1]], rw, cw)


@pytest.mark.parametrize(
    "distances",
    (
        [[1, 1], [1, 0]],
        [[0, 0], [0, 0]],
        [[0, 1], [2, 0]],
        [[0, -1], [-1, 0]],
        [[0, True], [True, 0]],
        [[0, Q(1)], [Q(1), 0]],
        [[0]],
    ),
)
def test_bad_metrics_refused(distances: list) -> None:
    with pytest.raises((TypeError, ValueError)):
        locality.weighted_symmetric_generator_bounds(
            [[3, -1], [-1, 3]], distances, Q(3, 2)
        )


def test_triangle_asymmetry_and_coercivity_refused() -> None:
    with pytest.raises(ValueError, match="triangle"):
        locality.weighted_symmetric_generator_bounds(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 1, 3], [1, 0, 1], [3, 1, 0]], 2
        )
    with pytest.raises(ValueError, match="symmetric"):
        locality.weighted_symmetric_generator_bounds(
            [[3, 1], [0, 3]], [[0, 1], [1, 0]], 2
        )
    with pytest.raises(ValueError, match="coercivity"):
        locality.weighted_symmetric_generator_bounds(
            [[1, -1], [-1, 1]], [[0, 1], [1, 0]], 2
        )
    with pytest.raises(ValueError, match="square"):
        locality.weighted_symmetric_generator_bounds([[3, 0]], [[0]], 2)


@pytest.mark.parametrize("weight", (True, 1.5, 1, 0, -1))
def test_invalid_generator_weight(weight: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        locality.weighted_symmetric_generator_bounds([[3]], [[0]], weight)


@pytest.mark.parametrize(
    "args",
    (
        (-1, 1, 1, 1),
        (1, -1, 1, 1),
        (1, 1, -1, 1),
        (1, 1, 1, 0),
        (True, 1, 1, 1),
        (0.1, 1, 1, 1),
    ),
)
def test_invalid_ball_inputs(args: tuple) -> None:
    with pytest.raises((TypeError, ValueError)):
        locality.quadratic_ball_budget(*args)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: locality.weighted_schur_bound([[1, 2]]),
        lambda: locality.weighted_symmetric_generator_bounds(
            [[3, -1], [-1, 3]], [[0, 1], [1, 0]], Q(3, 2)
        ),
        lambda: locality.quadratic_ball_budget(Q(1, 16), 2, Q(1, 4), Q(1, 4)),
    ),
)
@pytest.mark.parametrize("change", ("bound", "claim", "scope", "extra", "noncanonical"))
def test_resealed_changes_rejected(factory: object, change: str) -> None:
    cert = deepcopy(factory()["certificate"])
    if change == "bound":
        cert["payload"]["bounds"][next(iter(cert["payload"]["bounds"]))] = "999"
    elif change == "claim":
        cert["claim"] = "Unproved stronger claim"
    elif change == "scope":
        cert["payload"]["mixed_hessian_pde_locality_verified"] = True
    elif change == "extra":
        cert["payload"]["unexpected"] = True
    else:
        inputs = cert["payload"]["inputs"]
        if "matrix" in inputs:
            value = Q(inputs["matrix"][0][0])
            inputs["matrix"][0][0] = f"{2 * value.numerator}/{2 * value.denominator}"
        else:
            inputs["defect"] = "2/32"
    cert = seal_certificate(cert)
    assert verify_certificate_digest(cert)
    assert not locality.replay_locality_certificate(cert)


@pytest.mark.parametrize(
    "bad", (None, [], {}, {"payload": {}}, {"payload": {"type": "unknown"}})
)
def test_malformed_replay_refused(bad: object) -> None:
    assert not locality.replay_locality_certificate(bad)


def test_new_module_has_no_backend_or_consumer_imports() -> None:
    tree = ast.parse(Path(locality.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert set(modules) <= {
        "__future__",
        "collections.abc",
        "copy",
        "fractions",
        "typing",
        "omnibias.core.proof.certificate",
    }


def test_manual_documentation_examples_execute() -> None:
    manual = Path(__file__).resolve().parents[4] / "docs/core-verified-locality.md"
    blocks = re.findall(r"```python\n(.*?)\n```", manual.read_text(), re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, str(manual), "exec"), {})
