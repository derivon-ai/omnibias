# SPDX-License-Identifier: Apache-2.0
"""Exact block/tail regressions; independent radial identities live in ensemble-laws."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.plaquette_resolvent import (
    replay_su2_plaquette_linearized_inverse_certificate as replay,
)
from omnibias.geometry.gauge.transfer.plaquette_resolvent import (
    su2_plaquette_linearized_inverse as certify,
)


def _matrix(t: Q, size: int) -> list[list[Q]]:
    # Independent dense construction from the projected character product rule.
    matrix = [[Q(i == j) for j in range(size)] for i in range(size)]
    for n in range(1, size + 1):
        if n < size:
            matrix[n][n - 1] += 2 * t * n / ((n + 1) * (n + 3))
        if n > 1:
            matrix[n - 2][n - 1] -= 2 * t * (n + 2) / ((n - 1) * (n + 1))
    return matrix


def _inverse(matrix: list[list[Q]]) -> list[list[Q]]:
    # Dense Gauss--Jordan, independent of the production tridiagonal LU.
    size = len(matrix)
    aug = [row[:] + [Q(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for j in range(size):
        pivot = next(i for i in range(j, size) if aug[i][j])
        aug[j], aug[pivot] = aug[pivot], aug[j]
        value = aug[j][j]
        aug[j] = [x / value for x in aug[j]]
        for i in range(size):
            if i != j:
                factor = aug[i][j]
                aug[i] = [x - factor * y for x, y in zip(aug[i], aug[j], strict=True)]
    return [row[size:] for row in aug]


def _weight(n: int) -> Q:
    return Q(n * n * (n + 2) * (n + 1) ** 3, 2)


def _column_norms(matrix: list[list[Q]]) -> list[Q]:
    return [
        sum((_weight(i + 1) * abs(row[j]) for i, row in enumerate(matrix)), Q(0)) / _weight(j + 1)
        for j in range(len(matrix))
    ]


@pytest.fixture(scope="module")
def witness() -> dict[str, Any]:
    return certify(1, cutoff=4)


def test_finite_preconditioning_passes_beyond_the_exact_unpreconditioned_norm(
    witness: dict[str, Any],
) -> None:
    arithmetic = witness["witness"]["arithmetic"]
    assert arithmetic["exact_unpreconditioned_fourier_norm"] == "6"
    assert arithmetic["generic_linear_norm_upper"] == "128/3"
    assert arithmetic["all_spin_defect_norm"] == "187174/194355"
    assert arithmetic["preconditioner_norm"] == "7647/617"
    assert witness["fourier_inverse_upper"] == "2408805/7181"
    assert witness["weighted_hilbert_inverse_upper"] == "1"
    assert witness["status"] == "PASS" and witness["finite_gate_verified"]
    assert replay(witness["certificate"])


@pytest.mark.parametrize("kappa, cutoff", [(1, 16), (Q(1, 2), 32), (2, 8)])
def test_stronger_couplings_keep_the_hilbert_and_fourier_norms_distinct(
    kappa: int | Q, cutoff: int
) -> None:
    result = certify(kappa, cutoff=cutoff)
    arithmetic = result["witness"]["arithmetic"]
    assert Q(arithmetic["exact_unpreconditioned_fourier_norm"]) > 1
    assert 0 < Q(arithmetic["all_spin_defect_norm"]) < 1
    assert Q(result["fourier_inverse_upper"]) > 1
    assert result["weighted_hilbert_inverse_upper"] == "1"
    assert result["full_spin_fourier_reference_inverse_verified"] is True
    assert replay(result["certificate"])


def test_small_cutoff_refuses_fourier_bound_but_preserves_unconditional_hilbert_result() -> None:
    result = certify(1, cutoff=1)
    assert result["status"] == "INCONCLUSIVE"
    assert result["witness"]["arithmetic"]["all_spin_defect_norm"] == "6"
    assert result["full_spin_fourier_reference_inverse_verified"] is False
    assert result["fourier_inverse_upper"] is None
    assert result["weighted_hilbert_reference_inverse_verified"] is True
    assert result["weighted_hilbert_inverse_upper"] == "1"
    assert replay(result["certificate"])  # Faithfully replaying a refusal is allowed.


@pytest.mark.parametrize("kappa, cutoff", [(1, 4), (2, 3), (3, 2), (Q(5, 2), 6)])
def test_dense_exact_blocks_and_first_omitted_coupling_reproduce_full_defect(
    kappa: int | Q, cutoff: int
) -> None:
    t = Q(4, 3) / kappa**2
    result = certify(kappa, cutoff=cutoff)
    arithmetic = result["witness"]["arithmetic"]
    retained = _inverse(_matrix(t, cutoff))
    assert _column_norms(retained) == [
        Q(x) for x in arithmetic["retained_inverse_weighted_column_norms"]
    ]
    size = cutoff + 8
    operator = _matrix(t, size)
    defect = [[Q(0)] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            product = (
                sum((retained[i][k] * operator[k][j] for k in range(cutoff)), Q(0))
                if i < cutoff
                else operator[i][j]
            )
            defect[i][j] = Q(i == j) - product
    norms = _column_norms(defect)
    assert all(value == 0 for value in norms[: cutoff - 1])
    assert norms[cutoff - 1] == Q(arithmetic["retained_to_omitted_defect"])
    assert norms[cutoff] == Q(arithmetic["first_omitted_column_defect"])
    assert norms[cutoff + 1] == Q(arithmetic["all_further_omitted_columns_defect"])
    assert max(norms) == Q(arithmetic["all_spin_defect_norm"])
    if result["finite_gate_verified"]:
        # A larger full matrix must satisfy the same norm estimate, not merely
        # the retained block. This catches omitted-boundary mistakes.
        assert max(_column_norms(_inverse(operator))) <= Q(result["fourier_inverse_upper"])


def test_full_tail_monotonicity_is_an_exact_positive_polynomial_identity() -> None:
    # Rational evaluation checks the proved symbolic identity at large labels;
    # it does not replace its universal polynomial proof.
    for n in [*range(1, 40), 2**30 + 1, 2**54 + 3]:
        a = Q(2 * (n + 2) ** 2, n * (n + 1) ** 2)
        b = Q(2 * n * (n - 1), (n + 1) ** 3)
        an = Q(2 * (n + 3) ** 2, (n + 1) * (n + 2) ** 2)
        bn = Q(2 * (n + 1) * n, (n + 2) ** 3)
        positive = Q(
            4 * (n**5 + 6 * n**4 + 22 * n**3 + 47 * n**2 + 47 * n + 16),
            n * (n + 1) ** 3 * (n + 2) ** 3,
        )
        assert a + b - an - bn == positive > 0


@pytest.mark.parametrize("kappa", [True, False, 1.0, "1", None, 0, -1])
def test_nonexact_or_nonpositive_couplings_are_rejected(kappa: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(kappa)


@pytest.mark.parametrize("cutoff", [True, False, 4.0, "4", None, Q(4), 0, -1])
def test_cutoff_is_a_strict_positive_integer(cutoff: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(1, cutoff=cutoff)


@pytest.mark.parametrize(
    "field",
    [
        "retained_to_omitted_defect",
        "first_omitted_column_defect",
        "all_further_omitted_columns_defect",
        "fourier_inverse_upper",
        "retained_inverse_weighted_column_norms",
        "retained_lu_pivots",
        "retained_skew_balance_verified",
    ],
)
def test_resealed_arithmetic_tampering_is_rejected(witness: dict[str, Any], field: str) -> None:
    certificate = deepcopy(witness["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "0"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize(
    "field",
    [
        "actual_vacuum_claim",
        "actual_vacuum_verified",
        "physical_mass_gap_claim",
        "uniform_in_volume_claim",
        "all_scale_refinement_claim",
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "theorem_prover_verified",
        "mathlib_verified",
    ],
)
def test_reference_cannot_be_resealed_as_target_or_parent(
    witness: dict[str, Any], field: str
) -> None:
    assert witness[field] is False
    certificate = deepcopy(witness["certificate"])
    certificate["honesty"][field] = True
    assert not replay(seal_certificate(certificate))


def test_raw_inputs_operator_norm_and_unknown_fields_are_canonical(witness: dict[str, Any]) -> None:
    for key, value in [("kappa", "2/2"), ("cutoff", True)]:
        certificate = deepcopy(witness["certificate"])
        certificate["payload"]["witness"]["inputs"][key] = value
        assert not replay(seal_certificate(certificate))
    for key in ("operator", "fourier_norm", "constant_mode", "invented_premise"):
        certificate = deepcopy(witness["certificate"])
        certificate["payload"]["witness"][key] = "unearned"
        assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("certificate", [None, [], (), "certificate", 1, True, {}])
def test_malformed_certificate_returns_false(certificate: Any) -> None:
    assert replay(certificate) is False
