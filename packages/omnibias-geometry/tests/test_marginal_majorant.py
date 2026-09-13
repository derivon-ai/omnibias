# SPDX-License-Identifier: Apache-2.0
"""Independent rational Schur elimination and weighted-budget regressions."""

from __future__ import annotations

import copy
import itertools
import random
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.marginal_majorant import (
    conditional_poincare_schur as conditional,
)
from omnibias.geometry.gauge.transfer.marginal_majorant import (
    marginal_majorant_closure as certify,
)
from omnibias.geometry.gauge.transfer.marginal_majorant import (
    replay_conditional_poincare_schur_certificate as replay_conditional,
)
from omnibias.geometry.gauge.transfer.marginal_majorant import (
    replay_marginal_majorant_closure_certificate as replay,
)

Matrix = list[list[Q]]


def _decode(values: list[list[str]]) -> Matrix:
    return [[Q(v) for v in row] for row in values]


def _scalar_eliminate(matrix: Matrix, removed: tuple[int, ...], rho: Q) -> tuple[Matrix, list[int]]:
    """Rank-one elimination oracle: no inverse or production private helpers."""
    labels = list(range(len(matrix)))
    current = [row.copy() for row in matrix]
    for label in removed:
        pivot = labels.index(label)
        denominator = rho - 2 * current[pivot][pivot]
        assert denominator > 0
        keep = [i for i in range(len(labels)) if i != pivot]
        current = [[current[i][j] + 2 * current[i][pivot] * current[pivot][j] / denominator
                    for j in keep] for i in keep]
        labels = [labels[i] for i in keep]
    return current, labels


def _path_example(**kwargs: Any) -> dict[str, Any]:
    return certify([[Q(1, 20), Q(1, 40), 0],
                    [Q(1, 40), Q(1, 20), Q(1, 40)],
                    [0, Q(1, 40), Q(1, 20)]], [0, 2],
                   ricci_lower=Q(1, 2), distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                   decay_base=2, **kwargs)


def test_weighted_path_exact_kernel_output_and_shared_row_savings() -> None:
    result = _path_example()
    w = result["witness"]
    assert result["status"] == "PASS"
    assert w["effective_row_cap"] == "3/20"
    assert w["strict_ricci_margin"] == "1/5"
    assert w["inverse_kernel"] == [["5/2"]]
    assert w["weighted_inverse_kernel"] == [["5/2"]]
    assert w["output_matrix"] == [["17/320", "1/320"], ["1/320", "17/320"]]
    assert w["weighted_output_matrix"] == [["17/320", "1/80"], ["1/80", "17/320"]]
    assert w["weighted_schur_comparison_matrix"] == [["1/16", "1/80"], ["1/80", "1/16"]]
    assert w["output_weighted_row_sums"] == ["21/320", "21/320"]
    assert w["row_savings_lower"] == ["1/50", "1/50"]
    assert w["output_weighted_row_upper"] == ["13/100", "13/100"]
    assert w["retained_distances"] == [[0, 2], [2, 0]]
    assert all(w["checks"].values())
    assert replay(result["certificate"])


def test_shared_budget_is_stronger_than_independent_block_budget() -> None:
    rho, cap = Q(1), Q(1, 4)
    result = certify([[Q(1, 8), Q(1, 8)], [Q(1, 8), Q(1, 8)]], [0], ricci_lower=rho)
    naive = cap + 2 * cap**2 / (rho - 2 * cap)
    assert naive == Q(1, 2) > cap
    assert result["output_matrix"] == [["1/6"]]
    assert result["witness"]["output_weighted_row_upper"] == ["3/16"]


@pytest.mark.parametrize("size", [1, 2, 4])
def test_zero_matrix_empty_all_and_permuted_retained(size: int) -> None:
    matrix = [[Q(0)] * size for _ in range(size)]
    for keep in ([], list(range(size)), list(reversed(range(size)))):
        result = certify(matrix, keep, ricci_lower=1)
        assert result["status"] == "PASS"
        assert result["output_matrix"] == [["0"] * len(keep) for _ in keep]
        assert result["witness"]["effective_row_cap"] == "0"
        assert replay(result["certificate"])


def test_all_retained_is_identity_and_no_removed_kernel() -> None:
    matrix = [[Q(1, 10), Q(1, 20)], [Q(1, 20), Q(1, 8)]]
    result = certify(matrix, [1, 0], ricci_lower=1)
    assert _decode(result["output_matrix"]) == [[matrix[i][j] for j in [1, 0]] for i in [1, 0]]
    assert result["witness"]["inverse_kernel"] == []
    assert result["witness"]["row_savings_lower"] == ["0", "0"]


def test_all_subsets_and_elimination_orders_against_scalar_oracle() -> None:
    n, rho = 4, Q(1, 2)
    matrix = [[Q(1 + (i + j) % 3, 200) for j in range(n)] for i in range(n)]
    for count in range(n + 1):
        for removed in itertools.combinations(range(n), count):
            keep = list(reversed([i for i in range(n) if i not in removed]))
            direct = certify(matrix, keep, ricci_lower=rho)
            expected = _decode(direct["output_matrix"])
            for order in itertools.permutations(removed):
                value, labels = _scalar_eliminate(matrix, order, rho)
                assert [[value[labels.index(i)][labels.index(j)] for j in keep] for i in keep] == expected


def test_every_nested_retained_pair_associates_with_restricted_ambient_metric() -> None:
    n, rho, base = 4, Q(1, 2), Q(3, 2)
    matrix = [[Q(1 + (i + j) % 2, 300) for j in range(n)] for i in range(n)]
    distances = [[abs(i - j) for j in range(n)] for i in range(n)]
    for count in range(1, n + 1):
        for keep1_tuple in itertools.combinations(range(n), count):
            keep1 = list(reversed(keep1_tuple))
            first = certify(matrix, keep1, ricci_lower=rho, distances=distances, decay_base=base)
            cap = Q(first["witness"]["effective_row_cap"])
            for count2 in range(count + 1):
                for indices in itertools.combinations(range(count), count2):
                    keep2 = [keep1[i] for i in indices]
                    nested = certify(_decode(first["output_matrix"]), indices, ricci_lower=rho,
                                     row_cap=cap, distances=first["witness"]["retained_distances"],
                                     decay_base=base)
                    direct = certify(matrix, keep2, ricci_lower=rho, row_cap=cap,
                                     distances=distances, decay_base=base)
                    assert nested["status"] == direct["status"] == "PASS"
                    assert nested["output_matrix"] == direct["output_matrix"]
                    assert nested["witness"]["retained_distances"] == direct["witness"]["retained_distances"]


@pytest.mark.parametrize("seed", range(12))
def test_seeded_rational_matrices_exact_weighted_closure(seed: int) -> None:
    rng, n, rho, base, cap = random.Random(seed), 5, Q(1, 2), Q(2), Q(1, 8)
    matrix = [[Q(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            matrix[i][j] = matrix[j][i] = Q(rng.randrange(8), rng.randrange(1, 8))
    distances = [[abs(i - j) for j in range(n)] for i in range(n)]
    maxrow = max(sum((matrix[i][j] * base**distances[i][j] for j in range(n)), Q(0)) for i in range(n))
    matrix = [[v * cap / maxrow for v in row] for row in matrix]
    removed = tuple(rng.sample(range(n), 3))
    expected, labels = _scalar_eliminate(matrix, removed, rho)
    result = certify(matrix, labels, ricci_lower=rho, row_cap=cap, distances=distances, decay_base=base)
    assert _decode(result["output_matrix"]) == expected
    assert all(Q(v) <= cap for v in result["witness"]["output_weighted_row_sums"])
    assert all(result["witness"]["checks"].values())
    assert replay(result["certificate"])


@pytest.mark.parametrize("cap,expected", [(Q(1, 4) - Q(1, 10**12), "PASS"),
                                         (Q(1, 4), "INCONCLUSIVE"),
                                         (Q(1, 4) + Q(1, 10**12), "INCONCLUSIVE")])
def test_strict_ricci_boundary_is_exact(cap: Q, expected: str) -> None:
    result = certify([[Q(1, 10), 0], [0, Q(1, 10)]], [0], ricci_lower=Q(1, 2), row_cap=cap)
    assert result["status"] == expected
    assert replay(result["certificate"])
    if expected == "INCONCLUSIVE":
        assert result["output_matrix"] is None
        assert result["witness"]["failed_constraints"] == ["strict_ricci_margin"]


def test_too_small_supplied_weighted_cap_refuses_without_computing_output() -> None:
    result = _path_example(row_cap=Q(3, 20) - Q(1, 10**12))
    assert result["status"] == "INCONCLUSIVE"
    assert result["output_matrix"] is None
    assert result["witness"]["failed_constraints"] == ["supplied_row_cap_below_actual_weighted_rows"]
    assert result["witness"]["checks"] == {}
    assert replay(result["certificate"])


@pytest.mark.parametrize("kwargs", [
    {"matrix": []}, {"matrix": [[1, 0]]}, {"matrix": "1"},
    {"matrix": [[-1]]}, {"matrix": [[0, 1], [0, 0]]},
    {"matrix": [[True]]}, {"matrix": [[0.01]]}, {"matrix": [["1/10"]]},
    {"retained": [True]}, {"retained": [0.0]}, {"retained": [1]},
    {"retained": [-1]}, {"retained": [0, 0]}, {"retained": "0"},
    {"ricci_lower": True}, {"ricci_lower": 1.0}, {"ricci_lower": 0},
    {"row_cap": False}, {"row_cap": 0.1}, {"row_cap": -1},
    {"decay_base": True}, {"decay_base": 1.0}, {"decay_base": Q(1, 2)},
    {"decay_base": 2}, {"distances": [[True]]}, {"distances": [[0.0]]},
    {"distances": [[1]]}, {"distances": []},
])
def test_invalid_exact_inputs_raise(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {"matrix": [[Q(1, 10)]], "retained": [0], "ricci_lower": 1}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        certify(**arguments)


@pytest.mark.parametrize("distances", [
    [[0, 0], [0, 0]], [[0, 1], [2, 0]], [[0, -1], [-1, 0]],
    [[0, 1, 3], [1, 0, 1], [3, 1, 0]],
])
def test_invalid_metric_geometry_refused(distances: list[list[int]]) -> None:
    n = len(distances)
    with pytest.raises(ValueError):
        certify([[0] * n for _ in range(n)], [0], ricci_lower=1, distances=distances, decay_base=2)


@pytest.mark.parametrize("key", [
    "actual_vacuum_hessian_verified", "actual_marginal_verified", "all_scale_refinement_claim",
    "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim", "yang_mills_mass_gap_claim",
])
def test_scope_is_not_earned_and_resealed_forgery_is_rejected(key: str) -> None:
    result = _path_example()
    assert result[key] is False
    assert result["certificate"]["honesty"][key] is False
    forged = copy.deepcopy(result["certificate"])
    forged["honesty"][key] = True
    forged = seal_certificate(forged)
    assert verify_certificate_digest(forged)
    assert not replay(forged)


@pytest.mark.parametrize("field", [
    "inverse_kernel", "weighted_inverse_kernel", "output_matrix", "weighted_output_matrix",
    "weighted_schur_comparison_matrix", "output_weighted_row_sums", "row_savings_lower",
    "output_weighted_row_upper", "retained_distances", "checks", "inputs", "analytic_premises",
])
def test_resealed_arithmetic_and_input_mutations_are_not_certificates(field: str) -> None:
    forged = copy.deepcopy(_path_example()["certificate"])
    witness = forged["payload"]["witness"]
    if field == "inputs":
        witness[field]["ricci_lower"] = "3/5"
    elif field == "checks":
        witness[field]["same_weighted_row_cap_verified"] = False
    elif field == "analytic_premises":
        witness[field] = []
    elif field == "retained_distances":
        witness[field] = [[0, 1], [1, 0]]
    elif isinstance(witness[field][0], list):
        witness[field][0][0] = "0"
    else:
        witness[field][0] = "0"
    forged = seal_certificate(forged)
    assert verify_certificate_digest(forged)
    assert not replay(forged)


@pytest.mark.parametrize("value", [None, [], 1, "certificate", {}, {"digest": []}])
def test_malformed_top_level_replay_returns_false(value: Any) -> None:
    assert not replay(value)


def test_reserved_formal_tiers_not_earned_and_noncanonical_bool_rejected() -> None:
    result = _path_example()
    assert result["theorem_prover_verified"] is False
    assert result["mathlib_verified"] is False
    forged = copy.deepcopy(result["certificate"])
    forged["payload"]["witness"]["checks"]["same_weighted_row_cap_verified"] = 1
    assert not replay(seal_certificate(forged))


def test_public_transfer_exports() -> None:
    from omnibias.geometry.gauge import transfer

    assert transfer.marginal_majorant_closure is certify
    assert transfer.replay_marginal_majorant_closure_certificate is replay


def _conditional_example() -> dict[str, Any]:
    return conditional([3, 4, 5], [[0, Q(1, 4), 0], [Q(1, 4), 0, Q(1, 2)], [0, Q(1, 2), 0]],
                       [0, 2], distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]], decay_base=2)


def _schur_h(matrix: Matrix, removed: tuple[int, ...]) -> tuple[Matrix, list[int]]:
    """Independent elimination directly on H, with no auxiliary rho."""
    labels = list(range(len(matrix)))
    value = [row.copy() for row in matrix]
    for label in removed:
        k = labels.index(label)
        assert value[k][k] > 0
        keep = [i for i in range(len(labels)) if i != k]
        value = [[value[i][j] - value[i][k] * value[k][j] / value[k][k]
                  for j in keep] for i in keep]
        labels = [labels[i] for i in keep]
    return value, labels


def test_conditional_unequal_gaps_exact_weighted_schur_and_no_ricci_ceiling() -> None:
    result = _conditional_example()
    w = result["witness"]
    assert result["status"] == "PASS"
    assert w["reference_rho"] == "5"
    assert w["input_weighted_dominance_margins"] == ["2", "1", "3"]
    assert w["uniform_dominance_lower"] == "1"
    assert result["output_gaps"] == ["47/16", "19/4"]
    assert result["output_mixed_hessian"] == [["0", "1/16"], ["1/16", "0"]]
    assert w["output_comparison_matrix"] == [["47/16", "-1/8"], ["-1/8", "19/4"]]
    assert w["output_weighted_dominance_margins"] == ["39/16", "17/4"]
    assert w["retained_distances"] == [[0, 2], [2, 0]]
    assert "not a geometric Ricci" in w["reference_scope"]
    assert replay(w["source_certificate"])
    assert replay_conditional(result["certificate"])


def test_conditional_nested_transform_changes_auxiliary_rho_without_changing_schur() -> None:
    first = _conditional_example()
    second = conditional([Q(v) for v in first["output_gaps"]],
                         _decode(first["output_mixed_hessian"]), [0],
                         distances=first["witness"]["retained_distances"], decay_base=2)
    direct = conditional([3, 4, 5], [[0, Q(1, 4), 0], [Q(1, 4), 0, Q(1, 2)], [0, Q(1, 2), 0]],
                         [0], distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]], decay_base=2)
    assert second["witness"]["reference_rho"] == "19/4" != first["witness"]["reference_rho"]
    assert second["output_gaps"] == direct["output_gaps"] == ["223/76"]
    assert second["output_mixed_hessian"] == direct["output_mixed_hessian"] == [["0"]]


def test_conditional_every_subset_matches_independent_h_elimination() -> None:
    n = 4
    gaps = [Q(i + 2) for i in range(n)]
    mixed = [[Q(0) if i == j else Q(1 + (i + j) % 3, 50) for j in range(n)] for i in range(n)]
    h = [[gaps[i] if i == j else -2 * mixed[i][j] for j in range(n)] for i in range(n)]
    distances = [[abs(i - j) for j in range(n)] for i in range(n)]
    for count in range(n + 1):
        for removed in itertools.combinations(range(n), count):
            keep = list(reversed([i for i in range(n) if i not in removed]))
            result = conditional(gaps, mixed, keep, distances=distances, decay_base=2)
            assert result["status"] == "PASS"
            for order in itertools.permutations(removed):
                expected, labels = _schur_h(h, order)
                assert _decode(result["witness"]["output_comparison_matrix"]) == [
                    [expected[labels.index(i)][labels.index(j)] for j in keep] for i in keep]


@pytest.mark.parametrize("gaps", [[Q(1, 10)], [Q(10)], [Q(1, 2), Q(20), Q(30)]])
def test_conditional_zero_mixed_all_empty_permuted_and_high_gaps(gaps: list[Q]) -> None:
    n = len(gaps)
    for keep in ([], list(range(n)), list(reversed(range(n)))):
        result = conditional(gaps, [[0] * n for _ in range(n)], keep)
        assert result["status"] == "PASS"
        assert result["output_gaps"] == [str(gaps[i]) for i in keep]
        assert result["output_mixed_hessian"] == [["0"] * len(keep) for _ in keep]
        assert Q(result["witness"]["uniform_dominance_lower"]) == min(gaps)
        assert replay_conditional(result["certificate"])


@pytest.mark.parametrize("coupling,status", [(Q(1, 2) - Q(1, 10**12), "PASS"),
                                            (Q(1, 2), "INCONCLUSIVE"),
                                            (Q(1, 2) + Q(1, 10**12), "INCONCLUSIVE")])
def test_conditional_strict_dominance_boundary_and_replay(coupling: Q, status: str) -> None:
    result = conditional([1, 1], [[0, coupling], [coupling, 0]], [0])
    assert result["status"] == status
    assert Q(result["witness"]["uniform_dominance_lower"]) == 1 - 2 * coupling
    assert replay_conditional(result["certificate"])
    if status == "INCONCLUSIVE":
        assert result["output_gaps"] is None
        assert result["output_mixed_hessian"] is None
        assert "weighted_diagonal_dominance_verified" in result["witness"]["failed_constraints"]


def test_conditional_weights_can_refuse_an_unweighted_pass() -> None:
    matrix: Matrix = [[Q(0), Q(1, 3)], [Q(1, 3), Q(0)]]
    assert conditional([1, 1], matrix, [0])["status"] == "PASS"
    weighted = conditional([1, 1], matrix, [0], distances=[[0, 1], [1, 0]], decay_base=2)
    assert weighted["status"] == "INCONCLUSIVE"
    assert weighted["witness"]["uniform_dominance_lower"] == "-1/3"


@pytest.mark.parametrize("kwargs", [
    {"gaps": []}, {"gaps": "1"}, {"gaps": [True]}, {"gaps": [1.0]},
    {"gaps": ["1"]}, {"gaps": [0]}, {"gaps": [-1]}, {"gaps": [1, 2]},
    {"mixed_hessian": [[Q(1, 10)]]}, {"mixed_hessian": [[True]]},
    {"mixed_hessian": [[0.0]]}, {"mixed_hessian": [["0"]]},
    {"mixed_hessian": []}, {"mixed_hessian": [[-1]]},
    {"retained": [True]}, {"retained": [0.0]}, {"retained": [1]},
    {"decay_base": 1.0}, {"decay_base": True}, {"decay_base": Q(1, 2)},
    {"decay_base": 2}, {"distances": [[1]]}, {"distances": [[False]]},
])
def test_conditional_exact_input_guards(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {"gaps": [1], "mixed_hessian": [[0]], "retained": [0]}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        conditional(**arguments)


@pytest.mark.parametrize("key", [
    "actual_conditional_poincare_verified", "actual_mixed_hessian_verified", "actual_vacuum_hessian_verified",
    "actual_marginal_verified", "physical_gap_verified", "all_scale_refinement_claim",
    "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim", "yang_mills_mass_gap_claim",
])
def test_conditional_analytic_premises_and_parent_flags_are_not_earned(key: str) -> None:
    result = _conditional_example()
    assert result[key] is result["certificate"]["honesty"][key] is False
    certificate = copy.deepcopy(result["certificate"])
    certificate["honesty"][key] = True
    assert not replay_conditional(seal_certificate(certificate))


@pytest.mark.parametrize("field", ["inputs", "reference_rho", "output_gaps", "output_mixed_hessian",
                                  "output_comparison_matrix", "output_weighted_dominance_margins",
                                  "uniform_dominance_lower", "retained_distances", "checks",
                                  "source_certificate", "analytic_premises"])
def test_conditional_resealed_output_and_nested_source_tampering(field: str) -> None:
    certificate = copy.deepcopy(_conditional_example()["certificate"])
    witness = certificate["payload"]["witness"]
    if field == "inputs":
        witness[field]["gaps"][0] = "7"
    elif field == "source_certificate":
        source = witness[field]
        source["payload"]["witness"]["output_matrix"][0][0] = "0"
        witness[field] = seal_certificate(source)
    elif field == "checks":
        witness[field]["weighted_diagonal_dominance_verified"] = 1
    elif field == "analytic_premises":
        witness[field] = []
    elif field == "output_mixed_hessian":
        witness[field][0][1] = "0"
    elif isinstance(witness[field], str):
        witness[field] = "0"
    elif isinstance(witness[field][0], list):
        witness[field][0][0] = "0"
    else:
        witness[field][0] = "0"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay_conditional(certificate)


@pytest.mark.parametrize("value", [None, [], 1, "certificate", {}, {"digest": []}])
def test_conditional_malformed_replay(value: Any) -> None:
    assert not replay_conditional(value)


def test_conditional_public_transfer_exports() -> None:
    from omnibias.geometry.gauge import transfer

    assert transfer.conditional_poincare_schur is conditional
    assert transfer.replay_conditional_poincare_schur_certificate is replay_conditional
