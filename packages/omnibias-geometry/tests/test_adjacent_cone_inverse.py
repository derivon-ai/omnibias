# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.adjacent_cone_inverse import (
    _anchors,
    _dual_in_family,
)
from omnibias.geometry.gauge.transfer.adjacent_cone_inverse import (
    replay_su2_adjacent_cone_inverse_certificate as replay,
)
from omnibias.geometry.gauge.transfer.adjacent_cone_inverse import (
    su2_adjacent_cone_inverse as certify,
)


@pytest.fixture(scope="module")
def row() -> dict[str, Any]:
    return certify(6)


def test_original_inverse_duals_improve_the_same_canonical_parent(row: dict[str, Any]) -> None:
    assert row["status"] == "PASS"
    assert Q(row["original_N_inverse_upper"]) < Q(57, 25)
    old = row["witness"]["source_inverse_certificate"]["payload"]["witness"]["arithmetic"]
    assert Q(old["original_N_inverse_upper"]) > 3
    assert replay(row["certificate"])
    for dual in [*row["witness"]["preconditioner_duals"], row["witness"]["mixed_defect_dual"]]:
        weights = [Q(v) for v in dual["weights"]]
        assert min(weights) >= 0
        for constraint in dual["constraint_rows"]:
            assert sum((Q(x) * y for x, y in zip(constraint["weights"], weights, strict=True)), Q(0)) >= Q(constraint["lower"])
        assert dual["global_dual_optimality_claim"] is False


def test_single_column_hypothesis_is_false_for_original_anchor_norm() -> None:
    # Three normalized triangle rays all have N1. A maps each to ray101.
    rays = [(1, 0, 1), (0, 1, 1), (1, 1, 0)]
    normalized = [tuple(v / max(_anchors(s)) for v in _anchors(s)) for s in rays]
    assert all(max(v) == 1 for v in normalized)
    combined_input = tuple(sum(v[i] for v in normalized) for i in range(3))
    combined_output = tuple(3 * v for v in normalized[0])
    assert max(combined_input) == 2
    assert max(combined_output) == 3
    # Thus the largest column ratio1 cannot bound the full operator by1.


def test_exact_dual_search_handles_zero_coordinates_and_dependent_rows() -> None:
    constraints = [((Q(1), Q(0), Q(1)), Q(2)), ((Q(2), Q(0), Q(2)), Q(4))]
    result = _dual_in_family(constraints, (Q(1), Q(0), Q(0)), (Q(0), Q(0), Q(1)))
    assert Q(result["sum"]) == 2
    assert result["dual_feasibility_verified"]


def test_failed_parent_is_an_inconclusive_canonical_replay() -> None:
    row = certify(4, cutoff=1)
    assert row["status"] == "INCONCLUSIVE"
    assert row["original_N_inverse_upper"] is None
    assert replay(row["certificate"])


@pytest.mark.parametrize("key,value", [("original_N_inverse_upper", "1/100"),
                                     ("preconditioner_N_upper", "0"),
                                     ("mixed_ZR_N_to_M_upper", "0")])
def test_resealed_arithmetic_changes_fail(row: dict[str, Any], key: str, value: str) -> None:
    certificate = deepcopy(row["certificate"])
    certificate["payload"]["witness"]["arithmetic"][key] = value
    assert not replay(seal_certificate(certificate))


def test_dual_mutation_resealed_cannot_replace_full_tail(row: dict[str, Any]) -> None:
    certificate = deepcopy(row["certificate"])
    certificate["payload"]["witness"]["mixed_defect_dual"]["weights"] = ["0", "0", "0"]
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("flag", ["actual_vacuum_verified", "continuum_claim", "yang_mills_mass_gap_claim"])
def test_reference_inverse_never_earns_physical_or_parent_flags(row: dict[str, Any], flag: str) -> None:
    assert row[flag] is False
    certificate = deepcopy(row["certificate"])
    certificate["honesty"][flag] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("kappa", [True, 6.0, "6", None, 0, -1])
def test_exact_coupling_guards(kappa: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(kappa)


@pytest.mark.parametrize("value", [None, 0, [], {}, "x"])
def test_malformed_replay(value: Any) -> None:
    assert not replay(value)
