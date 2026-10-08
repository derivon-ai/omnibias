# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Derivative request metadata, normalization, and bounded lazy expansion."""

import json
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
from omnibias.core.jet_algebra import JetBudget
from omnibias.core.operator_spec import OperatorPlanningBudget, OperatorSpec


@pytest.mark.parametrize("spec", [
    OperatorSpec.directional(0), OperatorSpec.directional(32), OperatorSpec.mixed((2, 0, 3)),
    OperatorSpec.laplacian(4), OperatorSpec.laplacian(0),
    OperatorSpec.polynomial({(0, 0): Fraction(1, 3), (2, 0): 2, (0, 1): -4}),
])
def test_json_roundtrip_and_immutable_query(spec: OperatorSpec) -> None:
    assert OperatorSpec.from_config(json.loads(json.dumps(spec.to_config()))) == spec
    assert spec.explain()["jet_allocated"] is False
    with pytest.raises(FrozenInstanceError):
        spec.order = 1  # type: ignore[misc]


def test_readout_weights_are_raw_derivatives() -> None:
    spec = OperatorSpec.polynomial({(2, 1): Fraction(3, 2), (0, 0): -1})
    layout = spec.jet_spec(2)
    assert dict(spec.jet_readout(layout)) == {layout.index((2, 1)): Fraction(3), 0: Fraction(-1)}
    assert OperatorSpec.directional(4).resolve_terms(20) == (((4,), Fraction(1)),)
    assert OperatorSpec.directional(4).jet_spec(20).dim == 1
    with pytest.raises(ValueError, match="dimension"):
        spec.check_dimension(3)


def test_laplacian_expansion_and_alias() -> None:
    spec = OperatorSpec.laplacian_power(2)
    assert spec == OperatorSpec.laplacian(2)
    assert dict(spec.resolve_terms(3)) == {
        (4, 0, 0): 1, (0, 4, 0): 1, (0, 0, 4): 1,
        (2, 2, 0): 2, (2, 0, 2): 2, (0, 2, 2): 2,
    }
    assert OperatorSpec.laplacian(0).resolve_terms(2) == (((0, 0), Fraction(1)),)


def test_compact_requests_do_not_allocate_general_mixed_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    compact = OperatorSpec.mixed((1,) * 32, JetBudget(max_coefficients=100))
    assert compact.order == 32
    with pytest.raises(ValueError, match="coefficients"):
        compact.jet_spec(32)
    lap = OperatorSpec.laplacian(4)
    assert lap.terms == () and lap.check_dimension(5000) == 5000

    def forbidden(*args: object) -> None:
        raise AssertionError("an oversized expansion must not begin enumeration")

    monkeypatch.setattr("omnibias.core.operator_spec.combinations_with_replacement", forbidden)
    with pytest.raises(ValueError, match="terms"):
        lap.resolve_terms(5000)
    # Even a small number of terms can carry too much coordinate metadata.
    with pytest.raises(ValueError, match="planning"):
        OperatorSpec.laplacian().resolve_terms(5000)


def test_planning_budgets_and_exact_input_validation() -> None:
    budget = OperatorPlanningBudget(max_graph_nodes=3, max_planning_work=8,
                                    max_symbol_terms=2, max_order=8, max_coefficient_bits=6)
    assert OperatorPlanningBudget.from_config(budget.to_config()) == budget
    budget.check_graph(3, 8)
    with pytest.raises(ValueError, match="graph"):
        budget.check_graph(4)
    with pytest.raises(ValueError, match="planning"):
        budget.check_graph(2, 9)
    with pytest.raises(ValueError, match="order"):
        OperatorSpec.directional(9, planning_budget=budget)
    with pytest.raises(ValueError, match="bit budget"):
        OperatorSpec.polynomial({(1,): 128}, planning_budget=budget)
    with pytest.raises(TypeError, match="Fraction"):
        OperatorSpec.polynomial({(1,): 0.5})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="dimension"):
        OperatorSpec.polynomial({(1,): 1, (0, 1): 1})
    with pytest.raises(ValueError, match="nonzero"):
        OperatorSpec.polynomial({(0,): 0})
    with pytest.raises(ValueError, match="nonempty"):
        OperatorSpec.mixed(())


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_invalid_orders(invalid: object) -> None:
    for constructor in (OperatorSpec.directional, OperatorSpec.laplacian):
        with pytest.raises(ValueError):
            constructor(invalid)  # type: ignore[arg-type]


def test_bad_serialized_metadata_is_rejected() -> None:
    config = OperatorSpec.polynomial({(1,): Fraction(2, 3)}).to_config()
    config["terms"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        OperatorSpec.from_config(config)
    config["terms"] = [[[1], True, 3]]
    with pytest.raises(ValueError, match="numerator"):
        OperatorSpec.from_config(config)
    config["terms"] = [[[1], 1, 0]]
    with pytest.raises(ValueError, match="denominator"):
        OperatorSpec.from_config(config)
    config["version"] = 7
    with pytest.raises(ValueError, match="version"):
        OperatorSpec.from_config(config)
