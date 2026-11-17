# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-neutral chain descriptor and residual demand vocabulary."""

from __future__ import annotations

import pytest
from omnibias.core.chain import ChainDescriptor, LayerDescriptor
from omnibias.core.demand import DERIVATIVE_ORDER, Demand


def test_from_widths_and_structure() -> None:
    chain = ChainDescriptor.from_widths([40, 8, 8, 1], ["tanh", "tanh", None])
    assert chain.input_dim == 40 and chain.output_dim == 1
    assert chain.widths == (40, 8, 8, 1)
    assert chain.hidden_widths == (8, 8)
    assert chain.depth == 3
    assert chain.activation_names == frozenset({"tanh"})
    assert chain.narrowest_hidden == 8
    assert chain.first_layer_ratio == 5.0
    assert chain.parameter_count == 40 * 8 + 8 + 8 * 8 + 8 + 8 * 1 + 1
    assert chain.has_affine_readout and not chain.is_ridge
    assert not chain.is_vector_field()


def test_ridge_and_vector_field_flags() -> None:
    ridge = ChainDescriptor.ridge(5, 16, "sin")
    assert ridge.is_ridge and ridge.depth == 2
    assert ridge.layers[1].bias is False
    assert ChainDescriptor.ridge(5, 16, "sin", readout_bias=True).layers[1].bias
    field = ChainDescriptor.from_widths([7, 4, 7], ["tanh", None])
    assert field.is_vector_field() and field.is_vector_field(7) and not field.is_vector_field(6)


def test_bias_sequence_and_dtype() -> None:
    chain = ChainDescriptor.from_widths(
        [3, 4, 2], ["tanh", None], bias=[True, False], dtype="float32"
    )
    assert [layer.bias for layer in chain.layers] == [True, False]
    assert chain.dtype == "float32"


@pytest.mark.parametrize(
    "build",
    [
        lambda: ChainDescriptor(()),
        lambda: ChainDescriptor((LayerDescriptor(2, 3), LayerDescriptor(4, 1))),
        lambda: ChainDescriptor((LayerDescriptor(2, 3),), dtype="float128"),
        lambda: LayerDescriptor(0, 3),
        lambda: LayerDescriptor(2, 3, activation=""),
        lambda: ChainDescriptor.from_widths([3], []),
        lambda: ChainDescriptor.from_widths([3, 2], ["tanh", None]),
        lambda: ChainDescriptor.from_widths([3, 4, 2], ["tanh", None], bias=[True]),
    ],
)
def test_invalid_descriptors_are_refused(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


def test_config_round_trip_and_fingerprint() -> None:
    chain = ChainDescriptor.from_widths(
        [9, 5, 5, 3], ["tanh", "sin", None], bias=[True, True, False]
    )
    again = ChainDescriptor.from_config(chain.to_config())
    assert again == chain and again.fingerprint() == chain.fingerprint()
    other = ChainDescriptor.from_widths(
        [9, 5, 5, 3], ["tanh", "tanh", None], bias=[True, True, False]
    )
    assert other.fingerprint() != chain.fingerprint()
    assert len(chain.fingerprint()) == 64


def test_demand_orders_and_projectability() -> None:
    assert Demand.of().order == 0
    assert Demand.of("grad_sq").order == 1
    assert Demand.of("lap", "grad_sq").order == 2
    assert Demand.of("bilap").order == 4
    assert Demand.of("directional").order == 1
    assert Demand.of("grad", "lap").needs_full_gradient
    assert not Demand.of("grad", "lap").projectable
    assert Demand.of("grad_sq", "lap", "directional").projectable
    assert "value" in Demand.of("lap") and "lap" in Demand.of("lap")


def test_demand_validation_and_union() -> None:
    with pytest.raises(ValueError, match="unknown quantities"):
        Demand.of("hessian")
    merged = Demand.of("grad_sq").union(Demand.of("bilap"))
    assert merged.quantities == frozenset({"value", "grad_sq", "bilap"})
    assert merged.to_config() == ["value", "grad_sq", "bilap"]
    assert set(Demand.of(*DERIVATIVE_ORDER).quantities) == set(DERIVATIVE_ORDER)
