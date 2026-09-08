# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""GKSL residual (torch): shape and time-axis requirement."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_rho_components
from omnibias.qpinn.torch.equations import Lindblad, LindbladOutput, lindblad


def _rho_field(axes: tuple[str, ...]) -> OneLayerVectorField:
    torch.manual_seed(1)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(axes=axes),
        components=make_rho_components(dim=2),
        hidden=8,
        base="gaussian",
        dtype=torch.float64,
    )


def _payload() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ham = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    jumps = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], dtype=torch.complex128)
    rates = torch.tensor([0.5], dtype=torch.float64)
    return ham, jumps, rates


def test_lindblad_residual_shape() -> None:
    field = _rho_field(("x", "t"))
    coords = torch.zeros((5, 2), dtype=torch.float64)
    ham, jumps, rates = _payload()
    out = Lindblad(hamiltonian=ham, jumps=jumps, rates=rates)(field(coords))
    assert isinstance(out, LindbladOutput)
    assert out.residual.shape == (5, 8)
    out_fn = lindblad(field(coords), ham, jumps, rates)
    torch.testing.assert_close(out.residual, out_fn.residual)


def test_lindblad_requires_time_axis() -> None:
    field = _rho_field(("x",))
    coords = torch.zeros((3, 1), dtype=torch.float64)
    ham, jumps, rates = _payload()
    with pytest.raises(ValueError, match="time axis"):
        lindblad(field(coords), ham, jumps, rates)
