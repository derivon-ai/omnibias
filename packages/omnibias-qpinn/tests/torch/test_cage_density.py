# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard density-matrix cage (torch): Hermitian, trace-1, PSD by construction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_rho_components, rho_entry_names
from omnibias.qpinn.torch.cage import DensityMatrixField, make_density_matrix_field


def _assemble(state, dim: int = 2) -> torch.Tensor:
    rows = []
    for i in range(dim):
        row = []
        for j in range(dim):
            re_n, im_n = rho_entry_names("rho", i, j)
            row.append(torch.complex(state.ops.value(state, re_n), state.ops.value(state, im_n)))
        rows.append(torch.stack(row, dim=-1))
    return torch.stack(rows, dim=-2)


def _base_field() -> OneLayerVectorField:
    torch.manual_seed(0)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(axes=("x", "t")),
        components=make_rho_components(dim=2),
        hidden=8,
        base="gaussian",
        dtype=torch.float64,
    )


def test_caged_rho_is_trace_one_hermitian_psd() -> None:
    cage = make_density_matrix_field(base=_base_field(), dim=2)
    assert isinstance(cage, DensityMatrixField)
    coords = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    grid = torch.stack([coords, torch.zeros_like(coords)], dim=-1)
    state = cage(grid)
    rho = _assemble(state)
    herm = rho - rho.conj().transpose(-2, -1)
    traces = torch.einsum("...ii->...", rho).real
    torch.testing.assert_close(herm, torch.zeros_like(herm), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(traces, torch.ones_like(traces), atol=1e-12, rtol=0.0)
    a = rho.real
    b = rho.imag
    realified = torch.cat(
        [torch.cat([a, -b], dim=-1), torch.cat([b, a], dim=-1)],
        dim=-2,
    )
    eigs = torch.linalg.eigvalsh(realified)
    assert torch.all(eigs >= -1e-10)


def test_first_order_time_derivative_exists() -> None:
    cage = make_density_matrix_field(base=_base_field(), dim=2)
    coords = torch.zeros((4, 2), dtype=torch.float64)
    state = cage(coords)
    d_re = state.ops.derivative(state, "rho_re_0_1", axis="t", order=1)
    assert d_re.shape == (4,)
    with pytest.raises(NotImplementedError):
        state.ops.derivative(state, "rho_re_0_1", axis="t", order=2)
