# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures for the torch backend tests.

We build small, deterministic OneLayerVectorField instances carrying a
``(psi_re, psi_im)`` group; the small width keeps unit tests well below
1 second on CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components


@pytest.fixture
def coord_xt() -> CoordinateSpec:
    return CoordinateSpec(axes=("x", "t"))


@pytest.fixture
def coord_x() -> CoordinateSpec:
    return CoordinateSpec(axes=("x",))


@pytest.fixture
def psi_field_xt(coord_xt: CoordinateSpec) -> OneLayerVectorField:
    torch.manual_seed(0)
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord_xt,
        components=spec,
        hidden=8,
        base="gaussian",
        dtype=torch.float64,
    )


@pytest.fixture
def psi_field_x(coord_x: CoordinateSpec) -> OneLayerVectorField:
    torch.manual_seed(0)
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord_x,
        components=spec,
        hidden=8,
        base="gaussian",
        dtype=torch.float64,
    )


@pytest.fixture
def coords_xt() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(16, 2, dtype=torch.float64)


@pytest.fixture
def coords_x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.linspace(-2.0, 2.0, 17, dtype=torch.float64).unsqueeze(-1)
