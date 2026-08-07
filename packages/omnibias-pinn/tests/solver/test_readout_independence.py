# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract: FieldState caches must be independent of the readout parameters.

The frozen-feature linear solver builds a collocation plan once and re-reads
the same states while sweeping the readout. Declaring fields must match a
freshly-built field after a readout mutation; non-declaring fields must be
refused by the seam.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.fields._core.field_base import READOUT_INDEPENDENT_ATTR
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.constrained import HardCondition, dirichlet
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.solver._core.readout import (
    ReadoutDependentError,
    requires_readout_independent,
)
from omnibias.pinn.solver.torch.readout import readout_size, set_readout
from omnibias.pinn.torch.cage.conservation import HardBoundaryField
from omnibias.pinn.torch.cage.constrained import ConstrainedExpressionField
from omnibias.pinn.torch.cage.incompressible import StreamfunctionField
from omnibias.pinn.torch.cage.integral import IntegralConservationField
from omnibias.pinn.torch.fields.chebyshev import ChebyshevVectorField
from omnibias.pinn.torch.fields.jet_mlp import JetMLPVectorField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

# Compatibility aliases used by older call sites in this module.
_readout_size = readout_size
_set_readout = set_readout

DTYPE = torch.float64


def _space_time_spec() -> CoordinateSpec:
    return CoordinateSpec(
        axes=("t", "x"),
        periodicity=(False, True),
        time_axis="t",
        domain=((0.0, 1.0), (0.0, 1.0)),
    )


def _space_spec() -> CoordinateSpec:
    return CoordinateSpec(
        axes=("x", "y"),
        periodicity=(False, False),
        domain=((-1.0, 1.0), (-1.0, 1.0)),
    )


def _u_components() -> ComponentSpec:
    return ComponentSpec(("u",))


def _make_one_layer(*, seed: int = 0) -> OneLayerVectorField:
    torch.manual_seed(seed)
    return OneLayerVectorField(
        coordinate_spec=_space_spec(),
        components=_u_components(),
        hidden=8,
        base="tanh",
        dtype=DTYPE,
    )


def _make_spectral(*, seed: int = 0) -> SpectralVectorField:
    torch.manual_seed(seed)
    field = SpectralVectorField(
        coordinate_spec=_space_time_spec(),
        components=_u_components(),
        K=2,
        time_hidden=6,
        time_depth=1,
        activation="tanh",
        dtype=DTYPE,
    )
    with torch.no_grad():
        field.W_t.normal_(0.0, 0.5)
        field.beta_t.normal_(0.0, 0.1)
        field.V.normal_(0.0, 0.1)
        field.b_t.normal_(0.0, 0.1)
    return field


def _make_chebyshev(*, seed: int = 0) -> ChebyshevVectorField:
    torch.manual_seed(seed)
    field = ChebyshevVectorField(
        coordinate_spec=_space_time_spec(),
        components=_u_components(),
        K=3,
        domain=((0.0, 1.0),),
        time_hidden=6,
        time_depth=1,
        activation="tanh",
        dtype=DTYPE,
    )
    with torch.no_grad():
        field.W_t.normal_(0.0, 0.5)
        field.beta_t.normal_(0.0, 0.1)
        field.V.normal_(0.0, 0.1)
        field.b_t.normal_(0.0, 0.1)
    return field


def _make_jet(*, seed: int = 0) -> JetMLPVectorField:
    torch.manual_seed(seed)
    return JetMLPVectorField(
        coordinate_spec=_space_spec(),
        components=_u_components(),
        hidden=8,
        depth=2,
        base="tanh",
        jet_order=2,
        dtype=DTYPE,
    )


def _mutate_readout(field: object) -> None:
    """Perturb the live readout in place."""
    if isinstance(field, OneLayerVectorField):
        with torch.no_grad():
            field.c.weight.add_(0.37)
            field.c.bias.add_(-0.11)
        return
    if isinstance(field, (SpectralVectorField, ChebyshevVectorField)):
        with torch.no_grad():
            field.V.add_(0.29)
            field.b_t.add_(-0.07)
        return
    if isinstance(field, JetMLPVectorField):
        with torch.no_grad():
            field.net.linears[-1].weight.add_(0.31)
            field.net.linears[-1].bias.add_(-0.13)
        return
    if hasattr(field, "base"):
        _mutate_readout(field.base)
        return
    raise TypeError(f"no readout mutation for {type(field).__name__}")


def _coords_for(field: object) -> torch.Tensor:
    spec = field.coordinate_spec  # type: ignore[attr-defined]
    torch.manual_seed(3)
    return torch.rand(5, spec.ndim, dtype=DTYPE)


@pytest.mark.parametrize(
    "factory",
    [_make_one_layer, _make_spectral, _make_chebyshev, _make_jet],
    ids=["one_layer", "spectral", "chebyshev", "jet"],
)
def test_declaring_field_matches_fresh_field_after_readout_mutation(factory) -> None:
    field = factory(seed=0)
    assert getattr(field, READOUT_INDEPENDENT_ATTR) is True
    coords = _coords_for(field)
    state = field(coords)
    name = field.components.names[0]
    _ = state.ops.value(state, name)
    _ = state.ops.gradient(state, name)
    _mutate_readout(field)
    stale_value = state.ops.value(state, name)
    stale_grad = state.ops.gradient(state, name)
    fresh = factory(seed=0)
    _mutate_readout(fresh)
    fresh_state = fresh(coords)
    assert torch.equal(stale_value, fresh_state.ops.value(fresh_state, name))
    assert torch.equal(stale_grad, fresh_state.ops.gradient(fresh_state, name))


def test_one_layer_passes_readout_size_gate() -> None:
    field = _make_one_layer()
    c, h, n = _readout_size(field)
    assert c == 1 and h == 8 and n == 1 * 8 + 1
    theta = torch.zeros(n, dtype=DTYPE)
    _set_readout(field, theta)
    assert torch.equal(field.c.weight, torch.zeros_like(field.c.weight))


def test_nonlinear_integral_cage_is_refused() -> None:
    base = _make_one_layer()
    cage = IntegralConservationField(
        base=base,
        rule=gauss_legendre(((-1.0, 1.0), (-1.0, 1.0)), 4),
        conserved=("u",),
        total=1.0,
        degree=1,
        dtype=DTYPE,
    )
    assert getattr(cage, READOUT_INDEPENDENT_ATTR) is False
    with pytest.raises(ReadoutDependentError, match="IntegralConservationField"):
        requires_readout_independent(cage)
    with pytest.raises(ReadoutDependentError):
        _readout_size(cage)


def test_affine_cage_recurses_through_base() -> None:
    base = _make_one_layer()
    cage = HardBoundaryField(
        base=base,
        distance_fn=lambda coords: torch.ones(coords.shape[0], dtype=DTYPE),
        boundary_value_fn=lambda coords: {
            "u": torch.zeros(coords.shape[0], dtype=DTYPE)
        },
    )
    assert getattr(cage, READOUT_INDEPENDENT_ATTR) is True
    requires_readout_independent(cage)


def test_constrained_cage_over_spectral_survives_readout_sweep() -> None:
    base = _make_spectral(seed=1)
    conditions = (
        HardCondition("u", 1, dirichlet(0.0), 0.0),
        HardCondition("u", 1, dirichlet(1.0), 0.0),
    )
    cage = ConstrainedExpressionField(
        base=base,
        conditions=conditions,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        certify=False,
    )
    assert getattr(cage, READOUT_INDEPENDENT_ATTR) is True
    coords = _coords_for(base)
    state = cage(coords)
    _ = state.ops.value(state, "u")
    _ = state.ops.gradient(state, "u")
    _mutate_readout(cage)
    stale = state.ops.value(state, "u")
    fresh = ConstrainedExpressionField(
        base=_make_spectral(seed=1),
        conditions=conditions,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        certify=False,
    )
    _mutate_readout(fresh)
    fresh_state = fresh(coords)
    assert torch.equal(stale, fresh_state.ops.value(fresh_state, "u"))


def test_streamfunction_cage_declares_with_one_layer_base() -> None:
    torch.manual_seed(0)
    base = OneLayerVectorField(
        coordinate_spec=_space_spec(),
        components=ComponentSpec(("psi",)),
        hidden=8,
        base="tanh",
        dtype=DTYPE,
    )
    cage = StreamfunctionField(base=base, psi="psi")
    assert getattr(cage, READOUT_INDEPENDENT_ATTR) is True


def test_undeclared_object_is_refused() -> None:
    class _Bare:
        pass

    with pytest.raises(ReadoutDependentError, match="_Bare"):
        requires_readout_independent(_Bare())
