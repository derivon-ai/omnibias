# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity: the conservation cages on torch vs their JAX twins.

Both cages are pure algebra on top of the shared closed-form tower -- one
global scalar for :class:`IntegralConservationField`, one signed sum of
potential derivatives for :class:`FluxFormField` -- so with identical numpy
parameters the two backends must agree to float64 round-off. The index
bookkeeping is literally shared (``omnibias.pinn._core.fluxform``), which the
last test asserts as object identity rather than by value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.jax.architectures.pinn import JetMLP as JaxJetMLP
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.cage import (
    make_flux_form_field,
    make_integral_conservation_field,
)
from omnibias.pinn.jax.fields.jet_mlp import JetMLPVectorField as JaxJetField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage import FluxFormField, IntegralConservationField
from omnibias.pinn.torch.fields import JetMLPVectorField as TorchJetField

JET_ORDER = 3


def _params(in_dim: int, out_dim: int, *, hidden: int = 8, seed: int = 5):
    rng = np.random.default_rng(seed)
    dims = [hidden, hidden, out_dim]
    weights, biases = [], []
    prev = in_dim
    for d in dims:
        weights.append(rng.normal(scale=0.7 / np.sqrt(prev), size=(d, prev)))
        biases.append(rng.normal(scale=0.3, size=(d,)))
        prev = d
    return weights, biases


def _torch_base(cspec, mspec, weights, biases):
    field = TorchJetField(
        coordinate_spec=cspec,
        components=mspec,
        hidden=int(weights[0].shape[0]),
        depth=len(weights) - 1,
        base="tanh",
        jet_order=JET_ORDER,
    )
    with torch.no_grad():
        for lin, w, b in zip(field.net.linears, weights, biases, strict=True):
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
    return field


def _jax_base(cspec, mspec, weights, biases):
    net = JaxJetMLP(
        weights=tuple(jnp.asarray(w) for w in weights),
        biases=tuple(jnp.asarray(b) for b in biases),
        spec=jax_get_activation("tanh"),
        in_dim=cspec.ndim,
        out_dim=mspec.n_components,
    )
    return JaxJetField(
        coordinate_spec=cspec, components=mspec, net=net, jet_order=JET_ORDER
    )


def _allclose(t, j, *, atol: float = 1e-12) -> bool:
    t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return np.allclose(t_np, np.asarray(j), rtol=atol, atol=atol)


# ----------------------- integral conservation -------------------------


@pytest.mark.parametrize(("degree", "total"), [(1, 2.0), (2, 1.0), (3, 4.0)])
def test_integral_cage_parity(degree: int, total: float) -> None:
    bounds = ((-3.0, 5.0),)
    cspec = CoordinateSpec(("x",), domain=bounds)
    mspec = ComponentSpec(("psi_re", "psi_im", "p"))
    weights, biases = _params(1, 3)
    rule = gauss_legendre(bounds, 96)
    coords = np.linspace(-2.0, 4.0, 9).reshape(-1, 1)

    torch_cage = IntegralConservationField(
        base=_torch_base(cspec, mspec, weights, biases),
        rule=rule,
        conserved=("psi_re", "psi_im"),
        total=total,
        degree=degree,
        dtype=torch.float64,
    )
    jax_cage = make_integral_conservation_field(
        base=_jax_base(cspec, mspec, weights, biases),
        rule=rule,
        conserved=("psi_re", "psi_im"),
        total=total,
        degree=degree,
    )
    ts = torch_cage(torch.from_numpy(coords))
    js = jax_cage(jnp.asarray(coords))

    assert _allclose(
        ts.extra["_conservation_scale"], js.extra["_conservation_scale"]
    ), "the rescaling factor itself must agree"
    assert _allclose(torch_cage.integral(ts), jax_cage.integral(js))
    for name in ("psi_re", "psi_im", "p"):
        assert _allclose(tops.value(ts, name), jops.value(js, name)), f"value {name!r}"
        for order in (1, 2):
            assert _allclose(
                tops.derivative(ts, name, axis=0, order=order),
                jops.derivative(js, name, axis=0, order=order),
            ), f"d^{order} {name!r}"


# --------------------------- flux form ---------------------------------


@pytest.mark.parametrize("n_axes", [2, 3])
def test_flux_form_parity(n_axes: int) -> None:
    axes = ("t", "x", "y")[:n_axes]
    potentials = tuple(f"A{i}" for i in range(n_axes * (n_axes - 1) // 2))
    fluxes = ("rho", "fx", "fy")[:n_axes]
    cspec = CoordinateSpec(axes)
    mspec = ComponentSpec(potentials)
    weights, biases = _params(n_axes, len(potentials), seed=n_axes)
    coords = np.random.default_rng(n_axes).normal(size=(7, n_axes))

    torch_flux = FluxFormField(
        base=_torch_base(cspec, mspec, weights, biases),
        potential_names=potentials,
        flux_names=fluxes,
    )
    jax_flux = make_flux_form_field(
        base=_jax_base(cspec, mspec, weights, biases),
        potential_names=potentials,
        flux_names=fluxes,
    )
    ts = torch_flux(torch.from_numpy(coords))
    js = jax_flux(jnp.asarray(coords))

    for name in fluxes:
        assert _allclose(tops.value(ts, name), jops.value(js, name)), f"value {name!r}"
        for axis in range(n_axes):
            assert _allclose(
                tops.derivative(ts, name, axis=axis),
                jops.derivative(js, name, axis=axis),
            ), f"d_{axis} {name!r}"

    t_div = sum(tops.derivative(ts, n, axis=i) for i, n in enumerate(fluxes))
    j_div = sum(jops.derivative(js, n, axis=i) for i, n in enumerate(fluxes))
    assert float(t_div.abs().max().detach()) < 1e-11
    assert float(jnp.abs(j_div).max()) < 1e-11


def test_index_bookkeeping_is_one_shared_object() -> None:
    """Not "the two backends agree" -- they import the same function."""
    from omnibias.pinn._core.fluxform import antisymmetric_pairs as core_pairs
    from omnibias.pinn.jax.cage import antisymmetric_pairs as jax_pairs
    from omnibias.pinn.torch.cage import antisymmetric_pairs as torch_pairs

    assert torch_pairs is core_pairs
    assert jax_pairs is core_pairs
