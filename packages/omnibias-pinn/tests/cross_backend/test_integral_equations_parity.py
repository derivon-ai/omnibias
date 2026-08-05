# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for the Fredholm / Volterra residuals.

The nonlocal residuals do more per evaluation than the local ones -- a second
field evaluation at the quadrature nodes, and for Volterra a per-point pullback
of the integration domain. Parity therefore checks more machinery than usual,
which is exactly why it is worth checking.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("omnibias.measure")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get_activation  # noqa: E402
from omnibias.measure._core.measure import lebesgue  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax import equations as jeq  # noqa: E402
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JaxOLF  # noqa: E402
from omnibias.pinn.torch import equations as teq  # noqa: E402
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TorchOLF  # noqa: E402

torch.set_default_dtype(torch.float64)


def _shared_fields(hidden: int, seed: int, axes=("x",), time_axis=None):
    rng = np.random.default_rng(seed)
    dim = len(axes)
    W = rng.normal(size=(hidden, dim)) * 0.7
    beta = rng.normal(size=(hidden,)) * 0.3
    c = rng.normal(size=(1, hidden)) * 0.5
    b = rng.normal(size=(1,)) * 0.2
    cspec = CoordinateSpec(
        axes=axes, periodicity=(False,) * dim, time_axis=time_axis
    )
    mspec = ComponentSpec(names=("u",), groups={})
    jax_field = JaxOLF(
        coordinate_spec=cspec,
        components=mspec,
        spec=jax_get_activation("tanh"),
        W=jnp.asarray(W),
        beta=jnp.asarray(beta),
        c=jnp.asarray(c),
        b=jnp.asarray(b),
        hidden=hidden,
    )
    torch_field = TorchOLF(
        coordinate_spec=cspec, components=mspec, hidden=hidden, base="tanh"
    )
    with torch.no_grad():
        torch_field.W.weight.copy_(torch.tensor(W))
        torch_field.W.bias.copy_(torch.tensor(beta))
        torch_field.c.weight.copy_(torch.tensor(c))
        torch_field.c.bias.copy_(torch.tensor(b))
    return jax_field, torch_field


def test_fredholm_jax_torch_parity() -> None:
    jax_field, torch_field = _shared_fields(hidden=6, seed=11)
    mu = lebesgue([(0.0, 1.0)], 20)
    x = np.linspace(0.0, 1.0, 13).reshape(-1, 1)

    jout = jeq.fredholm(
        jax_field(jnp.asarray(x)),
        kernel=lambda a, t: jnp.exp(-((a[:, :1] - t[:, 0][None, :]) ** 2)),
        measure=mu,
        lam=0.6,
        source=lambda state: jnp.cos(state.coords[:, 0]),
    )
    tout = teq.fredholm(
        torch_field(torch.tensor(x)),
        kernel=lambda a, t: torch.exp(-((a[:, :1] - t[:, 0][None, :]) ** 2)),
        measure=mu,
        lam=0.6,
        source=lambda state: torch.cos(state.coords[:, 0]),
    )
    np.testing.assert_allclose(
        np.asarray(jout.residual), tout.residual.detach().numpy(),
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jout.integral), tout.integral.detach().numpy(),
        rtol=1e-9, atol=1e-12,
    )


def test_volterra_jax_torch_parity() -> None:
    jax_field, torch_field = _shared_fields(hidden=6, seed=13)
    mu = lebesgue([(0.0, 1.0)], 16)
    x = np.linspace(0.05, 1.0, 11).reshape(-1, 1)

    jout = jeq.volterra(
        jax_field(jnp.asarray(x)),
        kernel=lambda a, t: jnp.exp(-(a - t).squeeze(-1)),
        measure=mu,
        lam=0.5,
        source=lambda state: jnp.ones(state.coords.shape[0]),
    )
    tout = teq.volterra(
        torch_field(torch.tensor(x)),
        kernel=lambda a, t: torch.exp(-(a - t).squeeze(-1)),
        measure=mu,
        lam=0.5,
        source=lambda state: torch.ones(state.coords.shape[0]),
    )
    np.testing.assert_allclose(
        np.asarray(jout.residual), tout.residual.detach().numpy(),
        rtol=1e-9, atol=1e-12,
    )


def test_volterra_parity_holds_on_a_frozen_spatial_axis() -> None:
    """The space-time memory term: parity must survive the per-point pullback."""
    jax_field, torch_field = _shared_fields(
        hidden=5, seed=17, axes=("x", "t"), time_axis="t"
    )
    mu = lebesgue([(0.0, 1.0)], 12)
    pts = np.array([[0.3, 0.4], [1.0, 0.9], [-0.5, 1.6]])

    jout = jeq.volterra(
        jax_field(jnp.asarray(pts)),
        kernel=lambda a, t: jnp.ones(a.shape[:2]),
        measure=mu,
        lam=1.0,
    )
    tout = teq.volterra(
        torch_field(torch.tensor(pts)),
        kernel=lambda a, t: torch.ones(a.shape[:2], dtype=a.dtype),
        measure=mu,
        lam=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(jout.integral), tout.integral.detach().numpy(),
        rtol=1e-9, atol=1e-12,
    )
