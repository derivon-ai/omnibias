# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for the CCF self-similar residual."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get_activation  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax import equations as jeq  # noqa: E402
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JaxOLF  # noqa: E402
from omnibias.pinn.torch import equations as teq  # noqa: E402
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TorchOLF  # noqa: E402

torch.set_default_dtype(torch.float64)


def _shared_fields(hidden: int, seed: int):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(hidden, 1)) * 0.7
    beta = rng.normal(size=(hidden,)) * 0.3
    c = rng.normal(size=(1, hidden)) * 0.5
    b = rng.normal(size=(1,)) * 0.2
    cspec = CoordinateSpec(axes=("y",), periodicity=(True,), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    jax_field = JaxOLF(
        coordinate_spec=cspec, components=mspec, spec=jax_get_activation("tanh"),
        W=jnp.asarray(W), beta=jnp.asarray(beta), c=jnp.asarray(c), b=jnp.asarray(b),
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


@pytest.mark.parametrize("form", ["transport", "flux"])
def test_ccf_jax_torch_parity(form: str) -> None:
    jax_field, torch_field = _shared_fields(hidden=6, seed=7)
    y = (-np.pi + 2 * np.pi * np.arange(64) / 64).reshape(-1, 1)
    jout = jeq.cordoba_cordoba_fontelos(jax_field(jnp.asarray(y)), lam=0.55, form=form)
    tout = teq.cordoba_cordoba_fontelos(torch_field(torch.tensor(y)), lam=0.55, form=form)
    np.testing.assert_allclose(
        np.asarray(jout.residual), tout.residual.detach().numpy(),
        rtol=1e-9, atol=1e-12,
    )
