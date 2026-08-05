# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch <-> jax parity for the XPINN interface ops.

The geometry is shared numpy, so both backends are handed *the same* interface
points and the comparison is like for like: any difference is in the residual
ops, not in where the seam was sampled.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get_activation  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn._core.interface import (  # noqa: E402
    Interface,
    InterfaceSpec,
    interface_points,
    split_by_interface,
)
from omnibias.pinn.jax.fields import OneLayerVectorField as JaxField  # noqa: E402
from omnibias.pinn.jax.losses import (  # noqa: E402
    interface_loss as jax_loss,
)
from omnibias.pinn.jax.losses import (  # noqa: E402
    interface_residual as jax_residual,
)
from omnibias.pinn.jax.losses import (  # noqa: E402
    normal_derivative as jax_normal,
)
from omnibias.pinn.torch.fields import OneLayerVectorField as TorchField  # noqa: E402
from omnibias.pinn.torch.losses import (  # noqa: E402
    interface_loss as torch_loss,
)
from omnibias.pinn.torch.losses import (  # noqa: E402
    interface_residual as torch_residual,
)
from omnibias.pinn.torch.losses import (  # noqa: E402
    normal_derivative as torch_normal,
)

TOL = dict(rtol=1e-12, atol=1e-12)
CS = CoordinateSpec(("x", "y"))
COMPS = ComponentSpec(("u", "v"))
IFACE = Interface(normal=(1.0, 2.0), offset=0.25, label="seam")
BOX = ((-1.0, 1.0), (-1.0, 1.0))


def _pair(params, activation: str):
    """One parameter set, two fields -- the only way parity means anything."""
    torch_field = TorchField(
        coordinate_spec=CS,
        components=COMPS,
        hidden=params["H"],
        base=activation,
        dtype=torch.float64,
    )
    with torch.no_grad():
        torch_field.W.weight.copy_(torch.from_numpy(params["W"]))
        torch_field.W.bias.copy_(torch.from_numpy(params["beta"]))
        torch_field.c.weight.copy_(torch.from_numpy(params["c"]))
        torch_field.c.bias.copy_(torch.from_numpy(params["b"]))
    jax_field = JaxField(
        coordinate_spec=CS,
        components=COMPS,
        spec=jax_get_activation(activation),
        W=jnp.asarray(params["W"]),
        beta=jnp.asarray(params["beta"]),
        c=jnp.asarray(params["c"]),
        b=jnp.asarray(params["b"]),
        hidden=params["H"],
    )
    return torch_field, jax_field


@pytest.fixture
def seam_params():
    """Two independent parameter sets on a 2-D, 2-component field."""
    rng = np.random.default_rng(17)
    H, D, C = 6, 2, 2
    return [
        dict(
            W=rng.normal(scale=0.7, size=(H, D)),
            beta=rng.normal(scale=0.3, size=(H,)),
            c=rng.normal(scale=0.7, size=(C, H)),
            b=rng.normal(scale=0.2, size=(C,)),
            H=H,
        )
        for _ in range(2)
    ]


@pytest.fixture
def seam_points():
    return interface_points(IFACE, BOX, n_points=9, method="grid")


def test_the_two_backends_see_the_same_seam(seam_points) -> None:
    """Shared numpy geometry: not approximately the same points, the same ones."""
    t = torch.as_tensor(seam_points, dtype=torch.float64).numpy()
    j = np.asarray(jnp.asarray(seam_points))
    np.testing.assert_array_equal(t, j)
    assert float(np.abs(IFACE.signed_distance(t)).max()) < 1e-14


def test_normal_derivative_parity(riccati, seam_params, seam_points) -> None:
    tf, jf = _pair(seam_params[0], riccati)
    got_t = torch_normal(
        tf(torch.as_tensor(seam_points, dtype=torch.float64)), "u", normal=IFACE
    )
    got_j = jax_normal(jf(jnp.asarray(seam_points)), "u", normal=IFACE)
    np.testing.assert_allclose(got_t.detach().numpy(), np.asarray(got_j), **TOL)


@pytest.mark.parametrize("conductivity", [(1.0, 1.0), (3.0, 0.5)])
def test_interface_residual_and_loss_parity(
    riccati, seam_params, seam_points, conductivity
) -> None:
    tf_a, jf_a = _pair(seam_params[0], riccati)
    tf_b, jf_b = _pair(seam_params[1], riccati)
    spec = InterfaceSpec(IFACE, conductivity=conductivity, weights=(2.0, 0.25))
    xt = torch.as_tensor(seam_points, dtype=torch.float64)
    xj = jnp.asarray(seam_points)

    rt = torch.as_tensor(np.linspace(-1.0, 1.0, seam_points.shape[0]))
    rj = jnp.asarray(np.linspace(-1.0, 1.0, seam_points.shape[0]))
    out_t = torch_residual(
        tf_a(xt), tf_b(xt), spec, residuals=(rt, torch.zeros_like(rt))
    )
    out_j = jax_residual(
        jf_a(xj), jf_b(xj), spec, residuals=(rj, jnp.zeros_like(rj))
    )

    np.testing.assert_allclose(
        out_t.value_jump.detach().numpy(), np.asarray(out_j.value_jump), **TOL
    )
    np.testing.assert_allclose(
        out_t.flux_jump.detach().numpy(), np.asarray(out_j.flux_jump), **TOL
    )
    np.testing.assert_allclose(
        out_t.residual_jump.detach().numpy(), np.asarray(out_j.residual_jump), **TOL
    )
    for key, value in out_t.diag.items():
        assert float(np.asarray(out_j.diag[key])) == pytest.approx(
            value, rel=1e-12, abs=1e-12
        )

    loss_t = torch_loss(out_t, weights=spec.weights, residual_weight=0.5)
    loss_j = jax_loss(out_j, weights=spec.weights, residual_weight=0.5)
    assert float(loss_j) == pytest.approx(float(loss_t.detach()), rel=1e-12, abs=1e-12)


def test_routing_collocation_agrees_with_the_shared_geometry() -> None:
    """``split_by_interface`` is numpy, so the patches are identical by design."""
    x = np.random.default_rng(5).uniform(-1.0, 1.0, size=(64, 2))
    plus, minus = split_by_interface(IFACE, x)
    t_side = (
        (torch.as_tensor(x, dtype=torch.float64) @ torch.as_tensor(IFACE.unit_normal))
        - IFACE.offset
    ) >= 0
    j_side = ((jnp.asarray(x) @ jnp.asarray(IFACE.unit_normal)) - IFACE.offset) >= 0
    np.testing.assert_array_equal(t_side.numpy(), np.asarray(j_side))
    assert plus.shape[0] == int(t_side.sum()) == 64 - minus.shape[0]
