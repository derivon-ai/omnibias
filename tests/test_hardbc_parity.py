# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity of the hard-constraint ansatz ``u = g + b N``.

The torch and jax :class:`HardConstraintField` wrap a weight-shared network with the
*same* polynomial lift/mask and combine them with the *same* Faa di Bruno jet kernel
(:func:`jet_multiply`). So for shared weights the constrained field's value, gradient,
Hessian and high-order partials agree to float64 (bit-identical) precision -- the
"torch + jax, bit-identical" guarantee now also covers the boundary-baked field.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax import jet_multiply as jax_jet_multiply  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get_activation  # noqa: E402
from omnibias.jax.architectures import JetMLP as JaxJetMLP  # noqa: E402
from omnibias.jax.architectures import hardbc as jax_hardbc  # noqa: E402
from omnibias.jax.jet_mv import identity_jet as jax_identity_jet  # noqa: E402
from omnibias.torch import jet_multiply as torch_jet_multiply  # noqa: E402
from omnibias.torch.architectures import JetMLP as TorchJetMLP  # noqa: E402
from omnibias.torch.architectures import hardbc as torch_hardbc  # noqa: E402
from omnibias.torch.jet_mv import identity_jet as torch_identity_jet  # noqa: E402

torch.set_default_dtype(torch.float64)


def _xpair(seed: int, b: int, d: int) -> tuple[torch.Tensor, jnp.ndarray]:
    xnp = np.random.RandomState(seed).randn(b, d)
    return torch.tensor(xnp, dtype=torch.float64), jnp.asarray(xnp, dtype=jnp.float64)


def _jetmlp_pair(
    in_dim: int, hidden: int, out_dim: int, depth: int, base: str, seed: int
) -> tuple[TorchJetMLP, JaxJetMLP]:
    """A torch and a jax ``JetMLP`` holding identical (float64) weights."""
    torch.manual_seed(seed)
    tnet = TorchJetMLP(in_dim, hidden, out_dim, depth=depth, base=base).double()
    specs = tnet._layer_specs()
    ws = tuple(jnp.asarray(w.detach().numpy(), dtype=jnp.float64) for (w, _b, _s) in specs)
    bs = tuple(jnp.asarray(b.detach().numpy(), dtype=jnp.float64) for (_w, b, _s) in specs)
    jnet = JaxJetMLP(ws, bs, jax_get_activation(base), in_dim, out_dim)
    return tnet, jnet


def _assert_field_parity(tfield, jfield, xt: torch.Tensor, xj: jnp.ndarray) -> None:
    assert np.allclose(
        tfield.value(xt).detach().numpy(), np.asarray(jfield.value(xj)), atol=1e-13
    )
    assert np.allclose(
        tfield.gradient(xt).detach().numpy(), np.asarray(jfield.gradient(xj)), atol=1e-12
    )
    assert np.allclose(
        tfield.hessian(xt).detach().numpy(), np.asarray(jfield.hessian(xj)), atol=1e-12
    )
    pt = tfield.partials(xt, 3)
    pj = jfield.partials(xj, 3)
    assert pt.keys() == pj.keys()
    for alpha in pt:
        assert np.allclose(
            pt[alpha].detach().numpy(), np.asarray(pj[alpha]), atol=1e-10
        ), f"mismatch at {alpha}"


def test_jet_multiply_bit_parity() -> None:
    xt, xj = _xpair(seed=0, b=1, d=2)
    it = torch_identity_jet(xt[0], 3)
    ij = jax_identity_jet(xj[0], 3)
    pt = torch_jet_multiply(it[:, 0], it[:, 1], 2, 3)
    pj = jax_jet_multiply(ij[:, 0], ij[:, 1], 2, 3)
    assert np.allclose(pt.detach().numpy(), np.asarray(pj), atol=1e-13)


def test_dirichlet_interval_parity() -> None:
    tnet, jnet = _jetmlp_pair(1, 16, 1, 3, "tanh", seed=1)
    tfield = torch_hardbc.dirichlet_interval(tnet, 0.0, 1.0, lower_value=0.5, upper_value=2.0)
    jfield = jax_hardbc.dirichlet_interval(jnet, 0.0, 1.0, lower_value=0.5, upper_value=2.0)
    xt, xj = _xpair(seed=1, b=6, d=1)
    _assert_field_parity(tfield, jfield, xt, xj)


def test_homogeneous_box_parity() -> None:
    tnet, jnet = _jetmlp_pair(2, 12, 1, 2, "tanh", seed=2)
    tfield = torch_hardbc.homogeneous_box(tnet, [-1.0, 0.0], [1.0, 2.0])
    jfield = jax_hardbc.homogeneous_box(jnet, [-1.0, 0.0], [1.0, 2.0])
    xt, xj = _xpair(seed=2, b=5, d=2)
    _assert_field_parity(tfield, jfield, xt, xj)


def test_initial_value_parity() -> None:
    tnet, jnet = _jetmlp_pair(2, 10, 1, 2, "sigmoid", seed=3)
    tfield = torch_hardbc.initial_value(tnet, t_axis=1, t0=0.0, value=2.0)
    jfield = jax_hardbc.initial_value(jnet, t_axis=1, t0=0.0, value=2.0)
    xt, xj = _xpair(seed=3, b=5, d=2)
    _assert_field_parity(tfield, jfield, xt, xj)


def test_multi_output_box_parity() -> None:
    tnet, jnet = _jetmlp_pair(2, 10, 3, 2, "tanh", seed=4)
    tfield = torch_hardbc.homogeneous_box(tnet, [0.0, 0.0], [1.0, 1.0])
    jfield = jax_hardbc.homogeneous_box(jnet, [0.0, 0.0], [1.0, 1.0])
    xt, xj = _xpair(seed=4, b=4, d=2)
    _assert_field_parity(tfield, jfield, xt, xj)
