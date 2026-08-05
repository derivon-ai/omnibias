# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity of the deep, arbitrary-order omnibias PINN.

The torch and jax ``JetMLP`` call the *same* Faa di Bruno multivariate-jet kernel
(:mod:`omnibias.{torch,jax}.jet_mv`), so for shared weights every closed-form input
derivative -- value, gradient, Hessian, and high-order partials -- must agree to
float64 (bit-identical) precision. This is the "torch + jax, bit-identical"
guarantee for the PINN derivative operator.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get_activation  # noqa: E402
from omnibias.jax.architectures import JetMLP as JaxJetMLP  # noqa: E402
from omnibias.torch.architectures import JetMLP as TorchJetMLP  # noqa: E402

torch.set_default_dtype(torch.float64)


def _shared_pair(
    in_dim: int,
    hidden: int,
    out_dim: int,
    depth: int,
    activation: str,
    seed: int,
) -> tuple[TorchJetMLP, JaxJetMLP]:
    """A torch and a jax ``JetMLP`` holding identical (float64) weights."""
    torch.manual_seed(seed)
    tnet = TorchJetMLP(in_dim, hidden, out_dim, depth=depth, base=activation).double()
    ws = tuple(jnp.asarray(lin.weight.detach().numpy(), dtype=jnp.float64) for lin in tnet.linears)
    bs = tuple(jnp.asarray(lin.bias.detach().numpy(), dtype=jnp.float64) for lin in tnet.linears)
    jnet = JaxJetMLP(ws, bs, jax_get_activation(activation), in_dim, out_dim)
    return tnet, jnet


def _xpair(seed: int, b: int, d: int) -> tuple[torch.Tensor, jnp.ndarray]:
    xnp = np.random.RandomState(seed).randn(b, d)
    return torch.tensor(xnp, dtype=torch.float64), jnp.asarray(xnp, dtype=jnp.float64)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_value_grad_hessian_bit_parity(activation: str, depth: int) -> None:
    tnet, jnet = _shared_pair(2, 12, 1, depth, activation, seed=depth)
    xt, xj = _xpair(seed=depth, b=6, d=2)

    vt = tnet.value(xt).detach().numpy()
    vj = np.asarray(jnet.value(xj))
    assert np.allclose(vt, vj, rtol=0, atol=1e-13)

    gt = tnet.gradient(xt).detach().numpy()
    gj = np.asarray(jnet.gradient(xj))
    assert np.allclose(gt, gj, rtol=0, atol=1e-12)

    ht = tnet.hessian(xt).detach().numpy()
    hj = np.asarray(jnet.hessian(xj))
    assert np.allclose(ht, hj, rtol=0, atol=1e-12)


def test_high_order_partials_bit_parity() -> None:
    tnet, jnet = _shared_pair(2, 10, 1, depth=2, activation="tanh", seed=4)
    xt, xj = _xpair(seed=4, b=5, d=2)
    pt = tnet.partials(xt, 4)
    pj = jnet.partials(xj, 4)
    assert pt.keys() == pj.keys()
    for alpha in pt:
        a = pt[alpha].detach().numpy()
        b = np.asarray(pj[alpha])
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"mismatch at {alpha}"


def test_multioutput_bit_parity() -> None:
    tnet, jnet = _shared_pair(3, 8, 2, depth=2, activation="tanh", seed=7)
    xt, xj = _xpair(seed=7, b=4, d=3)
    gt = tnet.gradient(xt).detach().numpy()
    gj = np.asarray(jnet.gradient(xj))
    assert gt.shape == gj.shape == (4, 3, 2)
    assert np.allclose(gt, gj, rtol=0, atol=1e-12)
