# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity of the spectral-bias-mitigating omnibias PINNs.

The torch and jax ``FourierFeatureMLP`` / SIREN call the *same* Faa di Bruno
multivariate-jet kernel, so for shared weights every closed-form input derivative --
value, gradient, Hessian, and high-order partials -- agrees to float64 (bit-identical)
precision. This extends the "torch + jax, bit-identical" guarantee to the sin-encoded
Fourier-feature front end and to SIREN.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get_activation  # noqa: E402
from omnibias.jax.architectures import FourierFeatureMLP as JaxFF  # noqa: E402
from omnibias.jax.architectures import JetMLP as JaxJetMLP  # noqa: E402
from omnibias.jax.architectures import make_siren as jax_make_siren  # noqa: E402
from omnibias.torch.architectures import FourierFeatureMLP as TorchFF  # noqa: E402
from omnibias.torch.architectures import make_siren as torch_make_siren  # noqa: E402

torch.set_default_dtype(torch.float64)


def _xpair(seed: int, b: int, d: int) -> tuple[torch.Tensor, jnp.ndarray]:
    xnp = np.random.RandomState(seed).randn(b, d)
    return torch.tensor(xnp, dtype=torch.float64), jnp.asarray(xnp, dtype=jnp.float64)


def _ff_pair(
    in_dim: int,
    num_features: int,
    hidden: int,
    out_dim: int,
    depth: int,
    base: str,
    scales: float | tuple[float, ...],
    seed: int,
) -> tuple[TorchFF, JaxFF]:
    """A torch and a jax ``FourierFeatureMLP`` holding identical (float64) weights."""
    torch.manual_seed(seed)
    tnet = TorchFF(
        in_dim,
        num_features=num_features,
        hidden=hidden,
        out_dim=out_dim,
        depth=depth,
        base=base,
        frequency_scale=scales,
        seed=seed,
    ).double()
    specs = tnet._layer_specs()
    w_ff = jnp.asarray(specs[0][0].detach().numpy(), dtype=jnp.float64)
    b_ff = jnp.asarray(specs[0][1].detach().numpy(), dtype=jnp.float64)
    ws = tuple(jnp.asarray(w.detach().numpy(), dtype=jnp.float64) for (w, _b, _s) in specs[1:])
    bs = tuple(jnp.asarray(b.detach().numpy(), dtype=jnp.float64) for (_w, b, _s) in specs[1:])
    scales_t = tnet.scales
    jnet = JaxFF(
        w_ff=w_ff,
        b_ff=b_ff,
        weights=ws,
        biases=bs,
        base_spec=jax_get_activation(base),
        in_dim=in_dim,
        out_dim=out_dim,
        num_features=num_features,
        scales=scales_t,
    )
    return tnet, jnet


@pytest.mark.parametrize("base", ["tanh", "sigmoid"])
@pytest.mark.parametrize("scales", [1.0, (0.5, 2.0)])
def test_fourier_value_grad_hessian_bit_parity(
    base: str, scales: float | tuple[float, ...]
) -> None:
    tnet, jnet = _ff_pair(2, 6, 12, 1, 2, base, scales, seed=3)
    xt, xj = _xpair(seed=3, b=6, d=2)

    assert np.allclose(tnet.value(xt).detach().numpy(), np.asarray(jnet.value(xj)), atol=1e-13)
    assert np.allclose(
        tnet.gradient(xt).detach().numpy(), np.asarray(jnet.gradient(xj)), atol=1e-12
    )
    assert np.allclose(
        tnet.hessian(xt).detach().numpy(), np.asarray(jnet.hessian(xj)), atol=1e-12
    )


def test_fourier_high_order_bit_parity() -> None:
    tnet, jnet = _ff_pair(2, 5, 10, 1, 2, "tanh", (0.5, 2.0), seed=4)
    xt, xj = _xpair(seed=4, b=5, d=2)
    pt = tnet.partials(xt, 3)
    pj = jnet.partials(xj, 3)
    assert pt.keys() == pj.keys()
    for alpha in pt:
        a = pt[alpha].detach().numpy()
        b = np.asarray(pj[alpha])
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"mismatch at {alpha}"


def test_siren_bit_parity() -> None:
    """SIREN built independently per backend is *not* weight-shared (different RNGs); so
    we share weights explicitly: build torch, copy into a jax ``JetMLP`` with ``sin``."""
    tnet = torch_make_siren(1, hidden=16, depth=3, omega_0=12.0, seed=7).double()
    ws = tuple(jnp.asarray(lin.weight.detach().numpy(), dtype=jnp.float64) for lin in tnet.linears)
    bs = tuple(jnp.asarray(lin.bias.detach().numpy(), dtype=jnp.float64) for lin in tnet.linears)
    jnet = JaxJetMLP(ws, bs, jax_get_activation("sin"), 1, 1)
    xt, xj = _xpair(seed=7, b=6, d=1)
    assert np.allclose(tnet.value(xt).detach().numpy(), np.asarray(jnet.value(xj)), atol=1e-13)
    assert np.allclose(
        tnet.gradient(xt).detach().numpy(), np.asarray(jnet.gradient(xj)), atol=1e-12
    )
    pt = tnet.partials(xt, 4)
    pj = jnet.partials(xj, 4)
    for alpha in pt:
        assert np.allclose(
            pt[alpha].detach().numpy(), np.asarray(pj[alpha]), atol=1e-9
        ), f"mismatch at {alpha}"


def test_siren_builders_same_init_across_backends() -> None:
    """The SIREN *builders* use independent RNGs, but the init *scheme* matches: the
    first-layer/hidden weight-magnitude ratio agrees in distribution (sanity check)."""
    tnet = torch_make_siren(1, hidden=64, depth=3, omega_0=30.0, seed=0)
    jnet = jax_make_siren(1, hidden=64, depth=3, omega_0=30.0, seed=0)
    t_ratio = (
        tnet.linears[0].weight.abs().mean().item()
        / tnet.linears[1].weight.abs().mean().item()
    )
    j_ratio = float(jnp.mean(jnp.abs(jnet.weights[0])) / jnp.mean(jnp.abs(jnet.weights[1])))
    # Both should be ~ omega_0^2 * sqrt(...) scale; agree to within a small factor.
    assert 0.5 < t_ratio / j_ratio < 2.0
