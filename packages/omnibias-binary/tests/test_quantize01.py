# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The {0,1} Heaviside twin: forward, sigmoid-beta backward, and tanh conjugacy.

``binarize01(z, beta)`` is the codomain twin of ``binarize`` and is affinely
conjugate to it: ``binarize01(z, beta) == (binarize(z, beta / 2) + 1) / 2`` in both
the forward and the backward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.binary.jax import ops as jq  # noqa: E402
from omnibias.binary.torch import ops as tq  # noqa: E402

BETA = 6.0
Z = np.array([-2.0, -0.6, -0.1, 0.0, 0.1, 0.6, 2.0], dtype=np.float64)


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def test_forward_is_zero_one() -> None:
    out = _np(tq.binarize01(torch.as_tensor(Z), beta=BETA))
    assert np.array_equal(out, np.where(Z >= 0, 1.0, 0.0))
    assert out[Z == 0.0][0] == 1.0  # H(0) = 1


def test_backward_sigmoid_formula() -> None:
    z = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    tq.binarize01(z, beta=BETA).sum().backward()
    s = 1.0 / (1.0 + np.exp(-BETA * Z))
    assert np.allclose(_np(z.grad), BETA * s * (1.0 - s), rtol=1e-12, atol=1e-14)


def test_conjugacy_with_tanh_binarize() -> None:
    # Forward + backward equality binarize01(z, b) == (binarize(z, b/2) + 1) / 2.
    z1 = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    z2 = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    out01 = tq.binarize01(z1, beta=BETA)
    out_conj = (tq.binarize(z2, beta=BETA / 2.0) + 1.0) / 2.0
    assert np.allclose(_np(out01), _np(out_conj))
    out01.sum().backward()
    out_conj.sum().backward()
    assert np.allclose(_np(z1.grad), _np(z2.grad), rtol=1e-12, atol=1e-14)


def test_cross_backend_parity() -> None:
    z = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    out_t = tq.binarize01(z, beta=BETA)
    out_t.sum().backward()
    jout = jq.binarize01(jnp.asarray(Z), BETA)
    jgrad = jax.grad(lambda zz: jnp.sum(jq.binarize01(zz, BETA)))(jnp.asarray(Z))
    assert np.allclose(_np(out_t), _np(jout), rtol=1e-9, atol=1e-11)
    assert np.allclose(_np(z.grad), _np(jgrad), rtol=1e-9, atol=1e-11)


def test_learnable_beta_parity() -> None:
    z = torch.tensor(Z, dtype=torch.float64)
    beta = torch.tensor(BETA, dtype=torch.float64, requires_grad=True)
    tq.binarize01(z, beta).sum().backward()
    tgrad = float(beta.grad)
    jgrad = float(jax.grad(lambda b: jnp.sum(jq.binarize01(jnp.asarray(Z), b)))(BETA))
    assert jgrad == pytest.approx(tgrad, rel=1e-9, abs=1e-11)
