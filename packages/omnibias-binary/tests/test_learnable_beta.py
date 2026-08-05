# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learnable-beta parity: JAX quantizers now differentiate beta like torch."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from omnibias.binary.jax import ops as jq  # noqa: E402
from omnibias.binary.torch import ops as tq  # noqa: E402

BETA = 4.0
DELTA = 0.5
LO, HI = -1.0, 1.0
Z = np.array([-1.7, -0.55, -0.05, 0.2, 0.7, 1.4], dtype=np.float64)


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _torch_grad_beta(fn, *args):  # type: ignore[no-untyped-def]
    z = torch.tensor(Z, dtype=torch.float64)
    beta = torch.tensor(BETA, dtype=torch.float64, requires_grad=True)
    fn(z, beta, *args).sum().backward()
    return float(beta.grad)


def test_binarize_learnable_beta_matches_torch() -> None:
    tgrad = _torch_grad_beta(tq.binarize)
    jgrad = float(jax.grad(lambda b: jnp.sum(jq.binarize(jnp.asarray(Z), b)))(BETA))
    assert jgrad == pytest.approx(tgrad, rel=1e-9, abs=1e-11)


def test_ternarize_learnable_beta_matches_torch() -> None:
    tgrad = _torch_grad_beta(tq.ternarize, DELTA)
    jgrad = float(
        jax.grad(lambda b: jnp.sum(jq.ternarize(jnp.asarray(Z), b, DELTA)))(BETA)
    )
    assert jgrad == pytest.approx(tgrad, rel=1e-9, abs=1e-11)


def test_kbit_learnable_beta_matches_torch() -> None:
    # bits/lo/hi are positional non-differentiated args.
    z = torch.tensor(Z, dtype=torch.float64)
    beta = torch.tensor(BETA, dtype=torch.float64, requires_grad=True)
    tq.kbit_quantize(z, 2, LO, HI, beta).sum().backward()
    tgrad = float(beta.grad)
    jgrad = float(
        jax.grad(lambda b: jnp.sum(jq.kbit_quantize(jnp.asarray(Z), 2, LO, HI, b)))(BETA)
    )
    assert jgrad == pytest.approx(tgrad, rel=1e-9, abs=1e-11)


def test_beta_gradient_is_nonzero() -> None:
    # Sanity: the surrogate genuinely depends on beta near the boundary.
    jgrad = float(jax.grad(lambda b: jnp.sum(jq.binarize(jnp.asarray(Z), b)))(BETA))
    assert abs(jgrad) > 1e-6


def test_z_gradient_still_matches_after_beta_change() -> None:
    # Regression: enabling grad_beta must not perturb the z-gradient.
    z = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
    tq.binarize(z, BETA).sum().backward()
    tgrad_z = _np(z.grad)
    jgrad_z = _np(jax.grad(lambda zz: jnp.sum(jq.binarize(zz, BETA)))(jnp.asarray(Z)))
    assert np.allclose(tgrad_z, jgrad_z, rtol=1e-9, atol=1e-11)
