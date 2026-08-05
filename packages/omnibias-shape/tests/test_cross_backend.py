# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch <-> jax parity for the soft-shape / coverage operator surface."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.shape.jax import ops as jshape  # noqa: E402
from omnibias.shape.torch import ops as tshape  # noqa: E402

_ATOL = 1e-10


def _inputs():
    m, n, side, beta = 7, 8, 3.0, 2.0
    rng = np.random.default_rng(0)
    centers = rng.uniform(1.0, 6.0, size=(3, 2))
    gates = np.array([0.9, 0.7, 0.8])
    ones = (rng.uniform(size=(m, n)) > 0.4).astype(np.float64)
    return m, n, side, beta, centers, gates, ones


def test_soft_box_and_coverage_parity():
    m, n, side, beta, centers, gates, ones = _inputs()
    t_axes = (torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64))
    j_axes = (jnp.arange(m, dtype=jnp.float64), jnp.arange(n, dtype=jnp.float64))
    t_occ = tshape.soft_box(t_axes, torch.tensor(centers), side, beta)
    j_occ = jshape.soft_box(j_axes, jnp.asarray(centers), side, beta)
    assert np.allclose(t_occ.numpy(), np.asarray(j_occ), atol=_ATOL)
    t_cov, _ = tshape.soft_or_coverage(t_occ, torch.tensor(gates))
    j_cov, _ = jshape.soft_or_coverage(j_occ, jnp.asarray(gates))
    assert np.allclose(t_cov.numpy(), np.asarray(j_cov), atol=_ATOL)


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
def test_energy_grad_hessian_parity(loss: str):
    m, n, side, beta, centers, gates, ones = _inputs()
    t_axes = (torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64))
    j_axes = (jnp.arange(m, dtype=jnp.float64), jnp.arange(n, dtype=jnp.float64))
    tc, tg, to = torch.tensor(centers), torch.tensor(gates), torch.tensor(ones)
    jc, jg, jo = jnp.asarray(centers), jnp.asarray(gates), jnp.asarray(ones)

    t_occ = tshape.soft_box(t_axes, tc, side, beta)
    j_occ = jshape.soft_box(j_axes, jc, side, beta)
    t_e = float(tshape.coverage_energy(t_occ, tg, to, loss=loss, kappa=1.5))
    j_e = float(jshape.coverage_energy(j_occ, jg, jo, loss=loss, kappa=1.5))
    assert abs(t_e - j_e) < _ATOL

    t_g = tshape.coverage_energy_grad(t_axes, tc, side, beta, tg, to, loss=loss, kappa=1.5)
    j_g = jshape.coverage_energy_grad(j_axes, jc, side, beta, jg, jo, loss=loss, kappa=1.5)
    assert np.allclose(t_g.numpy(), np.asarray(j_g), atol=_ATOL)

    t_h = tshape.coverage_energy_hessian(t_axes, tc, side, beta, tg, to, loss=loss, kappa=1.5)
    j_h = jshape.coverage_energy_hessian(j_axes, jc, side, beta, jg, jo, loss=loss, kappa=1.5)
    assert np.allclose(t_h.numpy(), np.asarray(j_h), atol=_ATOL)


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
def test_wrt_all_grad_hessian_parity(loss: str):
    m, n, side, beta, centers, gates, ones = _inputs()
    t_axes = (torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64))
    j_axes = (jnp.arange(m, dtype=jnp.float64), jnp.arange(n, dtype=jnp.float64))
    tc, tg, to = torch.tensor(centers), torch.tensor(gates), torch.tensor(ones)
    jc, jg, jo = jnp.asarray(centers), jnp.asarray(gates), jnp.asarray(ones)
    lam = 0.4

    t_g = tshape.coverage_energy_grad(
        t_axes, tc, side, beta, tg, to, loss=loss, kappa=1.5, lam=lam, wrt="all"
    )
    j_g = jshape.coverage_energy_grad(
        j_axes, jc, side, beta, jg, jo, loss=loss, kappa=1.5, lam=lam, wrt="all"
    )
    assert np.allclose(t_g.numpy(), np.asarray(j_g), atol=_ATOL)

    t_h = tshape.coverage_energy_hessian(
        t_axes, tc, side, beta, tg, to, loss=loss, kappa=1.5, lam=lam, wrt="all"
    )
    j_h = jshape.coverage_energy_hessian(
        j_axes, jc, side, beta, jg, jo, loss=loss, kappa=1.5, lam=lam, wrt="all"
    )
    assert np.allclose(t_h.numpy(), np.asarray(j_h), atol=_ATOL)


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
def test_background_term_parity(loss: str):
    m, n, side, beta, centers, gates, ones = _inputs()
    bg = 1.0 - ones
    t_axes = (torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64))
    j_axes = (jnp.arange(m, dtype=jnp.float64), jnp.arange(n, dtype=jnp.float64))
    tc, tg, to, tb = torch.tensor(centers), torch.tensor(gates), torch.tensor(ones), torch.tensor(bg)
    jc, jg, jo, jb = jnp.asarray(centers), jnp.asarray(gates), jnp.asarray(ones), jnp.asarray(bg)
    mu = 0.6

    t_h = tshape.coverage_energy_hessian(
        t_axes, tc, side, beta, tg, to, loss=loss, kappa=1.5, bg_mask=tb, mu=mu, wrt="all"
    )
    j_h = jshape.coverage_energy_hessian(
        j_axes, jc, side, beta, jg, jo, loss=loss, kappa=1.5, bg_mask=jb, mu=mu, wrt="all"
    )
    assert np.allclose(t_h.numpy(), np.asarray(j_h), atol=_ATOL)


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
def test_lse_union_grad_hessian_parity(loss: str):
    m, n, side, beta, centers, gates, ones = _inputs()
    t_axes = (torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64))
    j_axes = (jnp.arange(m, dtype=jnp.float64), jnp.arange(n, dtype=jnp.float64))
    tc, tg, to = torch.tensor(centers), torch.tensor(gates), torch.tensor(ones)
    jc, jg, jo = jnp.asarray(centers), jnp.asarray(gates), jnp.asarray(ones)
    kw_t = dict(loss=loss, kappa=1.5, lam=0.3, union="lse", beta_u=8.0, wrt="all")
    kw_j = dict(loss=loss, kappa=1.5, lam=0.3, union="lse", beta_u=8.0, wrt="all")

    t_g = tshape.coverage_energy_grad(t_axes, tc, side, beta, tg, to, **kw_t)
    j_g = jshape.coverage_energy_grad(j_axes, jc, side, beta, jg, jo, **kw_j)
    assert np.allclose(t_g.numpy(), np.asarray(j_g), atol=_ATOL)

    t_h = tshape.coverage_energy_hessian(t_axes, tc, side, beta, tg, to, **kw_t)
    j_h = jshape.coverage_energy_hessian(j_axes, jc, side, beta, jg, jo, **kw_j)
    assert np.allclose(t_h.numpy(), np.asarray(j_h), atol=_ATOL)
