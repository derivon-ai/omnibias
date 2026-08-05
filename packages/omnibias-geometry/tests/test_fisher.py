# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Information-geometry bridge: exponential-family Fisher-Rao metric.

The Fisher information of an exponential family is a Riemannian metric, so it
flows straight into the connection / curvature ops:

1. the metric equals ``diag(A''(eta_k))`` from the closed-form cumulant tower;
2. it is *flat* -- scalar curvature is ``0`` (the exponential family is dually
   flat), a non-trivial fact validated through the autodiff curvature pipeline;
3. yet its Christoffel symbols are the closed-form tower ratio
   ``Gamma^k_kk = A'''(eta_k) / (2 A''(eta_k))``;
4. torch and jax agree to float64.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.geometry import ManifoldSpec
from omnibias.geometry.jax import ops as jgeo
from omnibias.geometry.torch import ops as tgeo

# Only genuine (convex, A'' > 0) log-partitions induce a valid Fisher metric:
# softplus -> Bernoulli, exp -> Poisson. (tanh is not a log-partition.)
COORDS = np.array([[0.3, -0.7], [1.0, 0.5], [-1.2, 0.9], [0.0, 0.0]], dtype=np.float64)
_BASES = ("softplus", "exp")


def _jc():
    return jnp.asarray(COORDS)


def _tc():
    return torch.tensor(COORDS, dtype=torch.float64)


# ----- metric value ---------------------------------------------------------


@pytest.mark.parametrize("base", _BASES)
def test_metric_is_diag_of_glm_variance(base: str) -> None:
    from omnibias.jax.information import glm_variance as jvar
    from omnibias.torch.information import glm_variance as tvar

    jm = jgeo.exponential_family_fisher_manifold(base=base, dim=2)
    tm = tgeo.exponential_family_fisher_manifold(base=base, dim=2)
    gj = np.asarray(jgeo.metric(_jc(), jm))
    gt = tgeo.metric(_tc(), tm).numpy()
    for i, row in enumerate(COORDS):
        vj = np.asarray(jvar(jnp.asarray(row), base=base))
        assert np.allclose(gj[i], np.diag(vj), atol=1e-12)
        vt = tvar(torch.tensor(row), base=base).numpy()
        assert np.allclose(gt[i], np.diag(vt), atol=1e-12)


# ----- flatness (dually-flat exponential family) ----------------------------


@pytest.mark.parametrize("base", _BASES)
def test_scalar_curvature_is_zero(base: str) -> None:
    jm = jgeo.exponential_family_fisher_manifold(base=base, dim=2)
    tm = tgeo.exponential_family_fisher_manifold(base=base, dim=2)
    rj = np.asarray(jgeo.scalar_curvature(_jc(), jm))
    rt = tgeo.scalar_curvature(_tc(), tm).numpy()
    assert np.allclose(rj, 0.0, atol=1e-8)
    assert np.allclose(rt, 0.0, atol=1e-8)


@pytest.mark.parametrize("base", _BASES)
def test_riemann_tensor_vanishes(base: str) -> None:
    jm = jgeo.exponential_family_fisher_manifold(base=base, dim=2)
    riem = np.asarray(jgeo.riemann_tensor(_jc(), jm))
    assert np.allclose(riem, 0.0, atol=1e-8)


# ----- Christoffel closed form ----------------------------------------------


@pytest.mark.parametrize("base", _BASES)
def test_christoffel_matches_tower_ratio(base: str) -> None:
    from omnibias.jax.information import exponential_family_cumulants as jcum
    from omnibias.jax.information import glm_variance as jvar

    jm = jgeo.exponential_family_fisher_manifold(base=base, dim=2)
    gamma = np.asarray(jgeo.christoffel(_jc(), jm))  # (B, k, i, j) = Gamma^k_ij
    app = np.asarray(jvar(_jc(), base=base))  # (B, 2)  A''
    appp = np.asarray(jcum(_jc(), base=base, order=3)[3])  # (B, 2)  A'''
    closed = appp / (2.0 * app)
    for k in range(2):
        assert np.allclose(gamma[:, k, k, k], closed[:, k], atol=1e-9)
    # only the (k, k, k) entries are non-zero for a separable diagonal metric
    mask = gamma.copy()
    for k in range(2):
        mask[:, k, k, k] = 0.0
    assert np.allclose(mask, 0.0, atol=1e-9)


def test_torch_jax_christoffel_parity() -> None:
    jm = jgeo.exponential_family_fisher_manifold(base="softplus", dim=2)
    tm = tgeo.exponential_family_fisher_manifold(base="softplus", dim=2)
    gj = np.asarray(jgeo.christoffel(_jc(), jm))
    gt = tgeo.christoffel(_tc(), tm).numpy()
    assert np.allclose(gj, gt, rtol=1e-9, atol=1e-11)


# ----- spec plumbing --------------------------------------------------------


def test_manifold_builder_returns_manifold_spec() -> None:
    m = jgeo.exponential_family_fisher_metric(base="softplus", dim=3)
    assert m.dim == 3
    assert "fisher" in m.name
    manifold = jgeo.exponential_family_fisher_manifold(base="exp", dim=2)
    assert isinstance(manifold, ManifoldSpec)
    assert manifold.dim == 2
