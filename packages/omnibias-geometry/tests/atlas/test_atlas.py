# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Region-wise Riemannian atlas (omnibias.geometry.atlas) validation.

A depth-1 atlas over ``R^2`` glues a **flat** chart (``x0 < 0``) to a **conformally-curved**
chart (``x0 > 0``). The blended metric ``g(x) = sum_l w_l(x) G_l(x)`` is a convex combination
of the region metrics, so it is SPD everywhere; deep in each region it recovers that region's
geometry, so scalar curvature and geodesic acceleration differ by region. All float64.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("omnibias.partition")  # the optional 'atlas' extra (alpha keystone)

from omnibias.geometry import ManifoldSpec, MetricSpec  # noqa: E402
from omnibias.geometry.atlas import AtlasSpec  # noqa: E402
from omnibias.geometry.atlas import jax as jatlas  # noqa: E402
from omnibias.geometry.atlas import torch as tatlas  # noqa: E402
from omnibias.geometry.jax import ops as jgeo  # noqa: E402
from omnibias.geometry.torch import ops as tgeo  # noqa: E402
from omnibias.partition import PartitionConfig  # noqa: E402
from omnibias.partition._core.params import PartitionParams  # noqa: E402
from omnibias.partition._core.weights import partition_weights  # noqa: E402

BETA = 16.0
CONF_A = 0.4
CONF_B = 0.3


# --- region metric factories (mirror the conftest (xp, stack) convention) ---
def _flat_region(xp, stack):  # type: ignore[no-untyped-def]
    def g(x):  # type: ignore[no-untyped-def]
        z = 0.0 * x[0]
        return stack([stack([1.0 + z, z]), stack([z, 1.0 + z])])

    return g


def _conformal_region(xp, stack):  # type: ignore[no-untyped-def]
    def g(x):  # type: ignore[no-untyped-def]
        lam = CONF_A * xp.sin(x[0]) + CONF_B * xp.cos(x[1])
        e2 = xp.exp(2.0 * lam)
        z = 0.0 * x[0]
        return stack([stack([e2, z]), stack([z, e2])])

    return g


def _params() -> PartitionParams:
    # depth-1 axis split on feature 0: region 0 = {x0 < 0} (flat), region 1 = {x0 > 0} (curved).
    cfg = PartitionConfig(n_features=2, depth=1, split_kind="axis", beta_final=BETA, anneal_steps=1)
    return PartitionParams(cfg, W=np.array([[1.0, 0.0]]), t=np.array([0.0]))


def _torch_atlas() -> ManifoldSpec:
    atlas = AtlasSpec(
        _params(),
        region_metrics=(_flat_region(torch, torch.stack), _conformal_region(torch, torch.stack)),
        beta=BETA,
    )
    return tatlas.atlas_manifold(atlas)


def _jax_atlas() -> ManifoldSpec:
    atlas = AtlasSpec(
        _params(),
        region_metrics=(_flat_region(jnp, jnp.stack), _conformal_region(jnp, jnp.stack)),
        beta=BETA,
    )
    return jatlas.atlas_manifold(atlas)


def _pure_conformal_torch() -> ManifoldSpec:
    return ManifoldSpec("conf", 2, MetricSpec(_conformal_region(torch, torch.stack), dim=2))


def _pure_flat_torch() -> ManifoldSpec:
    return ManifoldSpec("flat", 2, MetricSpec(_flat_region(torch, torch.stack), dim=2))


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ----------------------------------------------------------------------
# 1. The mathematical core: convex-combo-of-SPD is SPD (backend-free unit test)
# ----------------------------------------------------------------------
def test_convex_combination_of_spd_is_spd() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = int(rng.integers(2, 5))
        n = int(rng.integers(2, 6))
        mats = []
        for _ in range(n):
            a = rng.standard_normal((d, d))
            mats.append(a @ a.T + 0.5 * np.eye(d))  # SPD
        w = rng.random(n)
        w /= w.sum()  # partition of unity
        blend = sum(wi * m for wi, m in zip(w, mats, strict=True))
        assert np.allclose(blend, blend.T)
        assert np.linalg.eigvalsh(blend).min() > 0.0


# ----------------------------------------------------------------------
# 2. The builder's blend equals sum_l w_l G_l with the keystone POU weights
# ----------------------------------------------------------------------
def test_blended_metric_matches_keystone_weights() -> None:
    params = _params()
    manifold = _torch_atlas()
    flat = _flat_region(torch, torch.stack)
    conf = _conformal_region(torch, torch.stack)
    pts = np.array([[-1.7, 0.6], [-0.2, -1.1], [0.15, 0.9], [2.1, -0.4]])
    W = partition_weights(params, pts, BETA)  # (n, 2)
    for i, p in enumerate(pts):
        xt = torch.as_tensor(p, dtype=torch.float64)
        got = _np(manifold.metric.g_point(xt))
        want = W[i, 0] * _np(flat(xt)) + W[i, 1] * _np(conf(xt))
        assert np.allclose(got, want, atol=1e-12)


# ----------------------------------------------------------------------
# 3. SPD everywhere, including near the interface
# ----------------------------------------------------------------------
def test_blended_metric_is_spd_across_the_domain() -> None:
    manifold = _torch_atlas()
    xs = np.linspace(-3.0, 3.0, 25)
    ys = np.linspace(-3.0, 3.0, 7)
    coords = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    g = _np(tgeo.metric(torch.as_tensor(coords, dtype=torch.float64), manifold))  # (B, 2, 2)
    assert np.allclose(g, np.transpose(g, (0, 2, 1)), atol=1e-12)  # symmetric
    eigs = np.linalg.eigvalsh(g)
    assert eigs.min() > 0.0  # positive-definite everywhere


# ----------------------------------------------------------------------
# 4. Scalar curvature recovers each region's geometry (differs by region)
# ----------------------------------------------------------------------
def test_scalar_curvature_recovers_region_geometry() -> None:
    manifold = _torch_atlas()
    deep_flat = torch.as_tensor([[-2.6, 0.5]], dtype=torch.float64)
    deep_curved = torch.as_tensor([[2.6, 0.5]], dtype=torch.float64)

    r_flat = float(_np(tgeo.scalar_curvature(deep_flat, manifold))[0])
    r_curved = float(_np(tgeo.scalar_curvature(deep_curved, manifold))[0])

    # deep in the flat region the atlas is flat; deep in the curved region it recovers the
    # pure conformal curvature (blend is one-hot to ~exp(-beta*|x0|)).
    r_flat_ref = float(_np(tgeo.scalar_curvature(deep_flat, _pure_flat_torch()))[0])
    r_curved_ref = float(_np(tgeo.scalar_curvature(deep_curved, _pure_conformal_torch()))[0])

    assert abs(r_flat - r_flat_ref) < 1e-6
    assert abs(r_flat) < 1e-6  # flat
    assert abs(r_curved - r_curved_ref) < 1e-5
    assert abs(r_curved - r_flat) > 0.1  # genuinely different geometry per region


# ----------------------------------------------------------------------
# 5. Geodesics are straight in the flat region, bend in the curved region
# ----------------------------------------------------------------------
def test_geodesic_acceleration_differs_by_region() -> None:
    manifold = _torch_atlas()
    vel = torch.as_tensor([[1.0, 0.7]], dtype=torch.float64)
    deep_flat = torch.as_tensor([[-2.6, 0.5]], dtype=torch.float64)
    deep_curved = torch.as_tensor([[2.6, 0.5]], dtype=torch.float64)

    a_flat = _np(tgeo.geodesic_rhs(deep_flat, vel, manifold))
    a_curved = _np(tgeo.geodesic_rhs(deep_curved, vel, manifold))

    assert np.linalg.norm(a_flat) < 1e-6  # straight lines in the flat chart
    assert np.linalg.norm(a_curved) > 1e-2  # geodesics bend in the curved chart


# ----------------------------------------------------------------------
# 6. torch <-> jax parity of the downstream operators on the atlas metric
# ----------------------------------------------------------------------
def test_cross_backend_parity() -> None:
    tm = _torch_atlas()
    jm = _jax_atlas()
    coords = np.array([[-2.2, 0.4], [-0.3, 1.1], [0.25, -0.7], [2.3, 0.9]], dtype=np.float64)
    tc = torch.as_tensor(coords, dtype=torch.float64)
    jc = jnp.asarray(coords, dtype=jnp.float64)

    assert np.allclose(
        _np(tgeo.scalar_curvature(tc, tm)), _np(jgeo.scalar_curvature(jc, jm)), atol=1e-6
    )
    assert np.allclose(
        _np(tgeo.christoffel(tc, tm)), _np(jgeo.christoffel(jc, jm)), atol=1e-6
    )


# ----------------------------------------------------------------------
# 7. Wrong number of region metrics is rejected
# ----------------------------------------------------------------------
def test_wrong_number_of_region_metrics_raises() -> None:
    with pytest.raises(ValueError, match="region metrics"):
        AtlasSpec(_params(), region_metrics=(_flat_region(torch, torch.stack),), beta=BETA)
