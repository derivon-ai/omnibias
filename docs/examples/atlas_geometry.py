# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Region-wise Riemannian atlas: a differentiable atlas of local geometries.

`omnibias.geometry.atlas` blends one metric per region of an `omnibias.partition` soft
partition into a single per-point metric ``g(x) = sum_l w_l(x) G_l(x)``. Because the partition
weights are non-negative and sum to one, the blend is a **convex combination of the region
metrics** -- symmetric positive-definite wherever each region metric is, hence a valid
Riemannian metric everywhere. Every downstream operator (``scalar_curvature`` /
``geodesic_rhs`` / ...) is reused unchanged on this metric.

This CPU smoke glues a **flat** chart (``x0 < 0``) to a **conformally-curved** chart
(``x0 > 0``) and shows that:

* the metric is SPD across the whole domain (incl. the interface),
* scalar curvature recovers each region's geometry (0 in the flat chart, nonzero in the
  curved one),
* geodesics are straight in the flat chart and bend in the curved one,
* torch and jax agree.

Terminology: the partition gates harden as ``beta -> inf`` (the feasibility / temperature
sense of "collapse"), distinct from the founding ``delta -> 0`` bias collapse.
"""

from __future__ import annotations

import numpy as np

BETA = 16.0
CONF_A, CONF_B = 0.4, 0.3


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


def main() -> None:
    import torch
    from omnibias.geometry.atlas import AtlasSpec
    from omnibias.geometry.atlas import torch as tatlas
    from omnibias.geometry.torch import ops as tgeo
    from omnibias.partition import PartitionConfig
    from omnibias.partition._core.params import PartitionParams

    # depth-1 axis split on feature 0: region 0 = {x0<0} flat, region 1 = {x0>0} curved.
    cfg = PartitionConfig(n_features=2, depth=1, split_kind="axis", beta_final=BETA, anneal_steps=1)
    params = PartitionParams(cfg, W=np.array([[1.0, 0.0]]), t=np.array([0.0]))
    atlas = AtlasSpec(
        params,
        region_metrics=(_flat_region(torch, torch.stack), _conformal_region(torch, torch.stack)),
        beta=BETA,
    )
    manifold = tatlas.atlas_manifold(atlas)

    from omnibias.partition._core.weights import hardened_rules

    print("=== region-wise Riemannian atlas (flat | conformal) ===")
    print(f"regions={atlas.n_regions}  dim={atlas.dim}  beta={BETA}")
    print("hardened split rule:", hardened_rules(params)[0])

    # 1. SPD everywhere (including the interface at x0 = 0)
    xs = np.linspace(-3.0, 3.0, 61)
    grid = np.array([[x, 0.5] for x in xs], dtype=np.float64)
    g = tgeo.metric(torch.as_tensor(grid, dtype=torch.float64), manifold).detach().numpy()
    min_eig = np.linalg.eigvalsh(g).min()
    print(f"\n[SPD] min eigenvalue of g over a 61-pt line across the interface: {min_eig:.4e}")
    assert min_eig > 0.0

    # 2. scalar curvature deep in each region
    deep_flat = torch.as_tensor([[-2.6, 0.5]], dtype=torch.float64)
    deep_curved = torch.as_tensor([[2.6, 0.5]], dtype=torch.float64)
    r_flat = float(tgeo.scalar_curvature(deep_flat, manifold).item())
    r_curved = float(tgeo.scalar_curvature(deep_curved, manifold).item())
    print("\n[curvature] scalar curvature by region:")
    print(f"    flat chart  (x0=-2.6): R = {r_flat:+.6f}")
    print(f"    curved chart(x0=+2.6): R = {r_curved:+.6f}")
    assert abs(r_flat) < 1e-6
    assert abs(r_curved) > 0.1

    # 3. geodesic acceleration: straight in flat, bends in curved
    vel = torch.as_tensor([[1.0, 0.7]], dtype=torch.float64)
    a_flat = np.linalg.norm(tgeo.geodesic_rhs(deep_flat, vel, manifold).detach().numpy())
    a_curved = np.linalg.norm(tgeo.geodesic_rhs(deep_curved, vel, manifold).detach().numpy())
    print("\n[geodesic] |acceleration| for the same launch velocity:")
    print(f"    flat chart:   {a_flat:.4e}  (straight line)")
    print(f"    curved chart: {a_curved:.4e}  (bends)")
    assert a_flat < 1e-6 < a_curved

    # 4. torch vs jax agreement on the atlas curvature
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from omnibias.geometry.atlas import jax as jatlas
        from omnibias.geometry.jax import ops as jgeo

        jatlas_m = jatlas.atlas_manifold(
            AtlasSpec(
                params,
                region_metrics=(
                    _flat_region(jnp, jnp.stack),
                    _conformal_region(jnp, jnp.stack),
                ),
                beta=BETA,
            )
        )
        rj = float(jgeo.scalar_curvature(jnp.asarray([[2.6, 0.5]]), jatlas_m)[0])
        print(f"\n[parity] jax curved-chart R = {rj:+.6f}  (torch {r_curved:+.6f})")
        assert abs(rj - r_curved) < 1e-6
    except ModuleNotFoundError:
        print("\n[parity] jax not installed; skipping cross-backend check")

    print("\nOK: a single differentiable metric that is flat in one region and curved in "
          "another, SPD everywhere, curvature/geodesics recovered per region.")


if __name__ == "__main__":
    main()
