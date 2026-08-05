# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified soft partition of unity -- omnibias-partition (the keystone primitive).

Run:

    pip install "omnibias-partition[torch,jax]"
    python docs/examples/partition_quickstart.py

`depth` oblique split gates ``g(x) = sigmoid(beta (w.x - t))`` route an input into
``2**depth`` regions with weights that are non-negative, sum to one (a genuine partition of
unity), and **harden** to a crisp ``{0, 1}`` partition as ``beta -> inf``. That is the
**temperature collapse** axis (``beta -> inf``), *not* the founding
``delta -> 0`` bias collapse (the multi-bias limit to ``sigma^(K-1)``; see ``docs/theory.md``).

This deterministic, CPU-tiny smoke asserts the primitive's contract so it wires in as a CI
smoke:

1. the weights are a partition of unity and harden as ``beta -> inf``;
2. the soft->hard membership gap carries a **sound** certificate (bound >= measured);
3. the torch / jax weight twins reproduce the numpy reference bit-for-bit;
4. the per-region model registry blends via the one ``sum_l w_l * out_l`` engine.
"""

from __future__ import annotations

import numpy as np
from omnibias.partition import (
    PartitionConfig,
    certify_partition_gap,
    hardened_rules,
    init_params,
    partition_weights,
)
from omnibias.partition.registry import RegionModels


def partition_of_unity_demo() -> tuple[object, np.ndarray]:
    print("=== 1. a soft partition of unity that hardens as beta -> inf ===")
    cfg = PartitionConfig(n_features=3, depth=2, split_kind="axis", seed=0)
    params = init_params(cfg, rng=0)
    X = np.random.default_rng(1).standard_normal((256, 3))
    W = partition_weights(params, X, beta=4.0)
    assert np.all(W >= 0.0) and np.allclose(W.sum(1), 1.0)
    print(f"    {cfg.n_regions} regions; rows sum to 1 (max dev "
          f"{np.abs(W.sum(1) - 1).max():.1e}); hardened rules: {hardened_rules(params)}")
    return params, X


def certificate_demo(params: object, X: np.ndarray) -> None:
    print("=== 2. sound soft->hard membership-gap certificate ===")
    soft = certify_partition_gap(params, X, beta=4.0)
    sharp = certify_partition_gap(params, X, beta=64.0)
    assert soft.is_sound and sharp.is_sound
    assert sharp.max_gap <= soft.max_gap + 1e-9
    print(f"    beta=4  max L1 gap <= {soft.max_gap:.4f} (measured {soft.measured_max:.4f}, "
          f"sound={soft.is_sound})")
    print(f"    beta=64 max L1 gap <= {sharp.max_gap:.4f} (measured {sharp.measured_max:.4f}); "
          f"Gibbs scale log(R)/beta = {sharp.gibbs_scale:.4f}")


def parity_demo(params: object, X: np.ndarray) -> None:
    print("=== 3. bit-identical numpy <-> torch <-> jax weights ===")
    ref = partition_weights(params, X, beta=8.0)
    try:
        from omnibias.partition.torch import partition_weights as tw

        dt = float(np.max(np.abs(tw(params, X, 8.0).detach().cpu().numpy() - ref)))
        print(f"    torch vs numpy: {dt:.2e}")
        assert dt < 1e-9
    except ModuleNotFoundError:
        print("    (torch not installed -- skipping)")
    try:
        import jax

        jax.config.update("jax_enable_x64", True)  # float64 parity needs x64 enabled
        from omnibias.partition.jax import partition_weights as jw

        dj = float(np.max(np.abs(np.asarray(jw(params, X, 8.0)) - ref)))
        print(f"    jax   vs numpy: {dj:.2e}")
        assert dj < 1e-9
    except ModuleNotFoundError:
        print("    (jax not installed -- skipping)")


def registry_demo(params: object, X: np.ndarray) -> None:
    print("=== 4. per-region model registry: F(x) = sum_l w_l m_l(x) ===")
    n_regions = params.n_regions  # type: ignore[attr-defined]
    consts = np.linspace(0.0, 1.0, n_regions)
    models = [(lambda X, c=c: np.full((X.shape[0], 1), c)) for c in consts]
    reg = RegionModels(params, models)
    out = reg.combine(X, beta=8.0)
    assert out.min() >= consts.min() - 1e-9 and out.max() <= consts.max() + 1e-9
    print(f"    blended output in [{out.min():.3f}, {out.max():.3f}] "
          f"(convex combo of region values {np.round(consts, 3).tolist()})")


def main() -> None:
    params, X = partition_of_unity_demo()
    certificate_demo(params, X)
    parity_demo(params, X)
    registry_demo(params, X)
    print("\nAll partition-quickstart checks passed.")


if __name__ == "__main__":
    main()
