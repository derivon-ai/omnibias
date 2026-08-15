# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hyperplane arrangement G1–G4 (theory 01-03)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.partition._core.config import PartitionConfig
from omnibias.partition._core.params import init_params, region_code_matrix
from omnibias.partition._core.weights import partition_weights
from omnibias.partition.arrangement import (
    Arrangement,
    certify_cell_gap,
    enumerate_cells_vertices,
    general_position_normals,
    max_cells,
    realized_cells,
    soft_membership,
    tree_arrangement,
)


def test_g1_general_position_cell_count() -> None:
    rng = np.random.default_rng(0)
    for n, d in ((5, 2), (8, 2), (6, 3), (10, 3), (8, 4)):
        arr = general_position_normals(n, d, rng)
        cells = enumerate_cells_vertices(arr)
        assert len(cells) == max_cells(n, d), (n, d, len(cells), max_cells(n, d))


def test_g1_degenerate_strictly_smaller() -> None:
    # Two parallel lines in R^2: fewer cells than the simple maximum.
    arr = Arrangement(
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([-1.0, 1.0, 0.0], dtype=np.float64),
    )
    cells = enumerate_cells_vertices(arr)
    brute = realized_cells(
        arr,
        np.stack(
            np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2),
    )
    assert len(cells) == len(brute)
    assert len(cells) < max_cells(3, 2)


def test_g2_gap_soundness() -> None:
    rng = np.random.default_rng(1)
    arr = general_position_normals(5, 2, rng)
    cells = enumerate_cells_vertices(arr)
    signs = cells[0]
    grid = np.stack(
        np.meshgrid(np.linspace(-1.5, 1.5, 25), np.linspace(-1.5, 1.5, 25), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    extra = rng.normal(size=(80, 2))
    x = np.concatenate([grid, extra], axis=0)
    for beta in (1.0, 4.0, 16.0):
        cert = certify_cell_gap(arr, x, signs, beta=beta)
        assert cert.is_sound
        assert cert.bound >= cert.measured - 1e-12


def test_g3_tree_agrees_with_partition_weights() -> None:
    cfg = PartitionConfig(n_features=3, depth=3)
    params = init_params(cfg, rng=0)
    arr = tree_arrangement(params.W, params.t)
    x = np.array([[0.2, -0.1, 0.4], [1.0, 0.0, -0.5], [-0.3, 0.7, 0.1]], dtype=np.float64)
    beta = 3.5
    pw = partition_weights(params, x, beta)
    codes = region_code_matrix(3)
    eps = np.finfo(np.float64).eps
    for leaf in range(8):
        signs = tuple(1 if codes[leaf, j] > 0.5 else -1 for j in range(3))
        sm = soft_membership(arr, x, signs, beta=beta)
        for a, b in zip(sm, pw[:, leaf], strict=True):
            scale = max(abs(float(a)), abs(float(b)), 1.0)
            ulp = abs(float(a) - float(b)) / (eps * scale)
            assert ulp <= 4.0


def test_g4_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.partition.arrangement.jax import soft_membership as sm_jax
    from omnibias.partition.arrangement.torch import soft_membership as sm_torch

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2)
    arr = general_position_normals(4, 2, rng)
    x = rng.normal(size=(6, 2))
    signs = (1, -1, 1, -1)
    t = sm_torch(arr, torch.as_tensor(x, dtype=torch.float64), signs, beta=2.5)
    j = sm_jax(arr, jnp.asarray(x, dtype=jnp.float64), signs, beta=2.5)
    np.testing.assert_allclose(t.detach().cpu().numpy(), np.asarray(j), rtol=0.0, atol=1e-14)
