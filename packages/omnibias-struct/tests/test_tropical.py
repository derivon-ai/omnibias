# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tropical-log homotopy G1–G3 (theory 01-08). G4 is --full only."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.partition.arrangement import Arrangement, enumerate_cells_vertices
from omnibias.struct._core.tropical import (
    TropicalLinear,
    certify_tropical_gap,
    dual_subdivision,
    homotopy_gap_bound,
    newton_polytope,
    relaxed_grad,
    relaxed_hess,
    relaxed_value,
    tropical_value,
)


def _poly(n: int = 4, d: int = 2, seed: int = 0) -> TropicalLinear:
    rng = np.random.default_rng(seed)
    return TropicalLinear(rng.normal(size=n), rng.normal(size=(n, d)))


def test_g1_gap_soundness() -> None:
    poly = _poly()
    rng = np.random.default_rng(1)
    grid = np.stack(
        np.meshgrid(np.linspace(-1.5, 1.5, 21), np.linspace(-1.5, 1.5, 21), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    extra = rng.normal(size=(40, 2))
    x = np.concatenate([grid, extra], axis=0)
    for beta in (0.5, 2.0, 20.0):
        cert = certify_tropical_gap(poly, x, beta=beta)
        assert cert.is_sound
        assert cert.bound == pytest.approx(homotopy_gap_bound(poly, beta=beta))
        assert float(np.max(relaxed_value(poly, x, beta=beta) - tropical_value(poly, x))) <= cert.bound + 1e-12


def test_g2_subdivision_agrees_with_arrangement_sampler() -> None:
    poly = TropicalLinear(
        np.array([0.0, 0.2, -0.1, 0.4]),
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.2, 0.8]]),
    )
    cells = dual_subdivision(poly, n_samples=8000, seed=0)
    assert len(cells) >= 2
    # Difference arrangement of the tropical terms.
    n = poly.n
    normals = []
    offsets = []
    for i in range(n):
        for j in range(i + 1, n):
            normals.append(poly.exponents[i] - poly.exponents[j])
            offsets.append(poly.coeffs[j] - poly.coeffs[i])
    arr = Arrangement(np.asarray(normals), np.asarray(offsets))
    arr_cells = enumerate_cells_vertices(arr)
    # Each tropical full-dim cell sits inside some arrangement cell; counts are comparable.
    assert len(cells) <= len(arr_cells)
    verts = newton_polytope(poly)
    assert len(verts) >= 3


def test_g3_derivatives_vs_fd() -> None:
    poly = _poly(5, 2, seed=3)
    x = np.array([0.15, -0.22])
    beta = 3.0
    def rv(pt: np.ndarray) -> float:
        return float(np.asarray(relaxed_value(poly, pt, beta=beta)).reshape(-1)[0])

    g = relaxed_grad(poly, x, beta=beta)
    h = 1e-5
    fd = np.array(
        [
            (rv(x + np.array([h, 0.0])) - rv(x - np.array([h, 0.0]))) / (2 * h),
            (rv(x + np.array([0.0, h])) - rv(x - np.array([0.0, h]))) / (2 * h),
        ]
    )
    g1 = np.asarray(g).reshape(-1)
    rel = float(np.linalg.norm(g1 - fd) / max(np.linalg.norm(fd), 1e-12))
    assert rel <= 1e-6
    hess = np.asarray(relaxed_hess(poly, x, beta=beta))
    hh = 1e-4
    h00 = (rv(x + np.array([hh, 0.0])) - 2 * rv(x) + rv(x - np.array([hh, 0.0]))) / (hh * hh)
    scale = max(abs(float(hess[0, 0])), abs(h00), 1e-8)
    assert abs(float(hess[0, 0]) - h00) / scale <= 1e-4


def test_refuse_large() -> None:
    with pytest.raises(ValueError, match="refuse"):
        TropicalLinear(np.zeros(12), np.zeros((12, 2)))


def test_g4_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax.tropical import relaxed_value as rv_jax
    from omnibias.struct.torch.tropical import relaxed_value as rv_torch

    jax.config.update("jax_enable_x64", True)
    poly = _poly()
    x = np.array([[0.1, -0.2], [0.3, 0.4]])
    t = rv_torch(poly, torch.as_tensor(x, dtype=torch.float64), beta=2.0)
    j = rv_jax(poly, jnp.asarray(x, dtype=jnp.float64), beta=2.0)
    np.testing.assert_allclose(t.detach().cpu().numpy(), np.asarray(j), rtol=0.0, atol=1e-12)
