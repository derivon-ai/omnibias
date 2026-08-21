# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-09: differentiable topology surrogates, gates G1–G6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omnibias.shape.topology import (
    Inconclusive,
    PersistencePair,
    bimodal_saddle_index,
    bottleneck_distance,
    certified_component_count,
    cluster_laplacian,
    connected_components_1d,
    exact_h0_persistence,
    honesty_payload,
    soft_component_count,
    soft_euler_characteristic,
    soft_persistence,
)


def test_honesty_no_betti() -> None:
    assert honesty_payload()["equals_betti"] is False
    assert honesty_payload()["surrogate"] is True


def test_g1_hard_limit_rate() -> None:
    ev = (0.0, 0.0, 0.0, 0.41, 0.55, 0.72)
    true = 3
    betas = (8.0, 16.0, 32.0, 64.0, 128.0)
    errs = [abs(soft_component_count(ev, epsilon=0.2, beta=b).value - true) for b in betas]
    for i in range(len(errs) - 1):
        assert errs[i + 1] <= 0.55 * errs[i]


def test_g2_gap_bound_sound() -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(0.05, 0.9, 17)
    for eps in grid:
        ev = (0.0, 0.0, 0.41, 0.55)
        sc = soft_component_count(ev, epsilon=float(eps), beta=20.0)
        true = sum(1 for lam in ev if lam < eps)
        assert abs(sc.value - true) <= sc.gap_bound + 1e-12
    for _ in range(32):
        ev = tuple(float(v) for v in rng.uniform(0.0, 1.0, size=6))
        eps = float(rng.uniform(0.1, 0.8))
        sc = soft_component_count(ev, epsilon=eps, beta=15.0)
        true = sum(1 for lam in ev if lam < eps)
        assert abs(sc.value - true) <= sc.gap_bound + 1e-12
    masses = (0.9, 0.1, 0.8)
    eu = soft_euler_characteristic(masses, (0, 1, 0))
    hard = 1.0 - 0.0 + 1.0
    assert abs(eu.value - hard) <= eu.gap_bound + 1e-12
    assert eu.n_faces == 3


def test_g3_certified_count() -> None:
    lap = cluster_laplacian(3, n_per=2, gap=0.205)
    got = certified_component_count(lap, epsilon=0.2)
    assert got == 3
    # Near-degenerate: threshold on the spectrum.
    path = cluster_laplacian(1, n_per=2, gap=0.5)
    # eigenvalues 0 and 1.0; epsilon = 1.0 sits on the spectrum.
    bad = certified_component_count(path, epsilon=1.0)
    assert isinstance(bad, Inconclusive)
    rng = np.random.default_rng(1)
    for k in range(1, 5):
        for _ in range(8):
            g = float(rng.uniform(0.3, 0.8))
            lap_k = cluster_laplacian(k, n_per=2, gap=g)
            ans = certified_component_count(lap_k, epsilon=0.15)
            assert ans == k


def test_g4_persistence_bottleneck() -> None:
    xs = np.linspace(-2.0, 2.0, 401)
    field = (xs**2 - 1.0) ** 2
    exact = exact_h0_persistence(field)
    # Same samples, unsmoothed: the 1-D Morse oracle vs itself is 0.
    # Compare to the closed-form two-well diagram after mapping through sigmoid
    # at beta=1 (a monotone transform preserves the pairing order).
    soft = soft_persistence(field, beta=1.0, max_dim=0)
    ref = exact_h0_persistence(1.0 / (1.0 + np.exp(-field)))
    assert bottleneck_distance(soft, ref) <= 1e-6
    assert any(abs(p.persistence - 1.0) < 0.05 for p in exact if math_finite(p))


def math_finite(pair: PersistencePair) -> bool:
    return np.isfinite(pair.death)


def test_g5_persistence_prior() -> None:
    rng = np.random.default_rng(2)
    topo_plain = []
    topo_loss = []
    pix_plain = []
    pix_loss = []
    x = np.linspace(0.0, 1.0, 32)
    for _ in range(5):
        c1 = 0.28 + 0.02 * float(rng.normal())
        c2 = 0.72 + 0.02 * float(rng.normal())
        obs = np.exp(-((x - c1) ** 2) / (2 * 0.045**2)) + np.exp(-((x - c2) ** 2) / (2 * 0.045**2))
        mask = obs >= 0.25

        def run(use_topo: bool, image: np.ndarray, support: np.ndarray) -> tuple[float, int]:
            u = image.copy()
            for _step in range(24):
                grad = np.zeros_like(u)
                grad[support] = 2.0 * (u[support] - image[support])
                if use_topo:
                    # Superlevel death simplex of the extra H0 bar (1-D Morse saddle).
                    grad[bimodal_saddle_index(u)] -= 1.0
                u = np.clip(u - 0.15 * grad, 0.0, 1.0)
            pix = float(np.mean((u[support] - image[support]) ** 2))
            return pix, connected_components_1d(u, threshold=0.22)

        mse_p, n_p = run(False, obs, mask)
        mse_t, n_t = run(True, obs, mask)
        pix_plain.append(mse_p)
        pix_loss.append(mse_t)
        topo_plain.append(abs(n_p - 1))
        topo_loss.append(abs(n_t - 1))
    assert float(np.mean(topo_plain)) > 0.0
    assert float(np.mean(topo_loss)) <= 0.5 * float(np.mean(topo_plain)) + 1e-12
    assert float(np.mean(pix_loss)) <= float(np.mean(pix_plain)) + 0.05


def test_g6_torch_jax_parity() -> None:
    ev = np.array([0.0, 0.0, 0.41, 0.55], dtype=np.float64)
    ref = soft_component_count(ev, epsilon=0.2, beta=50.0).value
    torch = pytest.importorskip("torch")
    tw = __import__("omnibias.shape.topology.torch", fromlist=["soft_component_count"])
    got_t = float(tw.soft_component_count(torch.tensor(ev, dtype=torch.float64), epsilon=0.2, beta=50.0))
    assert abs(got_t - ref) <= 1e-15
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jw = __import__("omnibias.shape.topology.jax", fromlist=["soft_component_count"])
    got_j = float(jw.soft_component_count(ev, epsilon=0.2, beta=50.0))
    assert abs(got_j - ref) <= 2e-15


def test_worked_soft_count() -> None:
    ev = (0.0, 0.0, 0.0, 0.41, 0.55, 0.72)
    sc = soft_component_count(ev, epsilon=0.2, beta=50.0)
    assert abs(sc.value - 2.9998913) < 2e-4


def test_terminology_files() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "shape" / "topology"
    for name in ("_core.py", "torch.py", "jax.py", "__init__.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text.lower()
        assert "do not conflate" in text.lower()
