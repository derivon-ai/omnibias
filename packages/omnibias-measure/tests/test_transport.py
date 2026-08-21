# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-04: sliced OT of activation mixtures, gates G1–G6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omnibias.measure.transport import (
    ActivationMixture,
    named_worked_pair,
    quantile,
    sliced_wasserstein,
    w1_exact,
    worked_w1,
    wp_quantile,
)
from omnibias.measure.transport._core import (
    bootstrap_direction_stderr,
    highprec_w1,
    krawczyk_root_count,
    mc_sliced_wasserstein,
    quantile_dtheta_means,
    sign_change_roots,
)


def test_worked_example() -> None:
    mu, nu = named_worked_pair()
    w1 = w1_exact(mu, nu)
    assert abs(w1 - worked_w1()) < 1e-12
    assert abs(w1 - 0.2402292) < 5e-7


def test_g1_exactness() -> None:
    rng = np.random.default_rng(0)
    rels: list[float] = []
    pairs = [named_worked_pair()]
    for _ in range(6):
        n = int(rng.integers(1, 4))
        w = rng.random(n)
        w = w / w.sum()
        m = rng.normal(size=n)
        a = rng.uniform(0.6, 2.5, size=n)
        mu = ActivationMixture(w, m, a)
        n2 = int(rng.integers(1, 4))
        w2 = rng.random(n2)
        w2 = w2 / w2.sum()
        nu = ActivationMixture(w2, rng.normal(size=n2), rng.uniform(0.6, 2.5, size=n2))
        pairs.append((mu, nu))
    # Multi-root: two wide bumps vs two inner bumps.
    pairs.append(
        (
            ActivationMixture([0.5, 0.5], [-3.0, 3.0], [1.2, 1.2]),
            ActivationMixture([0.5, 0.5], [-1.0, 1.0], [1.2, 1.2]),
        )
    )
    for mu, nu in pairs:
        exact = w1_exact(mu, nu)
        trap = highprec_w1(mu, nu)
        rel = abs(exact - trap) / max(trap, 1e-15)
        rels.append(rel)
        assert rel <= 1e-12 or abs(exact - trap) <= 1e-12
        roots = sign_change_roots(mu, nu)
        assert krawczyk_root_count(mu, nu, roots) == int(roots.size)
    assert max(rels) <= 1e-12


def test_g2_quantile_gradient() -> None:
    mix = ActivationMixture([0.4, 0.6], [-0.7, 0.9], [1.1, 0.8])
    q = 0.35
    analytic = quantile_dtheta_means(mix, q)
    fd = np.empty_like(analytic)
    h = 1e-6
    for i in range(mix.n_components):
        m_hi = mix.means.reshape(-1).copy()
        m_lo = m_hi.copy()
        m_hi[i] += h
        m_lo[i] -= h
        q_hi = float(quantile(ActivationMixture(mix.weights, m_hi, mix.scales), q))
        q_lo = float(quantile(ActivationMixture(mix.weights, m_lo, mix.scales), q))
        fd[i] = (q_hi - q_lo) / (2.0 * h)
    rel = np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-12))
    assert rel <= 1e-8 or np.max(np.abs(analytic - fd)) <= 1e-8


def test_g3_variance_win() -> None:
    # Rotationally invariant pair: every slice is the same 1-D W_1, so the
    # closed-form estimator has only (here: zero) direction variance.
    mu = ActivationMixture([1.0], [[0.0, 0.0]], [1.0])
    nu = ActivationMixture([1.0], [[0.0, 0.0]], [2.0])
    cf_vals = [sliced_wasserstein(mu, nu, p=1, directions=8, seed=s).value for s in range(12)]
    mc_vals = [
        mc_sliced_wasserstein(mu, nu, n_samples=64, directions=8, seed=s).value for s in range(12)
    ]
    v_cf = float(np.var(cf_vals, ddof=1))
    v_mc = float(np.var(mc_vals, ddof=1))
    assert v_mc / max(v_cf, 1e-18) >= 100.0


def test_g4_metric() -> None:
    rng = np.random.default_rng(1)
    mixes = []
    for _ in range(6):
        n = 2
        w = rng.random(n)
        mixes.append(ActivationMixture(w / w.sum(), rng.normal(size=n), rng.uniform(0.8, 1.6, size=n)))
    for a, b in zip(mixes, mixes[1:] + mixes[:1], strict=True):
        assert abs(w1_exact(a, b) - w1_exact(b, a)) <= 1e-10
        assert w1_exact(a, a) <= 1e-10
    a, b, c = mixes[0], mixes[1], mixes[2]
    assert w1_exact(a, c) <= w1_exact(a, b) + w1_exact(b, c) + 1e-10


def test_g5_direction_stderr() -> None:
    mu = ActivationMixture([0.5, 0.5], [[-1.2, 0.3], [0.8, -0.4]], [1.0, 1.1])
    nu = ActivationMixture([0.6, 0.4], [[0.0, 0.0], [0.5, 0.7]], [0.9, 1.2])
    result = sliced_wasserstein(mu, nu, p=1, directions=24, seed=2)
    assert result.direction_stderr >= 0.0
    boot = bootstrap_direction_stderr(result.slices, n_boot=300, seed=3)
    # Same order; bootstrap SE of the mean tracks std/sqrt(L).
    rel = abs(result.direction_stderr - boot) / max(boot, 1e-15)
    assert rel < 0.35


def test_g6_parity() -> None:
    mu, _nu = named_worked_pair()
    xs = np.array([-1.5, 0.0, 0.7], dtype=np.float64)
    ref = mu.cdf(xs)
    torch = pytest.importorskip("torch")
    tw = __import__("omnibias.measure.transport.torch", fromlist=["cdf"])
    got_t = tw.cdf(mu, torch.as_tensor(xs)).detach().cpu().numpy()
    np.testing.assert_allclose(got_t, ref, rtol=0.0, atol=0.0)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jw = __import__("omnibias.measure.transport.jax", fromlist=["cdf"])
    got_j = np.asarray(jw.cdf(mu, xs))
    np.testing.assert_allclose(got_j, ref, rtol=0.0, atol=2e-16)


def test_wp_and_quantile_roundtrip() -> None:
    mu, nu = named_worked_pair()
    q = 0.6
    x = float(quantile(mu, q))
    assert abs(float(mu.cdf(x)) - q) < 1e-12
    w2 = wp_quantile(mu, nu, p=2, n_quad=32)
    assert w2 > 0.0


def test_terminology() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "measure" / "transport"
    for name in ("_core.py", "torch.py", "jax.py", "__init__.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text.lower()
        assert "do not conflate" in text.lower()
