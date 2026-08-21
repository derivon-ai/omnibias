# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-05: differentiable morphology, gates G1–G6."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from omnibias.shape.morphology import (
    StructuringElement,
    dilate,
    erode,
    exact_distance_transform,
    hard_dilate,
    hard_erode,
    morphological_gradient,
    morphology_gap_bound,
    named_worked_signal,
    named_worked_soft_center,
    opening,
    soft_distance_transform,
    soft_max_pool,
)


def test_worked_example() -> None:
    f = named_worked_signal()
    se = StructuringElement.flat(1)
    hard = hard_dilate(f, se)
    np.testing.assert_array_equal(hard, np.array([1.0, 3.0, 3.0, 3.0, 1.0]))
    soft = named_worked_soft_center(beta=2.0)
    assert abs(soft - 3.017988) < 5e-6
    assert 0.0 <= soft - 3.0 <= math.log(3.0) / 2.0
    high = float(dilate(f, se, beta=10.0).value[2])
    assert abs(high - 3.0) < 1e-8


def test_g1_hard_limit_rate() -> None:
    # Equal window values: the gap is exactly log(N)/beta (predicted rate).
    f = np.full(7, 2.0, dtype=np.float64)
    se = StructuringElement.flat(1)
    hard = hard_dilate(f, se)
    betas = [2.0, 4.0, 8.0, 16.0, 32.0]
    devs = [float(np.max(dilate(f, se, beta=b).value - hard)) for b in betas]
    # Four doublings: deviation halves each time (1/beta rate).
    for i in range(len(devs) - 1):
        a, b = devs[i], devs[i + 1]
        ratio = b / max(a, 1e-18)
        assert 0.4 <= ratio <= 0.6


def test_g2_bound_and_onesided() -> None:
    rng = np.random.default_rng(0)
    se = StructuringElement.flat(1)
    beta = 3.0
    bound = morphology_gap_bound(size=se.size, beta=beta, compositions=1)
    for _ in range(8):
        f = rng.normal(size=16)
        d = dilate(f, se, beta=beta)
        e = erode(f, se, beta=beta)
        hd = hard_dilate(f, se)
        he = hard_erode(f, se)
        assert np.all(d.value + 1e-12 >= hd)
        assert np.all(e.value <= he + 1e-12)
        assert np.all(d.value - hd <= bound + 1e-12)
        assert np.all(he - e.value <= bound + 1e-12)
        assert np.all(np.isfinite(he))
        assert np.all(np.isfinite(e.value))
    # Dense grid on the worked signal.
    f = named_worked_signal()
    d = dilate(f, se, beta=beta)
    assert np.all(d.value >= hard_dilate(f, se) - 1e-15)


def test_g3_composition_accounting() -> None:
    # Constant interior: each soft step hits nearly the full log(N)/beta.
    f = np.full(9, 2.0, dtype=np.float64)
    se = StructuringElement.flat(1)
    beta = 1.5
    one = morphology_gap_bound(size=se.size, beta=beta, compositions=1)
    two = morphology_gap_bound(size=se.size, beta=beta, compositions=2)
    op = opening(f, se, beta=beta)
    assert op.compositions == 2
    assert abs(op.gap_bound - two) < 1e-12
    # Gradient of a constant: errors add, so the one-operator bound is violated.
    g = morphological_gradient(f, se, beta=beta)
    assert g.compositions == 2
    assert g.observed_dev > one
    assert g.observed_dev <= two + 1e-12
    # Opening still reports the composition-aware bound (cannot silently regress).
    assert op.gap_bound == two


def test_g4_learnable_se() -> None:
    se_hand = StructuringElement.flat(2)
    clean = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float64)
    skills = []
    for seed in range(5):
        rng = np.random.default_rng(20 + seed)
        noisy = clean.copy()
        # Width-2 salt spikes on the background (flat-3 opening cannot clear them).
        for pos in (0, 7):
            if rng.random() > 0.2:
                noisy[pos] = 1.0
                if pos + 1 < noisy.size:
                    noisy[pos + 1] = 1.0
        def mse(se: StructuringElement, signal: np.ndarray = noisy) -> float:
            return float(np.mean((opening(signal, se, beta=8.0).value - clean) ** 2))

        hand = mse(se_hand)
        # Learned: 5-tap values, GD on MSE.
        vals = np.zeros(5, dtype=np.float64)
        off = np.arange(-2, 3, dtype=np.int64)
        lr = 0.35
        for _ in range(40):
            g = np.zeros_like(vals)
            for i in range(vals.size):
                hi, lo = vals.copy(), vals.copy()
                hi[i] += 1e-3
                lo[i] -= 1e-3
                g[i] = (mse(StructuringElement(off, hi)) - mse(StructuringElement(off, lo))) / 2e-3
            vals = vals - lr * g
        learned = mse(StructuringElement(off, vals))
        rel = (hand - learned) / max(hand, 1e-12)
        skills.append(rel)
    assert float(np.mean(skills)) >= 0.20
    assert float(np.mean(skills)) > 0.0


def test_g5_distance_transform() -> None:
    occ = np.array([0, 0, 1, 0, 0, 1, 0], dtype=np.float64)
    exact = exact_distance_transform(occ)
    beta = 4.0
    soft = soft_distance_transform(occ, beta=beta)
    assert np.all(soft.value <= exact + 1e-12)
    assert np.all(exact - soft.value <= soft.gap_bound + 1e-12)
    assert abs(soft.gap_bound - math.log(2.0) / beta) < 1e-12


def test_g6_parity() -> None:
    f = named_worked_signal()
    se = StructuringElement.flat(1)
    ref = dilate(f, se, beta=2.0).value
    torch = pytest.importorskip("torch")
    tw = __import__("omnibias.shape.morphology.torch", fromlist=["dilate"])
    got_t = tw.dilate(torch.as_tensor(f), se, beta=2.0).detach().cpu().numpy()
    np.testing.assert_allclose(got_t, ref, rtol=0.0, atol=1e-15)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jw = __import__("omnibias.shape.morphology.jax", fromlist=["dilate"])
    got_j = np.asarray(jw.dilate(f, se, beta=2.0))
    np.testing.assert_allclose(got_j, ref, rtol=0.0, atol=2e-15)


def test_overflow_shifted_lse() -> None:
    f = np.array([20.0, 21.0, 20.0], dtype=np.float64)
    se = StructuringElement.flat(1)
    out = dilate(f, se, beta=50.0)
    assert np.all(np.isfinite(out.value))
    assert np.all(out.value >= hard_dilate(f, se) - 1e-12)


def test_soft_max_pool_and_terminology() -> None:
    f = named_worked_signal()
    pooled = soft_max_pool(f, kernel=3, stride=1, beta=2.0)
    assert pooled.value.shape == (3,)
    assert abs(float(pooled.value[1]) - named_worked_soft_center(beta=2.0)) < 1e-12
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "shape" / "morphology"
    for name in ("_core.py", "torch.py", "jax.py", "__init__.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text.lower()
        assert "do not conflate" in text.lower()
