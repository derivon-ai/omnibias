# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 05-03: vectorized band / integral soft-binning embedding.

Checks that :mod:`omnibias.core.band_embed` reuses (rather than forks) the
:func:`omnibias.core.ftc.ftc_block` window formula, matches the theory
spec's worked example (section 5(iii)) exactly, and satisfies the two exact
identities from section 4(e): the telescoping sum and the FTC derivative
relationship.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.band_embed import band_embed, integral_embed, quantile_thresholds
from omnibias.core.ftc import ftc_block


def test_band_embed_matches_worked_example() -> None:
    """Spec section 5(iii): t=(-1,0,1), beta=4, x=0.3 -> (0.0055, 0.2260, 0.7112, 0.0573)."""
    got = band_embed(np.array([0.3]), np.array([-1.0, 0.0, 1.0]), beta=4.0)[0]
    expected = np.array([0.0055, 0.2260, 0.7112, 0.0573])
    assert np.allclose(got, expected, atol=5e-5)
    assert abs(got.sum() - 1.0) < 1e-12


def test_band_embed_interior_bin_matches_ftc_block() -> None:
    """The one finite (non-open-tail) bin of a 2-threshold grid == ftc_block's deriv."""
    t_lo, t_hi, beta, x = -0.4, 0.9, 2.5, 0.37
    interior = band_embed(np.array([x]), np.array([t_lo, t_hi]), beta=beta)[0, 1]
    b_hi = -beta * t_lo
    b_lo = -beta * t_hi
    _integral, deriv, _collapse = ftc_block(beta * x, 1.0, b_lo, b_hi)
    assert abs(float(interior) - deriv) < 1e-12


def test_integral_embed_interior_bin_matches_ftc_block() -> None:
    """The one finite bin of a 2-threshold grid's integral_embed == ftc_block's integral."""
    t_lo, t_hi, beta, x = 0.05, 1.3, 0.8, -1.1
    interior = integral_embed(np.array([x]), np.array([t_lo, t_hi]), beta=beta)[0, 1]
    b_hi = -beta * t_lo
    b_lo = -beta * t_hi
    integral, _deriv, _collapse = ftc_block(beta * x, 1.0, b_lo, b_hi)
    assert abs(float(interior) - integral) < 1e-12


def test_ftc_derivative_identity() -> None:
    """``d/dx integral_embed(x, t) == beta * band_embed(x, t)`` exactly (central FD)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=25)
    t = np.array([-2.0, -0.5, 0.5, 1.9])
    beta = 3.0
    h = 1e-6
    fd = (integral_embed(x + h, t, beta=beta) - integral_embed(x - h, t, beta=beta)) / (2 * h)
    analytic = beta * band_embed(x, t, beta=beta)
    assert np.max(np.abs(fd - analytic)) < 1e-6


@pytest.mark.parametrize("beta", [0.05, 1.0, 10.0, 500.0])
def test_band_embed_sums_to_one_for_any_beta(beta: float) -> None:
    """Telescoping sum is exact for *every* beta, not just as beta -> inf."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-10.0, 10.0, size=40)
    t = np.array([-3.0, -1.0, 0.2, 2.5])
    total = band_embed(x, t, beta=beta).sum(axis=-1)
    assert np.max(np.abs(total - 1.0)) < 1e-10


def test_integral_embed_telescoping_sum() -> None:
    """``sum_j integral_embed(x, t)[j] == beta * (x - t[0])`` exactly."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-5.0, 5.0, size=30)
    t = np.array([-1.5, 0.3, 1.1])
    beta = 2.2
    total = integral_embed(x, t, beta=beta).sum(axis=-1)
    expected = beta * (x - t[0])
    assert np.max(np.abs(total - expected)) < 1e-9


def test_band_embed_range_and_shape() -> None:
    x = np.linspace(-5.0, 5.0, 17)
    t = np.array([-3.0, -1.0, 1.0, 3.0])
    out = band_embed(x, t, beta=1.5)
    assert out.shape == (17, 5)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_band_embed_per_feature_grid_broadcast() -> None:
    """``x`` shape ``(n, d)`` against a per-feature ``(d, J)`` threshold grid."""
    n, d, j_thresh = 6, 3, 4
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, d))
    t = np.tile(np.linspace(-2.0, 2.0, j_thresh), (d, 1))
    out = band_embed(x, t, beta=1.0)
    assert out.shape == (n, d, j_thresh + 1)
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_band_embed_hardens_as_beta_grows() -> None:
    """Temperature collapse (beta -> inf): the soft bin sharpens to a crisp 0/1 indicator."""
    t = np.array([0.0, 1.0])
    inside, outside = np.array([0.5]), np.array([5.0])
    soft = band_embed(inside, t, beta=1.0)[0, 1]
    sharp = band_embed(inside, t, beta=200.0)[0, 1]
    assert abs(sharp - 1.0) < 1e-6
    assert abs(sharp - 1.0) < abs(soft - 1.0)
    sharp_out = band_embed(outside, t, beta=200.0)[0, 1]
    assert abs(sharp_out - 0.0) < 1e-6


def test_band_embed_rejects_non_increasing_t() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        band_embed(np.array([0.0]), np.array([1.0, 0.5]))


def test_band_embed_rejects_empty_t() -> None:
    with pytest.raises(ValueError, match="at least 1 interior threshold"):
        band_embed(np.array([0.0]), np.array([]))


def test_quantile_thresholds_properties() -> None:
    rng = np.random.default_rng(4)
    x_ref = rng.exponential(size=1000)
    t = quantile_thresholds(x_ref, n_bins=10)
    assert t.shape == (9,)
    assert np.all(np.diff(t) > 0.0)
    bins = band_embed(x_ref, t, beta=1000.0)
    hard_counts = bins.sum(axis=0)
    assert np.all(hard_counts > 0.5 * (x_ref.size / 10))
    assert np.all(hard_counts < 2.0 * (x_ref.size / 10))


def test_quantile_thresholds_handles_duplicates() -> None:
    """Heavily duplicated reference values must not produce non-increasing thresholds."""
    x_ref = np.concatenate([np.zeros(50), np.ones(50)])
    t = quantile_thresholds(x_ref, n_bins=8)
    assert np.all(np.diff(t) > 0.0)


def test_quantile_thresholds_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        quantile_thresholds(np.array([0.0, 1.0]), n_bins=1)
    with pytest.raises(ValueError, match="at least 2"):
        quantile_thresholds(np.array([1.0]), n_bins=3)
