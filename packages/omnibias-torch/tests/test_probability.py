# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable probability / measure operators (PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.torch.probability import (  # noqa: E402
    binned_calibration_error,
    cdf,
    empirical_band_mass,
    ks_statistic,
    model_band_mass,
    soft_histogram,
)

_F64 = torch.float64


@pytest.fixture(autouse=True)
def _default_f64():  # type: ignore[no-untyped-def]
    """Run these precision checks in float64 (restored afterwards)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _logistic_samples(n: int, loc: float = 0.0, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    return torch.tensor(loc + scale * np.log(u / (1.0 - u)), dtype=_F64)


# ----- model CDF / band mass ------------------------------------------------


def test_cdf_is_monotone_bounded_and_saturates() -> None:
    x = torch.linspace(-12.0, 12.0, 101, dtype=_F64)
    f = cdf(x, base="sigmoid")
    assert torch.all(f[1:] >= f[:-1])
    assert torch.all((f >= 0.0) & (f <= 1.0))
    assert float(f[0]) < 1e-4
    assert float(f[-1]) > 1.0 - 1e-4


def test_model_band_mass_matches_logistic_window() -> None:
    a, b, loc, scale = -0.5, 0.8, 0.2, 1.3
    got = model_band_mass(a, b, base="sigmoid", loc=loc, scale=scale)
    ref = torch.sigmoid(torch.tensor((b - loc) / scale, dtype=_F64)) - torch.sigmoid(
        torch.tensor((a - loc) / scale, dtype=_F64)
    )
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_tanh_and_sigmoid_normalizations_agree() -> None:
    # (tanh(u)+1)/2 == sigmoid(2u): tanh CDF at scale 1 == logistic CDF at scale 1/2.
    a, b, loc = -0.5, 0.8, 0.2
    mt = model_band_mass(a, b, base="tanh", loc=loc, scale=1.0)
    ms = model_band_mass(a, b, base="sigmoid", loc=loc, scale=0.5)
    assert torch.allclose(mt, ms, rtol=1e-10, atol=1e-12)


def test_non_cdf_base_is_refused() -> None:
    with pytest.raises(ValueError, match="not a CDF"):
        cdf(0.0, base="gaussian")


# ----- empirical measure + Glivenko-Cantelli --------------------------------


def test_empirical_band_mass_converges_to_model_mass() -> None:
    loc, scale = 0.3, 1.1
    samples = _logistic_samples(40000, loc=loc, scale=scale, seed=1)
    a, b = -0.7, 1.4
    emp = empirical_band_mass(samples, a, b, soft=False)
    mod = model_band_mass(a, b, base="sigmoid", loc=loc, scale=scale)
    assert abs(float(emp) - float(mod)) < 0.01


def test_soft_band_mass_approaches_hard_count() -> None:
    samples = _logistic_samples(5000, seed=2)
    a, b = -0.5, 0.7
    hard = empirical_band_mass(samples, a, b, soft=False)
    soft = empirical_band_mass(samples, a, b, soft=True, temperature=1e-3)
    assert abs(float(soft) - float(hard)) < 1e-2


def test_empirical_band_mass_temperature_guard() -> None:
    samples = _logistic_samples(16, seed=3)
    with pytest.raises(ValueError, match="temperature"):
        empirical_band_mass(samples, 0.0, 1.0, temperature=0.0)


# ----- calibration / goodness-of-fit ----------------------------------------


def test_calibration_error_small_when_model_matches_data() -> None:
    loc, scale = 0.0, 1.0
    samples = _logistic_samples(40000, loc=loc, scale=scale, seed=4)
    edges = torch.linspace(-4.0, 4.0, 17, dtype=_F64)
    matched = binned_calibration_error(
        samples, edges, base="sigmoid", loc=loc, scale=scale, soft=False
    )
    mismatched = binned_calibration_error(
        samples, edges, base="sigmoid", loc=2.0, scale=scale, soft=False
    )
    assert float(matched) < 0.05
    assert float(mismatched) > float(matched) + 0.2


def test_ks_statistic_separates_matched_from_mismatched() -> None:
    samples = _logistic_samples(4000, loc=0.0, scale=1.0, seed=5)
    matched = ks_statistic(samples, base="sigmoid", loc=0.0, scale=1.0)
    mismatched = ks_statistic(samples, base="sigmoid", loc=2.0, scale=1.0)
    assert float(matched) < 0.1
    assert float(mismatched) > 0.3


# ----- soft histogram (bank of bands) ---------------------------------------


def test_soft_histogram_normalizes_and_tracks_model() -> None:
    loc, scale = 0.0, 1.0
    samples = _logistic_samples(40000, loc=loc, scale=scale, seed=6)
    edges = torch.linspace(-4.0, 4.0, 17, dtype=_F64)
    hist = soft_histogram(samples, edges, temperature=0.05, normalize=True)
    assert torch.all(hist >= 0.0)
    assert float(hist.sum()) == pytest.approx(1.0, abs=1e-10)
    model_cdf = cdf(edges, base="sigmoid", loc=loc, scale=scale)
    model_bins = model_cdf[1:] - model_cdf[:-1]
    assert float((hist - model_bins).abs().max()) < 0.02


def test_soft_histogram_edges_guard() -> None:
    samples = _logistic_samples(16, seed=7)
    with pytest.raises(ValueError, match="length >= 2"):
        soft_histogram(samples, torch.tensor([0.0], dtype=_F64))


# ----- differentiability (calibration-as-loss) ------------------------------


def test_calibration_error_is_differentiable_in_location() -> None:
    samples = _logistic_samples(2000, loc=1.0, scale=1.0, seed=8)
    edges = torch.linspace(-4.0, 6.0, 21, dtype=_F64)
    loc = torch.tensor(0.0, dtype=_F64, requires_grad=True)
    ce = binned_calibration_error(
        samples, edges, base="sigmoid", loc=loc, scale=1.0, soft=True, temperature=0.3
    )
    ce.backward()
    assert loc.grad is not None
    assert torch.isfinite(loc.grad)
    # data sit at loc=1 but model at loc=0 -> a nonzero corrective gradient.
    assert float(loc.grad.abs()) > 1e-4


def test_empirical_band_mass_is_differentiable_in_edges() -> None:
    samples = _logistic_samples(2000, seed=9)
    low = torch.tensor(-0.5, dtype=_F64, requires_grad=True)
    high = torch.tensor(0.5, dtype=_F64, requires_grad=True)
    mass = empirical_band_mass(samples, low, high, soft=True, temperature=0.2)
    mass.backward()
    assert low.grad is not None and high.grad is not None
    # widening the band raises the mass: d/d(high) > 0, d/d(low) < 0.
    assert float(high.grad) > 0.0
    assert float(low.grad) < 0.0
