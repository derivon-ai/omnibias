# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable probability / measure operators (JAX) + torch<->jax parity."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.probability import (  # noqa: E402
    binned_calibration_error,
    cdf,
    empirical_band_mass,
    ks_statistic,
    model_band_mass,
    soft_histogram,
)


def _logistic_samples(n: int, loc: float = 0.0, scale: float = 1.0, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    return jnp.asarray(loc + scale * np.log(u / (1.0 - u)), dtype=jnp.float64)


# ----- model CDF / band mass ------------------------------------------------


def test_cdf_is_monotone_bounded_and_saturates() -> None:
    x = jnp.linspace(-12.0, 12.0, 101, dtype=jnp.float64)
    f = cdf(x, base="sigmoid")
    assert bool(jnp.all(f[1:] >= f[:-1]))
    assert bool(jnp.all((f >= 0.0) & (f <= 1.0)))
    assert float(f[0]) < 1e-4
    assert float(f[-1]) > 1.0 - 1e-4


def test_model_band_mass_matches_logistic_window() -> None:
    a, b, loc, scale = -0.5, 0.8, 0.2, 1.3
    got = model_band_mass(a, b, base="sigmoid", loc=loc, scale=scale)
    ref = jax.nn.sigmoid((b - loc) / scale) - jax.nn.sigmoid((a - loc) / scale)
    assert bool(jnp.allclose(got, ref, rtol=1e-12, atol=1e-12))


def test_tanh_and_sigmoid_normalizations_agree() -> None:
    a, b, loc = -0.5, 0.8, 0.2
    mt = model_band_mass(a, b, base="tanh", loc=loc, scale=1.0)
    ms = model_band_mass(a, b, base="sigmoid", loc=loc, scale=0.5)
    assert bool(jnp.allclose(mt, ms, rtol=1e-10, atol=1e-12))


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
    edges = jnp.linspace(-4.0, 4.0, 17, dtype=jnp.float64)
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


# ----- soft histogram -------------------------------------------------------


def test_soft_histogram_normalizes_and_tracks_model() -> None:
    loc, scale = 0.0, 1.0
    samples = _logistic_samples(40000, loc=loc, scale=scale, seed=6)
    edges = jnp.linspace(-4.0, 4.0, 17, dtype=jnp.float64)
    hist = soft_histogram(samples, edges, temperature=0.05, normalize=True)
    assert bool(jnp.all(hist >= 0.0))
    assert float(hist.sum()) == pytest.approx(1.0, abs=1e-10)
    model_cdf = cdf(edges, base="sigmoid", loc=loc, scale=scale)
    model_bins = model_cdf[1:] - model_cdf[:-1]
    assert float(jnp.abs(hist - model_bins).max()) < 0.02


def test_soft_histogram_edges_guard() -> None:
    samples = _logistic_samples(16, seed=7)
    with pytest.raises(ValueError, match="length >= 2"):
        soft_histogram(samples, jnp.asarray([0.0], dtype=jnp.float64))


# ----- differentiability ----------------------------------------------------


def test_empirical_band_mass_is_differentiable_in_edges() -> None:
    samples = _logistic_samples(2000, seed=9)

    def mass(low: jax.Array, high: jax.Array) -> jax.Array:
        return empirical_band_mass(samples, low, high, soft=True, temperature=0.2)

    glow, ghigh = jax.grad(mass, argnums=(0, 1))(
        jnp.asarray(-0.5), jnp.asarray(0.5)
    )
    assert float(ghigh) > 0.0  # widening the band raises the mass
    assert float(glow) < 0.0


# ----- torch <-> jax bit-parity ---------------------------------------------


def test_cross_backend_parity_with_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.torch import probability as tp

    rng = np.random.default_rng(123)
    u = rng.uniform(size=3000)
    data = 0.2 + 1.4 * np.log(u / (1.0 - u))
    edges_np = np.linspace(-5.0, 5.0, 21)
    a, b, loc, scale = -0.8, 1.1, 0.2, 1.4

    js = jnp.asarray(data, dtype=jnp.float64)
    je = jnp.asarray(edges_np, dtype=jnp.float64)
    ts = torch.tensor(data, dtype=torch.float64)
    te = torch.tensor(edges_np, dtype=torch.float64)
    # float64 scalar band edges so torch does not fall back to its float32 default.
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)

    def close(jv, tv, rtol=1e-12, atol=1e-12) -> bool:  # type: ignore[no-untyped-def]
        return bool(np.allclose(np.asarray(jv), tv.detach().cpu().numpy(), rtol=rtol, atol=atol))

    for base in ("sigmoid", "tanh", "arctan"):
        assert close(
            cdf(je, base=base, loc=loc, scale=scale),
            tp.cdf(te, base=base, loc=loc, scale=scale),
        )
        assert close(
            model_band_mass(a, b, base=base, loc=loc, scale=scale),
            tp.model_band_mass(ta, tb, base=base, loc=loc, scale=scale),
        )
    assert close(
        empirical_band_mass(js, a, b, soft=True, temperature=0.1),
        tp.empirical_band_mass(ts, a, b, soft=True, temperature=0.1),
    )
    assert close(
        binned_calibration_error(js, je, base="sigmoid", loc=loc, scale=scale, soft=True),
        tp.binned_calibration_error(ts, te, base="sigmoid", loc=loc, scale=scale, soft=True),
    )
    assert close(
        ks_statistic(js, base="sigmoid", loc=loc, scale=scale),
        tp.ks_statistic(ts, base="sigmoid", loc=loc, scale=scale),
    )
    assert close(
        soft_histogram(js, je, temperature=0.1),
        tp.soft_histogram(ts, te, temperature=0.1),
    )
