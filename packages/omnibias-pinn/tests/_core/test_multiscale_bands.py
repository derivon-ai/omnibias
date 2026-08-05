# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reading multi-scale band scales off a measured power spectrum.

The contract: given a field whose energy sits at known frequencies,
:func:`suggest_frequency_bands` must return *those* frequencies as Fourier /
Mscale band scales -- so a multi-scale field can be configured from data instead
of from the guessed ``1, 2, 4, 8`` ladder.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn._core.diagnostics import power_spectrum_per_d
from omnibias.pinn._core.multiscale import (
    dominant_wavenumbers,
    geometric_bands,
    suggest_frequency_bands,
)


def _two_tone(low: int, high: int, *, n: int = 256, amp: float = 0.5) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.sin(2 * np.pi * low * x) + amp * np.sin(2 * np.pi * high * x)


# -- the headline: recovering known frequencies ------------------------------- #


@pytest.mark.parametrize(("low", "high"), [(3, 20), (1, 7), (2, 40)])
def test_two_tone_signal_recovers_both_frequencies(low, high):
    """Equal-energy splitting must put one band on each tone, exactly."""
    bands = suggest_frequency_bands(_two_tone(low, high)[None, :], L=1.0, n_bands=2)
    assert bands == pytest.approx((float(low), float(high)))


def test_scale_is_cycles_per_unit_length_not_bin_index():
    """The index -> scale conversion is ``s = j / L``; a non-unit period must show it."""
    n = 256
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    u = np.sin(5 * x)  # 5 cycles across a period of 2 pi
    (scale,) = suggest_frequency_bands(u[None, :], L=2 * np.pi, n_bands=1)
    assert scale == pytest.approx(5.0 / (2 * np.pi))


def test_single_band_is_the_energy_weighted_mean():
    """One band collapses to the centroid of the whole spectrum."""
    u = _two_tone(3, 20, amp=0.5)
    (scale,) = suggest_frequency_bands(u[None, :], L=1.0, n_bands=1)
    # Energies are 1 and 0.25 in amplitude^2, so the centroid sits nearer the low tone.
    expected = (3.0 * 1.0 + 20.0 * 0.25) / 1.25
    assert scale == pytest.approx(expected, rel=1e-6)


def test_bands_are_ascending_and_strictly_increasing():
    u = _two_tone(4, 30)
    bands = suggest_frequency_bands(u[None, :], L=1.0, n_bands=5)
    assert list(bands) == sorted(bands)
    assert len(set(bands)) == len(bands)


def test_more_bands_than_tones_does_not_duplicate():
    """A two-spike spectrum cannot support five distinct bands, and must not fake them."""
    bands = suggest_frequency_bands(_two_tone(3, 20)[None, :], L=1.0, n_bands=5)
    assert bands == pytest.approx((3.0, 20.0))


def test_time_averaging_over_snapshots():
    """The spectrum is averaged over the leading (time) axis, so a mixed history works."""
    lo = _two_tone(3, 3)  # only the low tone
    hi = _two_tone(20, 20)  # only the high tone
    grid = np.stack([lo, hi], axis=0)
    bands = suggest_frequency_bands(grid, L=1.0, n_bands=2)
    assert bands == pytest.approx((3.0, 20.0))


# -- multi-dimensional ------------------------------------------------------- #


def test_two_dimensional_field_uses_the_isotropic_wavenumber():
    n = 64
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = np.sin(2 * np.pi * 2 * xx) * np.sin(2 * np.pi * 2 * yy)
    (scale,) = suggest_frequency_bands(u[None], L=1.0, n_bands=1)
    # |k| = sqrt(2^2 + 2^2) = 2.83, binned to the nearest integer shell (3).
    assert scale == pytest.approx(round(np.sqrt(8.0)))


def test_tuple_period_is_reduced_to_its_mean():
    n = 64
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = np.sin(2 * np.pi * 3 * xx) + np.sin(2 * np.pi * 3 * yy)
    scalar = suggest_frequency_bands(u[None], L=2.0, n_bands=1)
    tup = suggest_frequency_bands(u[None], L=(1.0, 3.0), n_bands=1)
    assert scalar == pytest.approx(tup)


# -- dominant_wavenumbers directly ------------------------------------------- #


def test_dominant_wavenumbers_matches_the_spectrum_it_is_given():
    u = _two_tone(3, 20)
    power = power_spectrum_per_d(u[None, :], 1.0)
    assert dominant_wavenumbers(power, n_bands=2) == pytest.approx((3.0, 20.0))


def test_dc_bin_is_dropped_by_default():
    """A large mean must not drag the lowest band toward zero."""
    u = 100.0 + _two_tone(3, 20)
    power = power_spectrum_per_d(u[None, :], 1.0)
    assert power[0] > power[3]  # the DC bin genuinely dominates
    assert dominant_wavenumbers(power, n_bands=2) == pytest.approx((3.0, 20.0))
    with_dc = dominant_wavenumbers(power, n_bands=2, drop_dc=False)
    assert with_dc[0] == pytest.approx(0.0)


def test_flat_zero_spectrum_returns_a_single_zero_band():
    assert dominant_wavenumbers(np.zeros(8)) == (0.0,)


def test_zero_field_is_floored_at_min_scale():
    u = np.zeros(64)
    assert suggest_frequency_bands(u[None, :], L=1.0, min_scale=0.25) == (0.25,)


# -- the fallback ladder ------------------------------------------------------ #


def test_geometric_bands_is_the_literature_ladder():
    assert geometric_bands(4) == (1.0, 2.0, 4.0, 8.0)
    assert geometric_bands(3, base=3.0, start=2.0) == (2.0, 6.0, 18.0)


# -- argument validation ------------------------------------------------------ #


@pytest.mark.parametrize(
    ("fn", "kwargs", "match"),
    [
        (geometric_bands, {"n_bands": 0}, "n_bands must be >= 1"),
        (geometric_bands, {"n_bands": 2, "base": 1.0}, "base must be > 1"),
        (geometric_bands, {"n_bands": 2, "start": 0.0}, "start must be > 0"),
    ],
)
def test_geometric_bands_rejects_bad_arguments(fn, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fn(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_bands": 0}, "n_bands must be >= 1"),
        ({"min_scale": 0.0}, "min_scale must be > 0"),
        ({"L": 0.0}, "L must be > 0"),
    ],
)
def test_suggest_rejects_bad_arguments(kwargs, match):
    u = _two_tone(3, 20)[None, :]
    call = {"L": 1.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        suggest_frequency_bands(u, **call)


def test_dominant_wavenumbers_rejects_negative_power():
    with pytest.raises(ValueError, match="non-negative"):
        dominant_wavenumbers(np.array([1.0, -1.0]))


def test_dominant_wavenumbers_rejects_non_1d():
    with pytest.raises(ValueError, match="must be 1-D"):
        dominant_wavenumbers(np.zeros((2, 3)))
