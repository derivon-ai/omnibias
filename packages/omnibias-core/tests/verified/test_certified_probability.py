# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified probabilities: rigorous CDF / band-mass enclosures + DKW GoF."""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.probability import (
    CertifiedGoFResult,
    band_mass_enclosure,
    cdf_enclosure,
    certified_gof,
    empirical_cdf,
)


def _true_cdf(name: str, x: float, loc: float, scale: float) -> float:
    u = (x - loc) / scale
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-u))
    if name == "tanh":
        return 0.5 * math.tanh(u) + 0.5
    if name == "arctan":
        return math.atan(u) / math.pi + 0.5
    raise AssertionError(name)


def _logistic_quantiles(n: int, loc: float = 0.0, scale: float = 1.0) -> list[float]:
    """Deterministic logistic samples = its own quantiles (no RNG, no flakiness)."""
    return [loc + scale * math.log(u / (1.0 - u)) for u in ((i + 0.5) / n for i in range(n))]


# ----- enclosure validity ---------------------------------------------------


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "arctan"])
@pytest.mark.parametrize("x", [-4.0, -1.0, -0.3, 0.0, 0.7, 2.5])
def test_cdf_enclosure_contains_truth(name: str, x: float) -> None:
    loc, scale = 0.4, 1.7
    enc = cdf_enclosure(name, x, loc=loc, scale=scale)
    true = _true_cdf(name, x, loc, scale)
    assert enc.lo <= true <= enc.hi
    assert 0.0 <= enc.lo <= enc.hi <= 1.0
    assert enc.width < 1e-9  # tight (mpmath) or a few libm ulps


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "arctan"])
def test_band_mass_enclosure_contains_truth(name: str) -> None:
    a, b, loc, scale = -0.6, 1.2, 0.1, 0.9
    enc = band_mass_enclosure(name, a, b, loc=loc, scale=scale)
    true = _true_cdf(name, b, loc, scale) - _true_cdf(name, a, loc, scale)
    assert enc.lo <= true <= enc.hi
    assert 0.0 <= enc.lo <= enc.hi <= 1.0


def test_cdf_enclosure_is_monotone() -> None:
    lo = cdf_enclosure("sigmoid", -1.0)
    hi = cdf_enclosure("sigmoid", 1.0)
    assert hi.lo > lo.hi


def test_unsupported_activation_raises() -> None:
    with pytest.raises(NotImplementedError, match="no certified CDF"):
        cdf_enclosure("gaussian", 0.0)


def test_band_mass_rejects_reversed_limits() -> None:
    with pytest.raises(ValueError, match="a <= b"):
        band_mass_enclosure("sigmoid", 1.0, -1.0)


def test_cdf_enclosure_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be"):
        cdf_enclosure("sigmoid", 0.0, scale=0.0)


# ----- empirical CDF --------------------------------------------------------


def test_empirical_cdf_counts() -> None:
    s = [0.0, 1.0, 2.0, 3.0]
    assert empirical_cdf(s, -1.0) == 0.0
    assert empirical_cdf(s, 1.5) == 0.5
    assert empirical_cdf(s, 3.0) == 1.0


def test_empirical_cdf_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        empirical_cdf([], 0.0)


# ----- certified DKW goodness-of-fit ----------------------------------------


def test_certified_gof_accepts_matched_model() -> None:
    samples = _logistic_quantiles(400, loc=0.0, scale=1.0)
    res = certified_gof(samples, "sigmoid", loc=0.0, scale=1.0, alpha=0.05)
    assert isinstance(res, CertifiedGoFResult)
    assert not res.rejected
    # quantile samples sit essentially on the model CDF -> tiny certified gap.
    assert res.certified_ks_lower_bound < res.epsilon


def test_certified_gof_rejects_location_shift() -> None:
    samples = _logistic_quantiles(400, loc=0.0, scale=1.0)
    res = certified_gof(samples, "sigmoid", loc=3.0, scale=1.0, alpha=0.05)
    assert res.rejected
    # the rejection is certified: the lower bound provably clears the threshold.
    assert res.certified_ks_lower_bound > res.epsilon


def test_certified_gof_rejects_scale_mismatch() -> None:
    samples = _logistic_quantiles(400, loc=0.0, scale=3.0)
    res = certified_gof(samples, "sigmoid", loc=0.0, scale=1.0, alpha=0.05)
    assert res.rejected


def test_certified_gof_records_metadata() -> None:
    samples = _logistic_quantiles(50)
    res = certified_gof(samples, "tanh", loc=0.0, scale=1.0)
    assert res.n == 50
    assert res.backend in ("mpmath", "libm_fallback", "libm")
    assert 0.0 <= res.certified_ks_lower_bound <= 1.0


def test_certified_gof_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        certified_gof([], "sigmoid")
