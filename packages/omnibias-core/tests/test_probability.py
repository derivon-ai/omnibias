# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-core probability metadata: CDF detection + DKW deviation."""

from __future__ import annotations

import math

import pytest
from omnibias.core.probability import cdf_normalization, dkw_epsilon, is_cdf_activation
from omnibias.core.spec import ActivationSpec


def _spec(name: str, lo: float | None, hi: float | None) -> ActivationSpec[float]:
    return ActivationSpec(
        name=name,
        forward=lambda z: z,
        limit_neg_inf=lo,
        limit_pos_inf=hi,
    )


# ----- CDF classification / normalization -----------------------------------


def test_sigmoid_like_is_identity_cdf() -> None:
    scale, shift = cdf_normalization(_spec("sigmoid", 0.0, 1.0))  # type: ignore[misc]
    assert scale == pytest.approx(1.0)
    assert shift == pytest.approx(0.0)
    assert is_cdf_activation(_spec("sigmoid", 0.0, 1.0))


def test_tanh_like_rescales_to_cdf() -> None:
    scale, shift = cdf_normalization(_spec("tanh", -1.0, 1.0))  # type: ignore[misc]
    assert scale == pytest.approx(0.5)
    assert shift == pytest.approx(0.5)


def test_arctan_like_rescales_to_cdf() -> None:
    scale, shift = cdf_normalization(_spec("arctan", -math.pi / 2, math.pi / 2))  # type: ignore[misc]
    assert scale == pytest.approx(1.0 / math.pi)
    assert shift == pytest.approx(0.5)


@pytest.mark.parametrize(
    "lo,hi",
    [
        (0.0, 0.0),  # gaussian-like density (zero swing) -- not a CDF
        (0.0, None),  # exp-like (no finite right asymptote)
        (None, 1.0),  # no finite left asymptote
        (1.0, 0.0),  # decreasing saturations
    ],
)
def test_non_cdf_activations_return_none(lo: float | None, hi: float | None) -> None:
    assert cdf_normalization(_spec("x", lo, hi)) is None
    assert not is_cdf_activation(_spec("x", lo, hi))


def test_normalization_maps_limits_to_unit_interval() -> None:
    # A generic increasing activation with limits (-3, 5): F(L-)=0, F(L+)=1.
    scale, shift = cdf_normalization(_spec("x", -3.0, 5.0))  # type: ignore[misc]
    assert scale * (-3.0) + shift == pytest.approx(0.0)
    assert scale * (5.0) + shift == pytest.approx(1.0)


# ----- DKW deviation --------------------------------------------------------


def test_dkw_matches_massart_formula() -> None:
    exact = math.sqrt(math.log(2.0 / 0.05) / (2.0 * 100))
    eps = dkw_epsilon(100, 0.05)
    assert eps == pytest.approx(exact, rel=1e-9)
    assert eps >= exact  # rounded outward -> sound (conservative) test


def test_dkw_decreasing_in_n() -> None:
    assert dkw_epsilon(10) > dkw_epsilon(100) > dkw_epsilon(1000)


def test_dkw_increases_as_alpha_shrinks() -> None:
    assert dkw_epsilon(100, 0.10) < dkw_epsilon(100, 0.01)


@pytest.mark.parametrize("n", [0, -5])
def test_dkw_rejects_nonpositive_n(n: int) -> None:
    with pytest.raises(ValueError, match="n must be"):
        dkw_epsilon(n, 0.05)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_dkw_rejects_alpha_out_of_range(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        dkw_epsilon(100, alpha)
