# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for omnibias.core.mollifier (theory 01-05 G1/G2/G3)."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.mollifier import (
    MollifierSpec,
    design_order,
    is_admissible,
    moments,
    tail_bound,
    true_outside_mass,
)
from omnibias.core.multipack import PackSpec


def test_logistic_and_sech_second_moments() -> None:
    """G1: closed-form M_2 matches logistic pi^2/3 and sech-type pi^2/12."""
    logi = MollifierSpec("sigmoid", 1.0, (PackSpec(0, 0.0),))
    sech = MollifierSpec("tanh", 1.0, (PackSpec(0, 0.0),))
    gauss = MollifierSpec("gaussian", 1.0, (PackSpec(0, 0.0),))
    m_logi = moments(logi, 2)
    m_sech = moments(sech, 2)
    m_gauss = moments(gauss, 2)
    assert m_logi[0] == pytest.approx(1.0)
    assert m_logi[1] == pytest.approx(0.0)
    assert m_logi[2] == pytest.approx(math.pi**2 / 3.0, rel=1e-12)
    assert m_sech[2] == pytest.approx(math.pi**2 / 12.0, rel=1e-12)
    assert m_gauss[2] == pytest.approx(1.0, rel=1e-12)
    assert m_gauss[0] == pytest.approx(1.0)


def test_scaled_moments_homogeneous() -> None:
    spec = MollifierSpec("tanh", 0.1, (PackSpec(0, 0.0),))
    m = moments(spec, 2)
    assert m[2] == pytest.approx((0.1**2) * (math.pi**2 / 12.0), rel=1e-12)


def test_order4_richardson_worked_example() -> None:
    """Spec §5: (4/3) phi_eps - (1/3) phi_{2 eps} kills M_2 and is exact on u^2."""
    spec = design_order("tanh", 4, scale=0.1)
    assert spec.pack_scales == pytest.approx((1.0, 2.0))
    assert spec.packs[0].weight == pytest.approx(4.0 / 3.0)
    assert spec.packs[1].weight == pytest.approx(-1.0 / 3.0)
    m = moments(spec, 4)
    assert m[0] == pytest.approx(1.0, rel=1e-12)
    assert abs(m[1]) <= 1e-12
    assert abs(m[2]) <= 1e-12
    assert abs(m[3]) <= 1e-12
    # Convolution against u^2 at 0 is M_2 == 0 (order-4 exact on degree < 4).
    assert spec.order == 4
    assert spec.is_positive is False


def test_order2_is_positive_density() -> None:
    spec = design_order("gaussian", 2, scale=1.0)
    assert spec.order == 2
    assert spec.is_positive is True
    assert is_admissible(spec, form_order=2) is True


def test_polynomial_reproduction_and_rate() -> None:
    """G2: degree < m reproduced; smoothing error on u^m is O(eps^m)."""
    # Order-2 sech: (phi_eps * u^2)(0) = M_2 = eps^2 * pi^2/12.
    errs: list[float] = []
    for eps in (0.2, 0.1, 0.05, 0.025):
        spec = design_order("tanh", 2, scale=eps)
        m = moments(spec, 2)
        # p(u)=1 -> M_0; p(u)=u -> M_1 ~ 0; error on u^2 is |M_2 - 0|.
        assert abs(m[0] - 1.0) <= 1e-12
        assert abs(m[1]) <= 1e-12
        errs.append(abs(m[2]))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    # Three halvings: each ratio ~ 4 for O(eps^2).
    assert all(3.5 <= r <= 4.5 for r in ratios)

    spec4 = design_order("tanh", 4, scale=0.05)
    m4 = moments(spec4, 3)
    assert all(abs(v - (1.0 if j == 0 else 0.0)) <= 1e-12 for j, v in enumerate(m4))


def test_tail_bound_sound_grid_and_sample() -> None:
    """G3: Interval contains the true outside mass; zero undercovers."""
    rng = random.Random(0)
    specs = [
        design_order("tanh", 2, scale=0.25),
        design_order("sigmoid", 2, scale=0.5),
        design_order("gaussian", 2, scale=0.4),
        design_order("tanh", 4, scale=0.3),
    ]
    widths = [0.5, 1.0, 1.5, 2.0, 3.0] + [rng.uniform(0.4, 4.0) for _ in range(20)]
    violations = 0
    n = 0
    for spec in specs:
        for w in widths:
            enclosed = tail_bound(spec, half_width=w)
            truth = true_outside_mass(spec, half_width=w)
            n += 1
            if not enclosed.contains(truth):
                violations += 1
    assert n >= 100
    assert violations == 0


def test_admissible_rejects_bad_form_order() -> None:
    spec = design_order("gaussian", 2)
    with pytest.raises(ValueError, match="form_order"):
        is_admissible(spec, form_order=-1)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        MollifierSpec("relu", 1.0, (PackSpec(0, 0.0),))
    with pytest.raises(ValueError, match="positive"):
        MollifierSpec("tanh", 0.0, (PackSpec(0, 0.0),))
    with pytest.raises(ValueError, match="even"):
        design_order("tanh", 3)
    with pytest.raises(ValueError, match="half_width"):
        tail_bound(design_order("tanh", 2), half_width=-1.0)
