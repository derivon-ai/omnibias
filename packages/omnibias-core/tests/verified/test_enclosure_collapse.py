# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 01-14: Enclosure Collapse algebra and the Width Law.

Enclosure Collapse is ``width -> 0`` of a sound enclosure: a point plus a proof,
not a derivative and not a 0/1 step.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.verified.enclosure_collapse import (
    CRITICAL_POINT_FACTOR,
    MEAN_VALUE_FACTOR,
    RecommendedAction,
    diagnose_width,
    honesty_payload,
    measured_critical_width,
    measured_mean_value_width,
    measured_taylor_remainder,
    predicted_width,
    rounding_floor_witness,
    two_ulp,
    width_law,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import WidthBudget
from omnibias.core.verified.sigma import sigma_tower_interval


def _true_sigma(name: str, x: float, n: int) -> float:
    return float(sigma_tower_interval(name, Interval.point(x), n)[n].mid)


def test_honesty_names_three_limits() -> None:
    payload = honesty_payload()
    assert payload["founding_bias_collapse"] is True
    assert payload["temperature_collapse"] is False
    assert payload["enclosure_collapse"] is True
    assert payload["width_to_zero"] is True
    assert payload["not_a_derivative"] is True
    assert payload["not_a_01_step"] is True
    assert payload["theorem_prover_verified"] is False


def test_g2_lemma_floor_two_ulp() -> None:
    witness = rounding_floor_witness()
    assert witness.width >= two_ulp(0.0)
    mid = witness.mid
    assert witness.contains(1.0 / 3.0) or abs(mid - 1.0 / 3.0) <= witness.width
    assert witness.lo < witness.hi


def test_g2_no_clamp_api() -> None:
    names = dir(__import__("omnibias.core.verified.enclosure_collapse", fromlist=["*"]))
    forbidden = [n for n in names if "clamp" in n.lower() or n in {"identify_endpoints", "force_point"}]
    assert forbidden == []


@pytest.mark.parametrize("activation,n,center", [("sigmoid", 0, 0.5), ("tanh", 0, 0.4)])
def test_g1_width_law_mean_value_limit(activation: str, n: int, center: float) -> None:
    law = width_law(activation, n, center, kind="mean_value")
    assert law.exponent == 1
    assert law.kind == "mean_value"
    errors: list[float] = []
    for r in (1e-2, 1e-3, 1e-4):
        enc = measured_mean_value_width(activation, n, center, r)
        ratio = enc.width / (MEAN_VALUE_FACTOR * r)
        errors.append(abs(ratio - law.predicted_leading.mid))
        pred = predicted_width(law, r)
        assert pred.lo >= 0.0
    assert errors[-1] < 5e-3
    assert errors[-1] <= errors[0]


@pytest.mark.parametrize("activation,n,center", [("sigmoid", 1, 0.0), ("tanh", 1, 0.0)])
def test_g1_width_law_critical_point_factor(activation: str, n: int, center: float) -> None:
    law = width_law(activation, n, center, kind="critical_point")
    assert law.exponent == 2
    assert CRITICAL_POINT_FACTOR == 1
    r = 1e-3
    enc = measured_critical_width(activation, n, center, r)
    ratio = enc.width / (r**2)
    assert abs(ratio - law.predicted_leading.mid) / max(law.predicted_leading.mid, 1e-12) < 0.25


def test_g1_width_law_taylor_remainder() -> None:
    activation, n, center, order = "sigmoid", 0, 0.3, 2
    law = width_law(activation, n, center, kind="taylor_remainder", order=order)
    assert law.exponent == order + 1
    r = 5e-3
    rem = measured_taylor_remainder(activation, n, center, r, order)
    pred = predicted_width(law, r)
    assert rem.width > 0.0
    assert pred.contains(rem.width) or abs(rem.width - pred.mid) / max(pred.mid, 1e-18) < 0.5


def test_width_law_enclosures_contain_grid_and_random() -> None:
    activation, n, center, r = "tanh", 0, 0.25, 0.05
    enc = measured_mean_value_width(activation, n, center, r)
    grid = [center - r + 2.0 * r * i / 40.0 for i in range(41)]
    rng = random.Random(0)
    sample = [rng.uniform(center - r, center + r) for _ in range(32)]
    for x in grid + sample:
        assert enc.contains(_true_sigma(activation, x, n))


def test_diagnose_width_maps_each_dominant() -> None:
    expected = {
        "truncation": "raise_order",
        "wrapping": "subdivide",
        "jacobian": "shrink_step",
        "rounding": "stop_floor",
    }
    for dominant, action in expected.items():
        parts = {"truncation": 0.0, "jacobian": 0.0, "wrapping": 0.0, "rounding": 0.0}
        parts[dominant] = 1.0
        rec = diagnose_width(WidthBudget(**parts))
        assert isinstance(rec, RecommendedAction)
        assert rec.dominant == dominant
        assert rec.action == action


def test_mean_value_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError, match="positive"):
        measured_mean_value_width("sigmoid", 0, 0.0, 0.0)
