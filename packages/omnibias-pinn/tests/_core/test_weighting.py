# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the backend-agnostic loss-weighting state machine.

Covers:

* :class:`LossWeighter` bookkeeping: EMA algebra, update cadence, clamping,
  the fresh-dict guarantee, and argument validation.
* :class:`GradNormWeighter`: the ``max|g_ref| / mean|g_k|`` target, the pinned
  reference weight, and convergence of the EMA to the target under a constant
  measurement.
* :class:`NTKWeighter`: the geometric-mean balance, agreement with the stateless
  ``ntk_balanced_loss`` weights at ``alpha = 0``, and equal traces giving
  equal weights.
* :class:`ConstantWeighter`: never moves, whatever it is fed.
* :meth:`LossWeighter.combine`: weighted sum, on plain floats and on numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.pinn._core.weighting import (
    ConstantWeighter,
    GradNormWeighter,
    GradStats,
    LossWeighter,
    NTKWeighter,
)


class _Fixed(LossWeighter):
    """Weighter with caller-supplied targets, for testing the base machinery."""

    def targets(self, stats):
        return stats


# ---------------- base machinery -----------------------------------


def test_ema_algebra_is_exactly_the_documented_update():
    w = _Fixed(["a", "b"], alpha=0.75, init=1.0)
    w.update({"a": 5.0, "b": 0.0})
    assert w["a"] == pytest.approx(0.75 * 1.0 + 0.25 * 5.0)
    assert w["b"] == pytest.approx(0.75 * 1.0 + 0.25 * 0.0)


def test_alpha_zero_takes_the_target_outright():
    w = _Fixed(["a"], alpha=0.0)
    w.update({"a": 3.0})
    assert w["a"] == 3.0


def test_alpha_one_freezes_the_weights():
    w = _Fixed(["a"], alpha=1.0)
    for _ in range(5):
        w.update({"a": 100.0})
    assert w["a"] == 1.0


def test_cadence_updates_on_the_first_call_then_every_n():
    w = _Fixed(["a"], alpha=0.0, every=3)
    seen = [w.update({"a": float(i)})["a"] for i in range(7)]
    # Steps 0, 3, 6 re-estimate; the rest hold.
    assert seen == [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0]
    assert w.step == 7
    assert w.n_updates == 3


def test_non_cadence_steps_ignore_the_stats_entirely():
    """So a caller may skip the expensive measurement and pass nothing."""
    w = _Fixed(["a"], alpha=0.0, every=2)
    w.update({"a": 2.0})
    assert w.update({})["a"] == 2.0  # would raise if the stats were read


def test_ema_converges_to_a_constant_target():
    w = _Fixed(["a"], alpha=0.9)
    for _ in range(400):
        w.update({"a": 7.0})
    assert w["a"] == pytest.approx(7.0, rel=1e-9)


def test_floor_and_ceiling_clamp_after_the_ema():
    w = _Fixed(["a"], alpha=0.0, init=1.0, floor=0.5, ceiling=2.0)
    assert w.update({"a": 0.0})["a"] == 0.5
    assert w.update({"a": 99.0})["a"] == 2.0


def test_non_finite_target_keeps_the_previous_weight():
    """A term with a vanishing gradient gives an infinite target."""
    w = _Fixed(["a"], alpha=0.5)
    w.update({"a": 3.0})
    before = w["a"]
    w.update({"a": math.inf})
    assert w["a"] == before
    w.update({"a": math.nan})
    assert w["a"] == before


def test_weights_property_returns_a_fresh_dict():
    w = _Fixed(["a"])
    got = w.weights
    got["a"] = 999.0
    assert w["a"] == 1.0


def test_repr_mentions_the_weights_and_the_step():
    w = _Fixed(["a"], every=4)
    text = repr(w)
    assert "a=" in text and "every=4" in text and "step=0" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 1.5}, "alpha"),
        ({"every": 0}, "every"),
        ({"floor": -1.0}, "floor"),
        ({"floor": 1.0, "ceiling": 0.5}, "ceiling"),
        ({"init": 9.0, "ceiling": 2.0}, "init"),
    ],
)
def test_constructor_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _Fixed(["a"], **kwargs)


def test_empty_and_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _Fixed([])
    with pytest.raises(ValueError, match="unique"):
        _Fixed(["a", "a"])


def test_blend_rejects_missing_and_unknown_keys():
    w = _Fixed(["a", "b"])
    with pytest.raises(ValueError, match="missing keys"):
        w.blend({"a": 1.0})
    with pytest.raises(ValueError, match="unknown keys"):
        w.blend({"a": 1.0, "b": 1.0, "c": 1.0})


def test_base_class_targets_is_abstract():
    with pytest.raises(NotImplementedError):
        LossWeighter(["a"]).update({"a": 1.0})


# ---------------- combine ------------------------------------------


def test_combine_is_the_weighted_sum():
    w = _Fixed(["a", "b"], alpha=0.0)
    w.update({"a": 2.0, "b": 3.0})
    assert w.combine({"a": 10.0, "b": 100.0}) == pytest.approx(2.0 * 10 + 3.0 * 100)


def test_combine_works_on_arrays():
    w = _Fixed(["a", "b"], alpha=0.0)
    w.update({"a": 2.0, "b": 0.5})
    got = w.combine({"a": np.array([1.0, 2.0]), "b": np.array([4.0, 8.0])})
    assert np.allclose(got, [2.0 * 1 + 0.5 * 4, 2.0 * 2 + 0.5 * 8])


def test_combine_rejects_a_key_mismatch():
    w = _Fixed(["a", "b"])
    with pytest.raises(ValueError, match="do not match"):
        w.combine({"a": 1.0})


# ---------------- GradNormWeighter ---------------------------------


def test_gradnorm_target_is_ref_max_over_term_mean():
    stats = {
        "pde": GradStats(max_abs=10.0, mean_abs=1.0),
        "bc": GradStats(max_abs=0.5, mean_abs=0.25),
    }
    w = GradNormWeighter(["pde", "bc"], reference="pde", alpha=0.0)
    got = w.update(stats)
    assert got["pde"] == 1.0
    assert got["bc"] == pytest.approx(10.0 / 0.25)


def test_gradnorm_pins_the_reference_weight_at_one_forever():
    stats = {
        "pde": GradStats(max_abs=3.0, mean_abs=2.0),
        "bc": GradStats(max_abs=1.0, mean_abs=1.0),
    }
    w = GradNormWeighter(["pde", "bc"], reference="pde", alpha=0.5)
    for _ in range(10):
        w.update(stats)
    assert w["pde"] == 1.0
    assert w["bc"] == pytest.approx(3.0, rel=1e-2)


def test_gradnorm_upweights_the_term_with_the_smaller_gradient():
    """The gradient pathology this exists to cure."""
    stats = {
        "pde": GradStats(max_abs=1.0, mean_abs=1.0),
        "big": GradStats(max_abs=1.0, mean_abs=10.0),
        "small": GradStats(max_abs=1.0, mean_abs=0.01),
    }
    w = GradNormWeighter(["pde", "big", "small"], reference="pde", alpha=0.0)
    got = w.update(stats)
    assert got["small"] > got["big"]
    assert got["small"] == pytest.approx(100.0, rel=1e-6)


def test_gradnorm_eps_guards_a_vanished_gradient():
    stats = {
        "pde": GradStats(max_abs=1.0, mean_abs=1.0),
        "dead": GradStats(max_abs=0.0, mean_abs=0.0),
    }
    w = GradNormWeighter(["pde", "dead"], reference="pde", alpha=0.0, eps=1e-8)
    got = w.update(stats)
    assert math.isfinite(got["dead"])
    assert got["dead"] == pytest.approx(1e8)


def test_gradnorm_rejects_an_unknown_reference_and_bad_eps():
    with pytest.raises(ValueError, match="reference"):
        GradNormWeighter(["a"], reference="nope")
    with pytest.raises(ValueError, match="eps"):
        GradNormWeighter(["a"], reference="a", eps=0.0)


def test_gradnorm_rejects_mismatched_stats():
    w = GradNormWeighter(["a", "b"], reference="a")
    with pytest.raises(ValueError, match="do not match"):
        w.update({"a": GradStats(1.0, 1.0)})


def test_gradstats_rejects_negative_magnitudes():
    with pytest.raises(ValueError, match="non-negative"):
        GradStats(max_abs=-1.0, mean_abs=1.0)


# ---------------- NTKWeighter --------------------------------------


def test_ntk_equal_traces_give_equal_weights():
    w = NTKWeighter(["a", "b", "c"], alpha=0.0)
    got = w.update({"a": 4.0, "b": 4.0, "c": 4.0})
    assert all(v == pytest.approx(1.0) for v in got.values())


def test_ntk_weights_have_unit_geometric_mean():
    w = NTKWeighter(["a", "b", "c"], alpha=0.0)
    got = w.update({"a": 1e-3, "b": 1.0, "c": 1e3})
    log_mean = sum(math.log(v) for v in got.values()) / 3
    assert log_mean == pytest.approx(0.0, abs=1e-12)


def test_ntk_matches_the_stateless_recipe_at_alpha_zero():
    traces = {"a": 0.25, "b": 9.0, "c": 2.0}
    log_t = {k: math.log(v) for k, v in traces.items()}
    mean_log = sum(log_t.values()) / 3
    expected = {k: math.exp(mean_log - log_t[k]) for k in traces}
    got = NTKWeighter(list(traces), alpha=0.0).update(traces)
    for k in traces:
        assert got[k] == pytest.approx(expected[k], rel=1e-14)


def test_ntk_floors_a_zero_trace_at_eps():
    w = NTKWeighter(["a", "b"], alpha=0.0, eps=1e-10)
    got = w.update({"a": 0.0, "b": 1.0})
    assert math.isfinite(got["a"]) and got["a"] > got["b"]


# ---------------- ConstantWeighter ---------------------------------


def test_constant_weighter_never_moves():
    w = ConstantWeighter({"pde": 1.0, "bc": 100.0})
    for _ in range(5):
        w.update({"pde": 0.0, "bc": 0.0})
    assert w.weights == {"pde": 1.0, "bc": 100.0}


def test_constant_weighter_still_combines():
    w = ConstantWeighter({"a": 2.0, "b": 3.0})
    assert w.combine({"a": 1.0, "b": 1.0}) == pytest.approx(5.0)


def test_constant_weighter_validates():
    with pytest.raises(ValueError, match="non-empty"):
        ConstantWeighter({})
    with pytest.raises(ValueError, match=">= 0"):
        ConstantWeighter({"a": -1.0})
