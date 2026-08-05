# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the causal time-marching schedule and driver.

Covers:

* :class:`TimeWindowSchedule` geometry: windows tile ``[t0, t1]`` exactly with
  and without overlap, bin edges partition a window, ``bin_index`` inverts the
  edges, and a single window is the whole interval.
* Epsilon annealing and the ``min w >= tolerance`` advance criterion.
* :func:`window_points`: shape, every bin equally populated, times inside their
  own bin, spatial coordinates inside the domain, seed determinism.
* :func:`slice_points`: constant time, matching spatial points across calls --
  the property that makes the warm start a plain value transfer.
* :class:`TimeMarcher`: window progression, the handoff landing at the *next*
  window's opening time under overlap, convergence bookkeeping, and the errors
  raised once marching is finished.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import (
    TimeMarcher,
    TimeWindowSchedule,
    slice_points,
    window_points,
)


@pytest.fixture
def cs() -> CoordinateSpec:
    return CoordinateSpec(
        ("x", "t"), domain=((0.0, 2.0), (0.0, 1.0)), time_axis="t"
    )


# ---------------- schedule geometry ---------------------------------


def test_windows_tile_the_interval_exactly():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=4)
    bounds = [s.window(k) for k in range(4)]
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == 1.0
    for (_, b), (a_next, _) in zip(bounds[:-1], bounds[1:], strict=True):
        assert b == pytest.approx(a_next)


def test_overlapping_windows_still_tile_but_share_a_seam():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=4, overlap=0.25)
    bounds = [s.window(k) for k in range(4)]
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == pytest.approx(1.0)
    for (_, b), (a_next, _) in zip(bounds[:-1], bounds[1:], strict=True):
        assert a_next < b  # genuinely overlapping
    assert s.width > TimeWindowSchedule(0.0, 1.0, n_windows=4).width


def test_one_window_is_the_whole_interval():
    s = TimeWindowSchedule(0.0, 3.0, n_windows=1)
    assert s.window(0) == (0.0, 3.0)
    assert s.width == pytest.approx(3.0)


def test_width_and_stride_satisfy_the_tiling_identity():
    s = TimeWindowSchedule(0.0, 5.0, n_windows=7, overlap=0.3)
    assert s.stride * (s.n_windows - 1) + s.width == pytest.approx(s.span)


def test_windows_iterator_matches_indexed_access():
    s = TimeWindowSchedule(1.0, 2.0, n_windows=3)
    assert list(s.windows()) == [s.window(k) for k in range(3)]
    assert len(s) == 3


def test_bin_edges_partition_the_window():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2, n_time_bins=8)
    edges = s.bin_edges(1)
    a, b = s.window(1)
    assert edges.shape == (9,)
    assert edges[0] == pytest.approx(a)
    assert edges[-1] == pytest.approx(b)
    assert np.all(np.diff(edges) > 0)
    centers = s.bin_centers(1)
    assert centers.shape == (8,)
    assert np.all((centers > edges[:-1]) & (centers < edges[1:]))


def test_bin_index_inverts_the_edges():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2, n_time_bins=5)
    centers = s.bin_centers(0)
    assert np.array_equal(s.bin_index(centers, 0), np.arange(5))


def test_bin_index_clips_the_closing_edge_into_the_last_bin():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=1, n_time_bins=4)
    a, b = s.window(0)
    assert int(s.bin_index(b, 0)) == 3
    assert int(s.bin_index(a, 0)) == 0


def test_out_of_range_window_raises():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2)
    with pytest.raises(IndexError, match="out of range"):
        s.window(2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t0": 1.0, "t1": 0.0}, "t1 > t0"),
        ({"n_windows": 0}, "n_windows"),
        ({"overlap": 1.0}, "overlap"),
        ({"n_time_bins": 0}, "n_time_bins"),
        ({"epsilon": -1.0}, "epsilon"),
        ({"epsilon_growth": 0.0}, "epsilon_growth"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"tolerance": 2.0}, "tolerance"),
    ],
)
def test_schedule_validates(kwargs, match):
    base = {"t0": 0.0, "t1": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        TimeWindowSchedule(**base)


# ---------------- the two adaptive knobs ----------------------------


def test_epsilon_anneals_geometrically():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=4, epsilon=0.5, epsilon_growth=2.0)
    assert [s.epsilon_at(k) for k in range(4)] == [0.5, 1.0, 2.0, 4.0]


def test_default_growth_holds_epsilon_fixed():
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3, epsilon=1.5)
    assert {s.epsilon_at(k) for k in range(3)} == {1.5}


def test_advance_criterion_is_the_smallest_weight():
    s = TimeWindowSchedule(0.0, 1.0, tolerance=0.1)
    assert s.is_converged([1.0, 0.8, 0.5, 0.1])
    assert not s.is_converged([1.0, 0.8, 0.5, 0.09])


def test_advance_criterion_rejects_empty_weights():
    s = TimeWindowSchedule(0.0, 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        s.is_converged([])


# ---------------- window_points -------------------------------------


def test_window_points_shape_is_bin_major(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2, n_time_bins=6)
    pts = window_points(cs, s, 0, per_bin=5)
    assert pts.shape == (6, 5, 2)


def test_every_time_bin_is_equally_populated(cs):
    """The reason for stratifying: an empty bin makes the causal weight junk."""
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2, n_time_bins=6)
    pts = window_points(cs, s, 1, per_bin=7)
    t = pts[:, :, 1]
    edges = s.bin_edges(1)
    for i in range(6):
        assert np.all((t[i] >= edges[i]) & (t[i] <= edges[i + 1]))


def test_window_points_stay_inside_the_spatial_domain(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3, n_time_bins=4)
    pts = window_points(cs, s, 2, per_bin=32)
    assert np.all((pts[:, :, 0] >= 0.0) & (pts[:, :, 0] <= 2.0))


def test_window_points_are_seed_deterministic_and_window_specific(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3, n_time_bins=4)
    a = window_points(cs, s, 0, per_bin=8, seed=3)
    b = window_points(cs, s, 0, per_bin=8, seed=3)
    c = window_points(cs, s, 0, per_bin=8, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_window_points_reshape_to_the_causal_layout(cs):
    """(n_bins, per_bin, D) -> flat for the field, back for the causal loss."""
    s = TimeWindowSchedule(0.0, 1.0, n_windows=1, n_time_bins=5)
    pts = window_points(cs, s, 0, per_bin=4)
    flat = pts.reshape(-1, 2)
    assert flat.shape == (20, 2)
    assert np.array_equal(flat.reshape(5, 4, 2), pts)


def test_window_points_validates(cs):
    s = TimeWindowSchedule(0.0, 1.0)
    with pytest.raises(ValueError, match="per_bin"):
        window_points(cs, s, 0, per_bin=0)


def test_marching_needs_a_time_axis_and_bounds():
    s = TimeWindowSchedule(0.0, 1.0)
    no_time = CoordinateSpec(("x", "y"), domain=((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="time axis"):
        window_points(no_time, s, 0, per_bin=2)
    no_bounds = CoordinateSpec(("x", "t"), time_axis="t")
    with pytest.raises(ValueError, match="bounds"):
        window_points(no_bounds, s, 0, per_bin=2)


# ---------------- slice_points --------------------------------------


def test_slice_points_sit_at_one_time(cs):
    pts = slice_points(cs, 0.375, n_points=9)
    assert pts.shape == (9, 2)
    assert np.all(pts[:, 1] == 0.375)


def test_same_seed_gives_the_same_spatial_points_at_different_times(cs):
    """The warm start is a value transfer, not an interpolation."""
    a = slice_points(cs, 0.25, n_points=16, seed=1)
    b = slice_points(cs, 0.75, n_points=16, seed=1)
    assert np.array_equal(a[:, 0], b[:, 0])
    assert not np.array_equal(a[:, 1], b[:, 1])


def test_slice_points_validates(cs):
    with pytest.raises(ValueError, match="n_points"):
        slice_points(cs, 0.0, n_points=0)


# ---------------- TimeMarcher ---------------------------------------


def test_marcher_walks_every_window_once(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3)
    m = TimeMarcher(cs, s, per_bin=4, n_slice=6)
    seen = []
    while not m.done:
        seen.append((m.window_index, m.bounds))
        m.advance()
    assert [k for k, _ in seen] == [0, 1, 2]
    assert seen[0][1][0] == 0.0
    assert seen[-1][1][1] == pytest.approx(1.0)


def test_marcher_exposes_the_annealed_epsilon(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3, epsilon=1.0, epsilon_growth=10.0)
    m = TimeMarcher(cs, s)
    got = []
    while not m.done:
        got.append(m.epsilon)
        m.advance()
    assert got == [1.0, 10.0, 100.0]


def test_handoff_lands_on_the_next_windows_opening_time(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3, overlap=0.3)
    m = TimeMarcher(cs, s, n_slice=5)
    handoff_t = float(m.handoff_points()[0, 1])
    m.advance(np.zeros(5))
    assert handoff_t == pytest.approx(m.initial_points()[0, 1])
    assert handoff_t < s.window(0)[1]  # strictly inside window 0, by overlap


def test_handoff_of_the_last_window_is_the_final_time(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2)
    m = TimeMarcher(cs, s, n_slice=4)
    m.advance()
    assert float(m.handoff_points()[0, 1]) == pytest.approx(1.0)


def test_handoff_and_initial_points_share_spatial_coordinates(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2)
    m = TimeMarcher(cs, s, n_slice=7)
    out = m.handoff_points()
    m.advance(np.arange(7.0))
    assert np.array_equal(out[:, 0], m.initial_points()[:, 0])


def test_warm_start_carries_the_handoff_values(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3)
    m = TimeMarcher(cs, s, n_slice=4)
    assert m.initial_values is None
    m.set_initial(np.zeros(4))
    assert np.array_equal(m.initial_values, np.zeros(4))
    m.advance(np.full(4, 3.0))
    assert np.array_equal(m.initial_values, np.full(4, 3.0))


def test_advance_without_values_keeps_the_previous_condition(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2)
    m = TimeMarcher(cs, s, n_slice=3)
    m.set_initial(np.ones(3))
    m.advance()
    assert np.array_equal(m.initial_values, np.ones(3))


def test_convergence_is_recorded_per_window(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=3, tolerance=0.5)
    m = TimeMarcher(cs, s)
    assert m.observe([1.0, 0.9]) is True
    m.advance()
    assert m.observe([1.0, 0.1]) is False
    m.advance()
    m.advance()  # never observed
    assert m.converged == (True, False, False)


def test_marcher_refuses_to_work_once_finished(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=1)
    m = TimeMarcher(cs, s)
    assert m.advance() is False
    assert m.done
    for call in (m.collocation, m.initial_points, m.handoff_points, m.advance):
        with pytest.raises(RuntimeError, match="marching finished"):
            call()


def test_marcher_collocation_matches_the_free_function(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2, n_time_bins=3)
    m = TimeMarcher(cs, s, per_bin=5, seed=11)
    m.advance()
    assert np.array_equal(
        m.collocation(), window_points(cs, s, 1, per_bin=5, seed=12)
    )


def test_marcher_validates_condition_lengths(cs):
    s = TimeWindowSchedule(0.0, 1.0, n_windows=2)
    m = TimeMarcher(cs, s, n_slice=4)
    with pytest.raises(ValueError, match="initial values"):
        m.set_initial(np.zeros(3))
    with pytest.raises(ValueError, match="handoff values"):
        m.advance(np.zeros(3))


def test_marcher_rejects_a_spec_without_a_time_axis():
    s = TimeWindowSchedule(0.0, 1.0)
    bad = CoordinateSpec(("x", "y"), domain=((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="time axis"):
        TimeMarcher(bad, s)


def test_marcher_repr_reports_position(cs):
    m = TimeMarcher(cs, TimeWindowSchedule(0.0, 1.0, n_windows=4))
    assert "window=0/4" in repr(m)
