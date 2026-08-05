# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Causal time marching: windows, time-binned collocation, and the warm start.

:func:`~omnibias.pinn.torch.losses.causal_residual_loss` implements the Wang &
Perdikaris weight ``w_i = exp(-eps sum_{j<i} L_j)``, which stops a PINN from
fitting late times before early ones. On its own it is only half the recipe: it
reweights whatever collocation points it is handed, over the *whole* time
interval, for the whole run. The other half -- the part that actually makes long
horizons tractable -- is to solve on a short window, then march.

This module is that half, and it is deliberately backend-free numpy:

* :class:`TimeWindowSchedule` -- the geometry (window bounds, optional overlap,
  time bins) plus the two knobs that make marching adaptive: the annealed causal
  sharpness ``epsilon_at(k)`` and the advance criterion
  :meth:`~TimeWindowSchedule.is_converged`.
* :func:`window_points` -- collocation stratified *by time bin*, returned already
  shaped ``(n_bins, per_bin, D)`` so a residual computed on it drops straight
  into ``causal_residual_loss`` with no reshaping guesswork.
* :class:`TimeMarcher` -- the driver: current window, its epsilon, its points,
  and the **warm start**, i.e. the handoff of the trained field's values at the
  next window's initial time so window ``k+1`` inherits a real initial condition
  from window ``k`` instead of restarting from noise.

Nothing here trains anything: the caller owns the optimiser loop, and the
marcher only answers "which points, which epsilon, may I advance yet". That
keeps it usable with the torch backend, the jax backend, or the solver drivers.

Reference
---------
Wang, Sankaran & Perdikaris, *Respecting Causality is All You Need for Training
Physics-informed Neural Networks*, arXiv:2203.07404 (2022).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from omnibias.pinn._core.coords import CoordinateSpec


@dataclass(frozen=True)
class TimeWindowSchedule:
    """A ladder of time windows with annealed causal sharpness.

    Parameters
    ----------
    t0, t1:
        The full time interval to march across.
    n_windows:
        How many windows to split it into. ``1`` recovers a plain
        whole-interval causal solve.
    overlap:
        Fraction of a window shared with its predecessor, in ``[0, 1)``. A small
        overlap (0.05-0.2) lets window ``k+1`` re-fit the seam rather than
        inheriting the handoff error at exactly one instant, at the cost of
        narrower windows.
    n_time_bins:
        Time bins *within* a window; the resolution at which causality is
        enforced. This is the ``n_time_bins`` that
        :class:`~omnibias.pinn.torch.losses.CausalConfig` declares.
    epsilon:
        Causal sharpness for the first window.
    epsilon_growth:
        Multiplied into ``epsilon`` once per window. ``> 1`` sharpens the causal
        filter as marching proceeds, which is what you want: later windows start
        from a warm, already-consistent state, so they can afford to insist more
        strongly on temporal ordering. ``1.0`` (the default) holds it fixed.
    tolerance:
        The advance criterion. Because the causal weights are non-increasing,
        the last bin's weight measures how much of the window is "unlocked";
        advance once it reaches ``tolerance``. This is Wang et al.'s
        ``min_i w_i > delta`` rule.
    """

    t0: float
    t1: float
    n_windows: int = 4
    overlap: float = 0.0
    n_time_bins: int = 32
    epsilon: float = 1.0
    epsilon_growth: float = 1.0
    tolerance: float = 1e-2

    def __post_init__(self) -> None:
        if not self.t1 > self.t0:
            raise ValueError(f"need t1 > t0, got t0={self.t0}, t1={self.t1}")
        if self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows}")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.n_time_bins < 1:
            raise ValueError(f"n_time_bins must be >= 1, got {self.n_time_bins}")
        if self.epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if self.epsilon_growth <= 0.0:
            raise ValueError(f"epsilon_growth must be > 0, got {self.epsilon_growth}")
        if not 0.0 < self.tolerance <= 1.0:
            raise ValueError(f"tolerance must be in (0, 1], got {self.tolerance}")

    # -- geometry -------------------------------------------------------

    @property
    def span(self) -> float:
        """Total marched duration ``t1 - t0``."""
        return float(self.t1 - self.t0)

    @property
    def width(self) -> float:
        """Duration of one window.

        Solves ``stride * (n - 1) + width == span`` with
        ``stride = width * (1 - overlap)``, so the windows tile ``[t0, t1]``
        exactly however much they overlap.
        """
        return self.span / (1.0 + (self.n_windows - 1) * (1.0 - self.overlap))

    @property
    def stride(self) -> float:
        """Time advanced per window."""
        return self.width * (1.0 - self.overlap)

    def __len__(self) -> int:
        return self.n_windows

    def window(self, k: int) -> tuple[float, float]:
        """Bounds ``(a_k, b_k)`` of window ``k``."""
        k = self._check_index(k)
        a = self.t0 + k * self.stride
        b = self.t1 if k == self.n_windows - 1 else a + self.width
        return float(a), float(b)

    def windows(self) -> Iterator[tuple[float, float]]:
        """Iterate over every window's bounds."""
        for k in range(self.n_windows):
            yield self.window(k)

    def bin_edges(self, k: int) -> np.ndarray:
        """The ``n_time_bins + 1`` bin edges of window ``k``."""
        a, b = self.window(k)
        return np.linspace(a, b, self.n_time_bins + 1)

    def bin_centers(self, k: int) -> np.ndarray:
        """Midpoint of each time bin of window ``k``."""
        edges = self.bin_edges(k)
        return 0.5 * (edges[:-1] + edges[1:])

    def bin_index(self, t: np.ndarray | float, k: int) -> np.ndarray:
        """Which time bin of window ``k`` each time in ``t`` falls in.

        Times are clipped into ``[0, n_time_bins - 1]``, so a point on the
        window's closing edge lands in the last bin rather than out of range.
        """
        a, b = self.window(k)
        tt = np.asarray(t, dtype=float)
        raw = (tt - a) / (b - a) * self.n_time_bins
        return np.clip(np.floor(raw), 0, self.n_time_bins - 1).astype(int)

    # -- the two adaptive knobs -----------------------------------------

    def epsilon_at(self, k: int) -> float:
        """Causal sharpness for window ``k``: ``epsilon * growth^k``."""
        k = self._check_index(k)
        return float(self.epsilon * self.epsilon_growth**k)

    def is_converged(self, causal_weights: Sequence[float] | np.ndarray) -> bool:
        """Whether a window's causal weights permit advancing.

        The weights of :func:`causal_weights_from_per_bin` are non-increasing, so
        the smallest is the last bin's: once *it* has risen to ``tolerance`` the
        residual is being minimised across the whole window, not just its
        opening, and the window is done.
        """
        w = np.asarray(causal_weights, dtype=float).reshape(-1)
        if w.size == 0:
            raise ValueError("causal_weights must be non-empty")
        return bool(w.min() >= self.tolerance)

    def _check_index(self, k: int) -> int:
        if not 0 <= k < self.n_windows:
            raise IndexError(
                f"window {k} out of range for {self.n_windows} windows"
            )
        return int(k)


def _time_axis_index(coordinate_spec: CoordinateSpec) -> int:
    """The index of the time axis, or a helpful error."""
    if coordinate_spec.time_axis is None:
        raise ValueError(
            f"time marching needs a time axis; CoordinateSpec"
            f"{coordinate_spec.axes} has none. Pass time_axis= when building it."
        )
    return coordinate_spec.axis_index(coordinate_spec.time_axis)


def _spatial_bounds(
    coordinate_spec: CoordinateSpec,
) -> tuple[list[int], list[tuple[float, float]]]:
    """Indices and bounds of the non-time axes."""
    if coordinate_spec.domain is None:
        raise ValueError(
            "time marching needs axis bounds; build the CoordinateSpec with "
            "domain=((lo, hi), ...)"
        )
    t_idx = _time_axis_index(coordinate_spec)
    idx = [i for i in range(coordinate_spec.ndim) if i != t_idx]
    return idx, [coordinate_spec.domain[i] for i in idx]


def window_points(
    coordinate_spec: CoordinateSpec,
    schedule: TimeWindowSchedule,
    k: int,
    *,
    per_bin: int,
    seed: int = 0,
) -> np.ndarray:
    """Collocation points for window ``k``, stratified by time bin.

    Returns shape ``(n_time_bins, per_bin, D)`` rather than a flat ``(N, D)``,
    which is the whole point: a residual evaluated on
    ``points.reshape(-1, D)`` and reshaped back to ``(n_time_bins, per_bin)``
    is exactly the ``(n_t, ...spatial)`` layout ``causal_residual_loss``
    expects, with every bin equally populated. Uniform sampling over the window
    would leave the bin counts to chance, and an empty bin makes the cumulative
    causal weight meaningless.

    Times are drawn uniformly *within* each bin and spatial coordinates
    uniformly over the domain.
    """
    if per_bin < 1:
        raise ValueError(f"per_bin must be >= 1, got {per_bin}")
    t_idx = _time_axis_index(coordinate_spec)
    space_idx, space_bounds = _spatial_bounds(coordinate_spec)
    edges = schedule.bin_edges(k)
    n_bins = schedule.n_time_bins
    rng = np.random.default_rng(seed)

    out = np.empty((n_bins, per_bin, coordinate_spec.ndim), dtype=float)
    lo = edges[:-1].reshape(n_bins, 1)
    hi = edges[1:].reshape(n_bins, 1)
    out[:, :, t_idx] = rng.uniform(size=(n_bins, per_bin)) * (hi - lo) + lo
    for i, (a, b) in zip(space_idx, space_bounds, strict=True):
        out[:, :, i] = rng.uniform(a, b, size=(n_bins, per_bin))
    return out


def slice_points(
    coordinate_spec: CoordinateSpec,
    t: float,
    *,
    n_points: int,
    seed: int = 0,
) -> np.ndarray:
    """``(n_points, D)`` points on the constant-time slice ``t``.

    Used for both ends of a window: the initial condition of window ``k`` and
    the handoff sample that seeds window ``k + 1``. Seeded identically, the two
    calls return the same spatial points, so the handoff is a plain value
    transfer with no interpolation.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    t_idx = _time_axis_index(coordinate_spec)
    space_idx, space_bounds = _spatial_bounds(coordinate_spec)
    rng = np.random.default_rng(seed)
    out = np.empty((n_points, coordinate_spec.ndim), dtype=float)
    out[:, t_idx] = float(t)
    for i, (a, b) in zip(space_idx, space_bounds, strict=True):
        out[:, i] = rng.uniform(a, b, size=n_points)
    return out


class TimeMarcher:
    """Drive a solve window by window, warm-starting each from the last.

    The loop this is built for::

        marcher = TimeMarcher(cs, schedule, per_bin=64, n_slice=128)
        marcher.set_initial(u0_values)              # the true IC at t0
        while not marcher.done:
            pts, eps = marcher.collocation(), marcher.epsilon
            ic_x, ic_u = marcher.initial_points(), marcher.initial_values
            for _ in range(steps):
                ... train, obtaining per-bin causal weights ...
            marcher.observe(weights)
            marcher.advance(field_values_at(marcher.handoff_points()))

    The marcher owns no tensors and no optimiser -- it answers *which points,
    which epsilon, may I advance yet* and carries the handoff values across the
    seam. Everything else is the caller's.

    Parameters
    ----------
    coordinate_spec:
        Must have a time axis and explicit ``domain`` bounds.
    schedule:
        The window ladder.
    per_bin:
        Collocation points per time bin, per window.
    n_slice:
        Points on each constant-time slice (initial condition and handoff).
    seed:
        Base seed. Window ``k`` samples with ``seed + k`` so windows never
        reuse a draw, while the slice points are fixed across all windows so
        the handoff needs no interpolation.
    """

    def __init__(
        self,
        coordinate_spec: CoordinateSpec,
        schedule: TimeWindowSchedule,
        *,
        per_bin: int = 32,
        n_slice: int = 64,
        seed: int = 0,
    ) -> None:
        _spatial_bounds(coordinate_spec)  # fail fast on a malformed spec
        self.coordinate_spec = coordinate_spec
        self.schedule = schedule
        self.per_bin = int(per_bin)
        self.n_slice = int(n_slice)
        self.seed = int(seed)
        self._k = 0
        self._initial: np.ndarray | None = None
        self._converged: list[bool] = []
        self._pending = False

    # -- where we are ---------------------------------------------------

    @property
    def window_index(self) -> int:
        """Index of the window currently being solved."""
        return self._k

    @property
    def done(self) -> bool:
        """Whether every window has been marched."""
        return self._k >= self.schedule.n_windows

    @property
    def bounds(self) -> tuple[float, float]:
        """Bounds of the current window."""
        return self.schedule.window(self._k)

    @property
    def epsilon(self) -> float:
        """Annealed causal sharpness of the current window."""
        return self.schedule.epsilon_at(self._k)

    @property
    def converged(self) -> tuple[bool, ...]:
        """Whether each already-marched window met the advance criterion.

        Recorded rather than enforced: a window that hit its step budget without
        converging is a fact the caller should be able to report, not a reason
        to raise.
        """
        return tuple(self._converged)

    def __repr__(self) -> str:
        a, b = (
            self.schedule.window(min(self._k, self.schedule.n_windows - 1))
            if self.schedule.n_windows
            else (self.schedule.t0, self.schedule.t1)
        )
        return (
            f"TimeMarcher(window={self._k}/{self.schedule.n_windows}, "
            f"t=[{a:.4g}, {b:.4g}], epsilon="
            f"{self.schedule.epsilon_at(min(self._k, self.schedule.n_windows - 1)):.4g})"
        )

    # -- points ---------------------------------------------------------

    def collocation(self) -> np.ndarray:
        """``(n_time_bins, per_bin, D)`` collocation points for this window."""
        self._require_active()
        return window_points(
            self.coordinate_spec,
            self.schedule,
            self._k,
            per_bin=self.per_bin,
            seed=self.seed + self._k,
        )

    def initial_points(self) -> np.ndarray:
        """``(n_slice, D)`` points at this window's opening time."""
        self._require_active()
        return slice_points(
            self.coordinate_spec,
            self.bounds[0],
            n_points=self.n_slice,
            seed=self.seed,
        )

    def handoff_points(self) -> np.ndarray:
        """``(n_slice, D)`` points where the next window's condition is sampled.

        The *next* window's opening time, not this one's closing time -- with
        ``overlap > 0`` those differ, and it is the next window's opening that
        needs a value. On the last window the two coincide at ``t1``.
        """
        self._require_active()
        t = (
            self.bounds[1]
            if self._k == self.schedule.n_windows - 1
            else self.schedule.window(self._k + 1)[0]
        )
        return slice_points(
            self.coordinate_spec, t, n_points=self.n_slice, seed=self.seed
        )

    # -- the warm start -------------------------------------------------

    @property
    def initial_values(self) -> np.ndarray | None:
        """Condition values at :meth:`initial_points`, or ``None`` if unset."""
        return self._initial

    def set_initial(self, values: np.ndarray) -> None:
        """Seed the first window with the true initial condition."""
        v = np.asarray(values, dtype=float)
        if v.shape[0] != self.n_slice:
            raise ValueError(
                f"expected {self.n_slice} initial values (one per slice point), "
                f"got {v.shape[0]}"
            )
        self._initial = v

    def observe(self, causal_weights: Sequence[float] | np.ndarray) -> bool:
        """Record whether this window met the advance criterion, and report it.

        Call it as often as you like during a window; the latest verdict is the
        one :meth:`advance` files into :attr:`converged`.
        """
        self._require_active()
        self._pending = self.schedule.is_converged(causal_weights)
        return self._pending

    def advance(self, handoff_values: np.ndarray | None = None) -> bool:
        """Move to the next window, carrying ``handoff_values`` as its condition.

        ``handoff_values`` are the trained field evaluated at
        :meth:`handoff_points` -- the warm start. Returns ``False`` once the
        last window has been advanced past.
        """
        self._require_active()
        self._converged.append(self._pending)
        self._pending = False
        if handoff_values is not None:
            v = np.asarray(handoff_values, dtype=float)
            if v.shape[0] != self.n_slice:
                raise ValueError(
                    f"expected {self.n_slice} handoff values (one per slice "
                    f"point), got {v.shape[0]}"
                )
            self._initial = v
        self._k += 1
        return not self.done

    def _require_active(self) -> None:
        if self.done:
            raise RuntimeError(
                f"marching finished: all {self.schedule.n_windows} windows done"
            )


__all__ = [
    "TimeMarcher",
    "TimeWindowSchedule",
    "slice_points",
    "window_points",
]
