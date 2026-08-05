# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Point observations of a solution component (the data an inverse solve fits).

An inverse problem needs a third loss term beside the PDE residual and the
boundary / initial conditions: measurements of the solution itself. Without them
the coefficients are unidentifiable -- a PDE residual alone is satisfied by many
``(field, coefficient)`` pairs, so the recovered value would be whatever the
optimiser drifted to.

The arrays are numpy so this stays backend-free like the rest of ``_core``; the
driver converts them to the field's dtype / device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Observations:
    """Measured values of one component at scattered points.

    Parameters
    ----------
    component
        Which solution component was measured.
    coords
        ``(M, ndim)`` measurement locations, in the domain's axis order.
    values
        ``(M,)`` measured values.
    weight
        Multiplies this block's mean-squared misfit. Raise it when the data are
        trustworthy relative to the residual, lower it for noisy data -- it is the
        inverse-noise-variance knob of the equivalent MAP estimate.
    """

    component: str
    coords: np.ndarray
    values: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if coords.ndim != 2:
            raise ValueError(
                f"observation coords must be (M, ndim), got shape {coords.shape}"
            )
        if coords.shape[0] != values.shape[0]:
            raise ValueError(
                f"{self.component!r}: {coords.shape[0]} coords but "
                f"{values.shape[0]} values"
            )
        if coords.shape[0] == 0:
            raise ValueError(f"{self.component!r}: no observations given")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.component!r}: observations must be finite")
        if self.weight <= 0.0:
            raise ValueError(f"{self.component!r}: weight must be > 0")
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return int(self.values.shape[0])


def check_observations(
    observations: Sequence[Observations], *, components: Sequence[str], ndim: int
) -> tuple[Observations, ...]:
    """Validate observations against a system's components and dimension."""
    if not observations:
        raise ValueError(
            "an inverse solve needs observations of the solution; the PDE residual "
            "alone does not identify the coefficients"
        )
    known = set(components)
    for obs in observations:
        if obs.component not in known:
            raise ValueError(
                f"observations reference unknown component {obs.component!r}; "
                f"this system has {tuple(components)!r}"
            )
        if obs.coords.shape[1] != ndim:
            raise ValueError(
                f"{obs.component!r}: observation coords have "
                f"{obs.coords.shape[1]} columns but the domain is {ndim}-D"
            )
    return tuple(observations)


def sample_observations(
    solution: Any,
    component: str,
    coords: np.ndarray,
    *,
    noise: float = 0.0,
    seed: int = 0,
    weight: float = 1.0,
) -> Observations:
    """Sample a fitted solution to build synthetic :class:`Observations`.

    This is the honest way to set up a recovery study: solve the *forward* problem
    at known coefficients, measure it at a handful of points, optionally add
    Gaussian noise of standard deviation ``noise``, then check whether the inverse
    solve gets the coefficients back.
    """
    pts = np.asarray(coords, dtype=float)
    values = np.asarray(solution.evaluate(pts, component).detach().cpu().numpy())
    values = values.reshape(-1).astype(float)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        values = values + rng.normal(0.0, noise, size=values.shape)
    return Observations(
        component=component, coords=pts, values=values, weight=weight
    )


__all__ = ["Observations", "check_observations", "sample_observations"]
