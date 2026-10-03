# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Concentration enclosure for a sampled expectation (pure stdlib).

The Tier C poly-Laplacian estimator in :mod:`omnibias.jax.laplacian` /
:mod:`omnibias.torch.laplacian` reports ``Delta^k f`` as the mean of a
sampled directional derivative, times a closed-form normaliser. That mean is
a Monte Carlo estimate, not a value the omnibias closed-form tower produces
exactly -- so it needs its own honesty story, distinct from
:class:`~omnibias.core.verified.interval.Interval` arithmetic.

:func:`hoeffding_enclosure` supplies that story: given ``n`` i.i.d. samples of
a bounded random variable and a *declared* range ``[lo, hi]``, Hoeffding's
inequality gives

.. math::

    \Pr\!\left[\left|\bar X - \mathbb{E}[X]\right| > t\right]
        \le 2 \exp\!\left(-\frac{2 n t^2}{(hi - lo)^2}\right),

so inverting for a target failure probability ``delta`` yields an interval
around the sample mean that contains the true expectation with probability
``>= 1 - delta``.

Honesty note
-------------
This is a genuine, exactly-proved tail bound -- *conditional* on the premise
that every draw of the random variable truly lies in ``[lo, hi]``. It is
**not** the same grade of guarantee as :class:`~omnibias.core.verified.interval.Interval`
arithmetic, which encloses with certainty (no failure probability, no
distributional premise). Call the result a *probabilistic* enclosure, never a
``theorem_prover_verified`` or otherwise sound one. As a runtime sanity guard
(not a proof that unseen draws respect the range), every supplied sample is
checked against ``[lo, hi]`` at call time and a violation raises immediately
rather than silently mislabelling the result as sound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, inf, isfinite, log, nextafter, sqrt

from omnibias.core.verified.interval import Interval


@dataclass(frozen=True)
class ConcentrationReport:
    """Result of :func:`hoeffding_enclosure`.

    Attributes
    ----------
    interval : Interval
        Outward-rounded ``[mean - t, mean + t]``, containing the true
        expectation with probability ``>= 1 - delta`` conditional on
        ``value_range`` bounding every draw.
    mean : float
        Sample mean (via :func:`math.fsum` for a compensated summation).
    stderr : float
        Sample standard error ``std / sqrt(n)`` (diagnostic only; the
        Hoeffding half-width ``t`` is what actually backs ``interval``).
    delta : float
        Declared failure probability.
    n : int
        Number of samples.
    value_range : tuple[float, float]
        The declared ``(lo, hi)`` bound on every draw.
    """

    interval: Interval
    mean: float
    stderr: float
    delta: float
    n: int
    value_range: tuple[float, float]


def hoeffding_enclosure(
    samples: Sequence[float],
    *,
    value_range: tuple[float, float],
    delta: float = 1e-9,
) -> ConcentrationReport:
    r"""Probabilistic enclosure of ``E[X]`` from i.i.d. samples of a bounded ``X``.

    Parameters
    ----------
    samples
        Non-empty sequence of i.i.d. draws of the random variable.
    value_range
        ``(lo, hi)``, a declared bound that every draw (seen or unseen) is
        assumed to respect. Every entry of ``samples`` is checked against it;
        a violation raises ``ValueError`` rather than silently returning a
        bound that does not actually hold.
    delta
        Target failure probability, ``0 < delta < 1``. The returned interval
        contains the true expectation with probability ``>= 1 - delta``.

    Returns
    -------
    ConcentrationReport

    Notes
    -----
    The interval half-width is ``(hi - lo) * sqrt(log(2 / delta) / (2 * n))``
    -- widening with the declared range and shrinking as ``1 / sqrt(n)``, the
    standard Hoeffding rate. The final addition step is outward-rounded
    (:func:`math.nextafter`) so the float bound is never tighter than the
    exact real one; the ``log`` / ``sqrt`` evaluations themselves use ordinary
    (non-directed) libm, so this stays a *practical* concentration bound, not
    an :mod:`omnibias.core.verified`-grade certified enclosure end to end.
    """
    n = len(samples)
    if n < 1:
        raise ValueError("hoeffding_enclosure requires at least one sample")
    lo, hi = value_range
    if not (isfinite(lo) and isfinite(hi)):
        raise ValueError(f"value_range must be finite, got {value_range}")
    if lo > hi:
        raise ValueError(f"value_range lo={lo} must be <= hi={hi}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    for i, x in enumerate(samples):
        xf = float(x)
        if not isfinite(xf):
            raise ValueError(f"sample {i} is not finite: {xf!r}")
        if xf < lo or xf > hi:
            raise ValueError(
                f"sample {i}={xf!r} falls outside the declared value_range "
                f"[{lo}, {hi}]; the Hoeffding premise does not hold for this "
                "data, so no enclosure can be returned"
            )
    mean = fsum(float(x) for x in samples) / n
    width = hi - lo
    if width == 0.0:
        half_width = 0.0
    else:
        half_width = width * sqrt(log(2.0 / delta) / (2.0 * n))
    lo_out = nextafter(mean - half_width, -inf)
    hi_out = nextafter(mean + half_width, inf)
    interval = Interval(lo_out, hi_out)
    if n == 1:
        stderr = 0.0
    else:
        variance = fsum((float(x) - mean) ** 2 for x in samples) / (n - 1)
        stderr = sqrt(variance / n)
    return ConcentrationReport(
        interval=interval,
        mean=mean,
        stderr=stderr,
        delta=delta,
        n=n,
        value_range=(lo, hi),
    )


__all__ = [
    "ConcentrationReport",
    "hoeffding_enclosure",
]
