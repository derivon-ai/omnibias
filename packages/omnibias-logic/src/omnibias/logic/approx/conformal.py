# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Split-conformal coverage intervals for a model-count predictor -- STATISTICAL, not sound.

:class:`ConformalCounter` wraps *any* point predictor of the model count (a cheap Monte-Carlo
estimator by default, or a learned surrogate) in a **split-conformal** interval. After
calibrating on ``(problem, true_count)`` pairs it emits, for a fresh problem, an interval with
**marginal coverage** ``>= 1 - alpha`` -- i.e. averaged over the calibration/test
distribution, the true count lands inside at least a ``1 - alpha`` fraction of the time.

Honest scope -- this is **NOT** worst-case sound. The coverage is *distribution-dependent*
(it holds under exchangeability of the calibration and test instances, not adversarially) and
it returns an :class:`ApproxCount`, never a :class:`~omnibias.logic.model_count.certificate.CountCertificate`.
For a rigorous, instance-wise guarantee use :func:`~omnibias.logic.count_enclosure` or the
exact router.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from omnibias.logic.approx.result import ApproxCount
from omnibias.logic.model_count.problem import _formula_satisfied

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

    from omnibias.logic.model_count.problem import ModelCountProblem


def monte_carlo_estimate(
    problem: ModelCountProblem, rng: random.Random, samples: int
) -> float:
    r"""Unbiased Monte-Carlo point estimate ``Z0 * mean_x([x sat] * prod_i w_i(x_i))``."""
    n = problem.n
    fracs = problem.weight_fractions()
    clauses = problem.cnf.clauses
    total = 0.0
    for _ in range(samples):
        bits = tuple(rng.randint(0, 1) for _ in range(n))
        if _formula_satisfied(bits, clauses):
            weight = 1.0
            for i in range(n):
                weight *= float(fracs[i][bits[i]])
            total += weight
    return (total / samples) * float(2**n)


class ConformalCounter:
    r"""A split-conformal wrapper turning a count predictor into a coverage interval.

    Parameters
    ----------
    alpha:
        Target miscoverage; the calibrated interval has marginal coverage ``>= 1 - alpha``.
    predictor:
        A point predictor ``problem -> estimate``. Defaults to a seeded Monte-Carlo estimator.
    seed, samples:
        Seed and sample budget for the default Monte-Carlo predictor.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        predictor: Callable[[ModelCountProblem], float] | None = None,
        *,
        seed: int = 0,
        samples: int = 4000,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.samples = samples
        self._rng = random.Random(seed)
        self._predictor: Callable[[ModelCountProblem], float] = (
            predictor if predictor is not None else self._default_predictor
        )
        self._quantile: float | None = None

    def _default_predictor(self, problem: ModelCountProblem) -> float:
        return monte_carlo_estimate(problem, self._rng, self.samples)

    def fit(
        self,
        problems: Sequence[ModelCountProblem],
        true_counts: Sequence[float],
    ) -> ConformalCounter:
        """Calibrate the score quantile on ``(problem, true_count)`` pairs (split-conformal)."""
        if len(problems) != len(true_counts):
            raise ValueError("problems and true_counts must have equal length")
        if not problems:
            raise ValueError("need at least one calibration point")
        scores = sorted(
            abs(float(true) - self._predictor(problem))
            for problem, true in zip(problems, true_counts, strict=True)
        )
        n_cal = len(scores)
        rank = math.ceil((n_cal + 1) * (1.0 - self.alpha)) - 1
        self._quantile = scores[rank] if rank < n_cal else math.inf
        return self

    def predict(self, problem: ModelCountProblem) -> ApproxCount:
        """Emit the calibrated marginal-coverage interval for ``problem`` (NOT sound)."""
        if self._quantile is None:
            raise RuntimeError("ConformalCounter.predict called before fit")
        estimate = self._predictor(problem)
        lower = max(0.0, estimate - self._quantile)
        upper = estimate + self._quantile
        return ApproxCount(
            estimate=estimate,
            interval=(lower, upper),
            method="split_conformal",
            confidence=1.0 - self.alpha,
        )


__all__ = ["ConformalCounter", "monte_carlo_estimate"]
