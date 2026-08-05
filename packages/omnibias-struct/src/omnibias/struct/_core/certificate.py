# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The closed-form soft-DP optimality-gap certificate.

:func:`certify_soft_dp` sandwiches the hard DP optimum ``V*`` and the soft (``lse_beta``)
value ``V_beta`` with the closed-form gap bound ``log(N) / beta`` (``N`` = exact path
count). In the ``max`` convention (Viterbi / longest path / best CTC alignment)

.. math::
    V^* \;\le\; V_\beta \;\le\; V^* + \tfrac{\log N}{\beta},

the left inequality being ``lse_beta >= max``; in the ``min`` convention (shortest path)
the sandwich mirrors to ``V^* - \log N / \beta \le V_\beta \le V^*``. The certificate is
**honest**: it never claims ``V_beta == V*`` (the soft value is a genuine relaxation),
only that the gap is provably at most ``log(N) / beta`` and vanishes as ``beta -> inf``.
The optional ``brute_force_value`` records the exact-DP agreement self-check.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.struct._core.gap import logsumexp_gap_bound


@dataclass(frozen=True)
class DPGapCertificate:
    r"""A closed-form sandwich certificate for a soft-DP value against the hard optimum.

    Attributes
    ----------
    hard_value:
        The hard DP optimum ``V*`` (max score, or min cost when ``sense == "min"``).
    soft_value:
        The soft ``lse_beta`` value ``V_beta`` from the differentiable backend.
    gap_bound:
        The rigorous closed-form gap ``log(num_paths) / beta``.
    beta:
        The inverse temperature used for the relaxation.
    num_paths:
        The exact number ``N`` of complete paths / alignments.
    sense:
        ``"max"`` (Viterbi / longest / CTC) or ``"min"`` (shortest path).
    method:
        Always ``"logsumexp_gap"`` -- the closed-form bound (no optimization).
    agrees_with_bruteforce:
        ``True`` / ``False`` if a brute-force optimum was supplied and matched (the
        exact-DP self-check), else ``None``.
    tol:
        Numerical tolerance for the sandwich checks.
    """

    hard_value: float
    soft_value: float
    gap_bound: float
    beta: float
    num_paths: int
    sense: str = "max"
    method: str = "logsumexp_gap"
    agrees_with_bruteforce: bool | None = None
    tol: float = 1e-9

    @property
    def lse_ge_max_holds(self) -> bool:
        """Whether the ``lse_beta >= max`` side holds (``<=`` for the ``min`` sense)."""
        if self.sense == "min":
            return self.soft_value <= self.hard_value + self.tol
        return self.soft_value >= self.hard_value - self.tol

    @property
    def gap_bound_holds(self) -> bool:
        """Whether the soft value stays within ``gap_bound`` of the hard optimum."""
        if self.sense == "min":
            return self.soft_value >= self.hard_value - self.gap_bound - self.tol
        return self.soft_value <= self.hard_value + self.gap_bound + self.tol

    @property
    def absolute_gap(self) -> float:
        """The realized ``|V_beta - V*|`` (``<= gap_bound`` when sound)."""
        return abs(self.soft_value - self.hard_value)

    @property
    def is_sound(self) -> bool:
        """Whether the full closed-form sandwich holds for the measured values."""
        return self.lse_ge_max_holds and self.gap_bound_holds

    @property
    def certified(self) -> bool:
        """Whether a rigorous closed-form gap was produced *and* the sandwich holds."""
        return "logsumexp_gap" in self.method and self.is_sound


def certify_soft_dp(
    hard_value: float,
    soft_value: float,
    num_paths: int,
    beta: float,
    *,
    sense: str = "max",
    stepwise_bound: float | None = None,
    brute_force_value: float | None = None,
    tol: float = 1e-9,
) -> DPGapCertificate:
    r"""Build a :class:`DPGapCertificate` sandwiching ``soft_value`` and ``hard_value``.

    Parameters
    ----------
    hard_value:
        The hard DP optimum ``V*`` (from :func:`omnibias.struct.viterbi` etc.).
    soft_value:
        The soft ``lse_beta`` value ``V_beta`` (from a backend ``soft_*``).
    num_paths:
        The exact path count ``N`` (:func:`omnibias.struct.count_paths`); must be ``>= 1``
        (a problem with no complete paths is infeasible and cannot be certified).
    beta:
        The inverse temperature ``beta > 0``.
    sense:
        ``"max"`` (default) or ``"min"``.
    stepwise_bound:
        Optional tighter per-step bound (:func:`omnibias.struct.stepwise_gap_bound`); when
        given, the certified ``gap_bound`` is the (still-sound) minimum of it and the
        global ``log(N) / beta``, and ``method`` records the tightening.
    brute_force_value:
        Optional exact-DP oracle value; when given, ``agrees_with_bruteforce`` records
        whether ``hard_value`` matched it within ``tol``.
    tol:
        Numerical tolerance for the sandwich / agreement checks.
    """
    if sense not in ("max", "min"):
        raise ValueError(f"sense must be 'max' or 'min', got {sense!r}")
    if num_paths < 1:
        raise ValueError(
            f"no complete paths (num_paths={num_paths}); the problem is infeasible and "
            "cannot be certified"
        )
    global_bound = logsumexp_gap_bound(num_paths, beta)
    if stepwise_bound is None:
        gap_bound, method = global_bound, "logsumexp_gap"
    else:
        gap_bound = min(global_bound, float(stepwise_bound))
        method = "min(logsumexp_gap, stepwise)"
    agrees = None if brute_force_value is None else abs(hard_value - brute_force_value) <= tol
    return DPGapCertificate(
        hard_value=float(hard_value),
        soft_value=float(soft_value),
        gap_bound=float(gap_bound),
        beta=float(beta),
        num_paths=int(num_paths),
        sense=sense,
        method=method,
        agrees_with_bruteforce=agrees,
        tol=float(tol),
    )


__all__ = ["DPGapCertificate", "certify_soft_dp"]
