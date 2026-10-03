# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""G1 passage diagnostics: frozen-exponent kill-sequence assessment only.

The Lean ``frozen_exponent_obstruction`` quantifies over a single frozen
``(C, gamma)`` majorant. Chart-dependent ``gamma(s/r)`` candidates and
definitional closing maps that set ``h_max = h_e`` are **not** discharged
here; they record route-specific negatives for adversarial replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.dynamics.scale_dichotomy import KILL_EPS, kill_sep

__all__ = [
    "FrozenExponentReport",
    "chi_tracking_gamma",
    "existing_section_log_w_ratio",
    "frozen_exponent_kill_sequence_report",
    "frozen_majorant_log_ratio",
    "frozen_majorant_ratio",
]


def frozen_majorant_log_ratio(s: float, r: float, kappa: float, C: float, gamma: float) -> float:
    """Log of ``(s/r * exp(-(s/r)*kappa)) / (C * exp(-gamma*kappa))``."""
    if not all(math.isfinite(v) and v > 0 for v in (s, r, kappa, C, gamma)):
        raise ValueError("frozen_majorant_log_ratio requires positive finite inputs")
    return math.log(s / (r * C)) + (gamma - s / r) * kappa


def frozen_majorant_ratio(s: float, r: float, kappa: float, C: float, gamma: float) -> float:
    """``(s/r * exp(-(s/r)*kappa)) / (C * exp(-gamma*kappa))``."""
    return math.exp(frozen_majorant_log_ratio(s, r, kappa, C, gamma))


def chi_tracking_gamma(s: float, r: float) -> float:
    """Chart-dependent candidate ``gamma = s/r`` (tested separately, not a G1 discharge)."""
    if s <= 0 or r <= 0:
        raise ValueError("s and r must be positive")
    return s / r


@dataclass(frozen=True)
class FrozenExponentReport:
    """Negative diagnostic on the kill sequence for one frozen ``(C, gamma)``."""

    epsilon: float
    sep: float
    kappa: float
    frozen_log_ratio: float
    frozen_exceeds_one: bool
    route_specific: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "sep": self.sep,
            "kappa": self.kappa,
            "frozen_log_ratio": self.frozen_log_ratio,
            "frozen_exceeds_one": self.frozen_exceeds_one,
            "route_specific": self.route_specific,
        }


def frozen_exponent_kill_sequence_report(
    *,
    eps: float = KILL_EPS,
    C: float = 1.0,
    gamma: float = 1.0,
) -> FrozenExponentReport:
    """Replay the kill sequence for a **single** frozen majorant (not chart-uniform)."""
    sep = kill_sep(eps)
    kappa = 1.0 / sep
    frozen_log = frozen_majorant_log_ratio(sep, 1.0, kappa, C, gamma)
    return FrozenExponentReport(
        epsilon=eps,
        sep=sep,
        kappa=kappa,
        frozen_log_ratio=frozen_log,
        frozen_exceeds_one=frozen_log > 0.0,
        route_specific=True,
    )


def existing_section_log_w_ratio(eps: float, power: int, sigma: float) -> float:
    """``log(W_max/W_e)`` for ``h_max = eps^power``, ``h_e = eps^3 sigma^2``.

    Uses the same log-domain identity as
    :func:`~omnibias.dynamics.scale_dichotomy.outgoing_log_factor`.
    """
    if sigma <= 0 or eps <= 0:
        raise ValueError("eps and sigma must be positive")
    return (power - 3) * eps * math.log(eps) - 2.0 * eps * math.log(sigma)
