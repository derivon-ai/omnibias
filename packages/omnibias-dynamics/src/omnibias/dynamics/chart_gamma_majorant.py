# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Track C2: chart-dependent ``gamma(s/r)`` majorant probes on the kill sequence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.dynamics.g1_passage import chi_tracking_gamma, frozen_majorant_log_ratio
from omnibias.dynamics.scale_dichotomy import KILL_EPS, kill_sep

__all__ = [
    "ChartGammaProbe",
    "chart_gamma_majorant_ratio",
    "probe_chart_gamma_on_kill_sequence",
]


def chart_gamma_majorant_ratio(s: float, r: float, kappa: float, C: float) -> float:
    """Evaluate the chart-dependent candidate ``gamma = s/r`` against the frozen template."""
    gamma = chi_tracking_gamma(s, r)
    return math.exp(frozen_majorant_log_ratio(s, r, kappa, C, gamma))


@dataclass(frozen=True)
class ChartGammaProbe:
    epsilon: float
    sep: float
    kappa: float
    majorant_constant: float
    chart_gamma: float
    log_ratio: float
    exceeds_one: bool
    route_specific: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "sep": self.sep,
            "kappa": self.kappa,
            "majorant_constant": self.majorant_constant,
            "chart_gamma": self.chart_gamma,
            "log_ratio": self.log_ratio,
            "exceeds_one": self.exceeds_one,
            "route_specific": True,
        }


def probe_chart_gamma_on_kill_sequence(
    *,
    eps: float = KILL_EPS,
    C: float = 1.0,
) -> ChartGammaProbe:
    """Test ``gamma(s/r)`` on the corrected kill sequence; does not discharge G1."""
    sep = kill_sep(eps)
    kappa = 1.0 / sep
    gamma = chi_tracking_gamma(sep, 1.0)
    log_ratio = frozen_majorant_log_ratio(sep, 1.0, kappa, C, gamma)
    return ChartGammaProbe(
        epsilon=eps,
        sep=sep,
        kappa=kappa,
        majorant_constant=C,
        chart_gamma=gamma,
        log_ratio=log_ratio,
        exceeds_one=log_ratio > 0.0,
        route_specific=True,
    )
