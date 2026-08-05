# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic lattice statistics (pure Python: jackknife, Creutz ratios).

These helpers operate on plain Python floats / sequences, so they are shared
verbatim by the torch and jax lattice backends.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def jackknife_std(samples: Sequence[float]) -> float:
    """Simple jackknife standard error of the mean."""
    vals = [float(s) for s in samples]
    n = len(vals)
    if n < 2:
        return 0.0
    total = sum(vals)
    jk = [(total - v) / (n - 1) for v in vals]
    jk_mean = sum(jk) / n
    var = sum((x - jk_mean) ** 2 for x in jk) / (n - 1)
    return math.sqrt(var) / (n**0.5)


def ensemble_mean_jackknife(samples: Sequence[float]) -> tuple[float, float]:
    """Ensemble mean and delete-1 jackknife standard error."""
    vals = [float(s) for s in samples]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    total = sum(vals)
    jk = [(total - vals[i]) / (n - 1) for i in range(n)]
    jk_mean = sum(jk) / n
    err = ((n - 1) / n * sum((x - jk_mean) ** 2 for x in jk)) ** 0.5
    return mean, err


def effective_mass(corr: Sequence[float], tau: int) -> float:
    """Lattice effective mass ``-ln(C(tau+1)/C(tau))``."""
    c0 = float(corr[tau])
    c1 = float(corr[tau + 1])
    if c0 <= 0.0 or c1 <= 0.0:
        return float("nan")
    return -math.log(c1 / c0)


def creutz_ratio_value(w_rt: float, w_r1t1: float, w_r1t: float, w_rt1: float) -> float:
    """Single Creutz ratio ``chi = -ln( W(R,T) W(R-1,T-1) / [W(R-1,T) W(R,T-1)] )``."""
    numer = w_rt * w_r1t1
    denom = w_r1t * w_rt1
    if numer <= 0.0 or denom <= 0.0:
        return float("nan")
    return -math.log(numer / denom)


def wilson_loops_ensemble(
    samples: dict[tuple[int, int], Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Jackknife ensemble means for Wilson loops keyed as ``"RxT"``."""
    out: dict[str, dict[str, float]] = {}
    for key in sorted(samples):
        r_val, t_val = key
        mean, err = ensemble_mean_jackknife(samples[key])
        out[f"{r_val}x{t_val}"] = {"value": mean, "err": err}
    return out


def creutz_ratios_ensemble(
    wilson_samples: dict[tuple[int, int], Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Creutz ratios ``chi(R,T)`` with jackknife errors from Wilson-loop samples."""
    keys = sorted(wilson_samples)
    sample_lists = {k: [float(x) for x in wilson_samples[k]] for k in keys}
    n = len(next(iter(sample_lists.values())))
    out: dict[str, dict[str, float]] = {}

    for r_val, t_val in keys:
        if r_val < 2 or t_val < 2:
            continue
        needed = (
            (r_val, t_val),
            (r_val - 1, t_val - 1),
            (r_val - 1, t_val),
            (r_val, t_val - 1),
        )
        if any(k not in sample_lists for k in needed):
            continue
        chi_full = creutz_ratio_value(
            sum(sample_lists[(r_val, t_val)]) / n,
            sum(sample_lists[(r_val - 1, t_val - 1)]) / n,
            sum(sample_lists[(r_val - 1, t_val)]) / n,
            sum(sample_lists[(r_val, t_val - 1)]) / n,
        )
        chi_jk: list[float] = []
        for i in range(n):
            chi_i = creutz_ratio_value(
                (sum(sample_lists[(r_val, t_val)]) - sample_lists[(r_val, t_val)][i]) / (n - 1),
                (sum(sample_lists[(r_val - 1, t_val - 1)]) - sample_lists[(r_val - 1, t_val - 1)][i]) / (n - 1),
                (sum(sample_lists[(r_val - 1, t_val)]) - sample_lists[(r_val - 1, t_val)][i]) / (n - 1),
                (sum(sample_lists[(r_val, t_val - 1)]) - sample_lists[(r_val, t_val - 1)][i]) / (n - 1),
            )
            if math.isfinite(chi_i):
                chi_jk.append(chi_i)
        if not math.isfinite(chi_full):
            continue
        if len(chi_jk) >= 2:
            jk_center = sum(chi_jk) / len(chi_jk)
            err = ((n - 1) / n * sum((x - jk_center) ** 2 for x in chi_jk)) ** 0.5
        else:
            err = 0.0
        out[f"{r_val}x{t_val}"] = {"value": chi_full, "err": err}
    return out


def string_tension_from_creutz(
    creutz: dict[str, dict[str, float]],
) -> dict[str, float | str | bool]:
    """Largest-accessible-loop Creutz-ratio estimate of string tension (fixed spacing).

    This is **not** continuum-extrapolated; it selects the largest-area finite
    Creutz ratio available in the input dict.
    """
    best_key = None
    best_area = -1
    for key, payload in creutz.items():
        r_s, t_s = key.split("x")
        area = int(r_s) * int(t_s)
        if area > best_area and math.isfinite(payload["value"]):
            best_area = area
            best_key = key
    if best_key is None:
        return {
            "value": float("nan"),
            "err": float("nan"),
            "from": "creutz_ratio(R,T)",
            "converged": False,
        }
    return {
        "value": float(creutz[best_key]["value"]),
        "err": float(creutz[best_key]["err"]),
        "from": f"creutz_ratio({best_key.replace('x', ',')})",
        "converged": False,
    }


__all__ = [
    "creutz_ratio_value",
    "creutz_ratios_ensemble",
    "effective_mass",
    "ensemble_mean_jackknife",
    "jackknife_std",
    "string_tension_from_creutz",
    "wilson_loops_ensemble",
]
