# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fit a per-device :class:`~omnibias.core.planner.CostModel` from measurements and score dispatch regret.

A measurement is the steady-state time of one algorithm on one problem together with the machine-independent
:class:`~omnibias.core.planner.Work` the registry assigns to it; measurements of the algorithms that could serve one
problem share a ``group``. :func:`fit_cost_model` fits the rates and one efficiency per algorithm by minimising the
robust log error of the predicted times. :func:`dispatch_regret` then asks, per group, how much slower the algorithm the
model ranks first is than the best one measured. Everything is plain Python, so a device can be calibrated with only the
standard library on the machine that owns the device.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.core.planner import CostModel, Work

__all__ = [
    "Measurement",
    "RegretRow",
    "cross_validate",
    "dispatch_regret",
    "fit_cost_model",
    "summarize_regret",
]

#: Everything :meth:`CostModel.predict_work` reads besides the per-algorithm efficiencies. ``solver_s``, ``sync_s`` and ``units``
#: only move from the base value when the measurements carry solver calls, synchronizations or serial-flop floors.
_RATES = ("rate_gemm", "rate_gemm_f32", "rate_elem", "bandwidth", "launch_s", "solver_s", "sync_s", "units")


@dataclass(frozen=True)
class Measurement:
    """Steady-state seconds of ``algorithm`` on one problem (``group``), with its work model and compile seconds."""

    group: str
    algorithm: str
    work: Work
    seconds: float
    dtype: str = "float64"
    compile_seconds: float | None = None

    def to_config(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "algorithm": self.algorithm,
            "work": self.work.to_config(),
            "seconds": self.seconds,
            "dtype": self.dtype,
            "compile_seconds": self.compile_seconds,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Measurement:
        return cls(
            group=str(config["group"]),
            algorithm=str(config["algorithm"]),
            work=Work.from_config(config["work"]),
            seconds=float(config["seconds"]),
            dtype=str(config.get("dtype", "float64")),
            compile_seconds=(
                None if config.get("compile_seconds") is None else float(config["compile_seconds"])
            ),
        )


@dataclass(frozen=True)
class RegretRow:
    group: str
    chosen: str
    best: str
    regret: float  # measured seconds of the chosen algorithm / best measured seconds (>= 1)
    predicted_order: tuple[str, ...]

    def to_config(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "chosen": self.chosen,
            "best": self.best,
            "regret": self.regret,
            "predicted_order": list(self.predicted_order),
        }


def _predict(p: Mapping[str, float], m: Measurement) -> float:
    """:meth:`CostModel.predict_work` on the parameter dict ``p`` (the fit and the planner share one formula)."""
    w = m.work
    gemm = p["rate_gemm"] if m.dtype == "float64" else p["rate_gemm_f32"]
    core = (
        max(
            w.gemm_flops / gemm,
            w.traffic_bytes / p["bandwidth"],
            w.serial_flops / (gemm / max(p["units"], 1.0)),
        )
        + w.elem_flops / p["rate_elem"]
    )
    return (
        p.get("eff:" + m.algorithm, 1.0) * core
        + w.launches * p["launch_s"]
        + w.solver_calls * p["solver_s"]
        + w.syncs * p["sync_s"]
    )


def _huber(e: float, delta: float) -> float:
    a = abs(e)
    return 0.5 * e * e if a <= delta else delta * (a - 0.5 * delta)


def _loss(
    p: Mapping[str, float],
    base: Mapping[str, float],
    data: Sequence[Measurement],
    prior: float,
    delta: float,
) -> float:
    total = 0.0
    for m in data:
        pred = _predict(p, m)
        total += _huber(math.log(max(pred, 1e-30)) - math.log(m.seconds), delta)
    total /= max(1, len(data))
    for k, v in p.items():
        total += prior * (math.log(v) - math.log(base.get(k, 1.0))) ** 2 / max(1, len(p))
    return total


def fit_cost_model(
    measurements: Iterable[Measurement],
    *,
    device: str,
    base: CostModel | None = None,
    sweeps: int = 80,
    prior: float = 0.02,
    huber: float = 0.7,
    source: str | None = None,
) -> CostModel:
    """Fit the rates and per-algorithm efficiencies of ``base`` to ``measurements`` (coordinate descent in log space).

    The loss is the Huber-robust mean squared error of ``log(predicted) - log(measured)`` plus a weak pull toward
    ``base`` that keeps the rates physical where the data cannot identify them. ``compile_s_per_unit`` is the geometric
    mean of measured compile seconds per work-model unit.
    """
    data = [m for m in measurements if m.seconds > 0 and math.isfinite(m.seconds)]
    if not data:
        raise ValueError("no usable measurements")
    model = base or CostModel(device=device)
    p: dict[str, float] = {k: float(getattr(model, k)) for k in _RATES}
    if all(m.dtype == "float64" for m in data):
        fixed = {"rate_gemm_f32"}
    else:
        fixed = set()
    for m in data:
        p.setdefault("eff:" + m.algorithm, float(model.efficiency.get(m.algorithm, 1.0)))
    start = dict(p)
    best = _loss(p, start, data, prior, huber)
    steps = (4.0, 2.0, 1.4, 1.15, 1.04)
    for _ in range(sweeps):
        improved = False
        for k in list(p):
            if k in fixed:
                continue
            lo = start[k] * 1e-4
            hi = start[k] * 1e4
            for s in steps:
                for cand in (p[k] * s, p[k] / s):
                    if not lo <= cand <= hi:
                        continue
                    old = p[k]
                    p[k] = cand
                    loss = _loss(p, start, data, prior, huber)
                    if loss < best - 1e-12:
                        best = loss
                        improved = True
                        break
                    p[k] = old
        if not improved:
            break
    compiles = [
        m.compile_seconds / max(m.work.compile_units, 1e-9)
        for m in data
        if m.compile_seconds is not None and m.compile_seconds > 0
    ]
    compile_unit = (
        math.exp(sum(math.log(c) for c in compiles) / len(compiles))
        if compiles
        else model.compile_s_per_unit
    )
    eff = {k[4:]: v for k, v in p.items() if k.startswith("eff:")}
    return CostModel(
        device=device,
        rate_gemm=p["rate_gemm"],
        rate_gemm_f32=p["rate_gemm_f32"],
        rate_elem=p["rate_elem"],
        bandwidth=p["bandwidth"],
        launch_s=p["launch_s"],
        compile_s_per_unit=compile_unit,
        efficiency=eff,
        solver_s=p["solver_s"],
        sync_s=p["sync_s"],
        units=p["units"],
        source=source or f"fitted on {len(data)} measurements, loss {best:.4g}",
    )


def dispatch_regret(
    measurements: Iterable[Measurement], model: CostModel, *, expected_uses: int = 1
) -> list[RegretRow]:
    """Per group: the algorithm ``model`` ranks first (by ``seconds + compile / expected_uses``) against the best measured."""
    groups: dict[str, list[Measurement]] = defaultdict(list)
    for m in measurements:
        groups[m.group].append(m)
    uses = max(1, int(expected_uses))
    rows = []
    for group, ms in sorted(groups.items()):
        scored = []
        for m in ms:
            pred = model.predict_work(m.algorithm, m.work, m.dtype)
            scored.append((pred.seconds + pred.compile_seconds / uses, m))
        scored.sort(key=lambda t: t[0])
        chosen = scored[0][1]
        best = min(ms, key=lambda m: m.seconds)
        rows.append(
            RegretRow(
                group=group,
                chosen=chosen.algorithm,
                best=best.algorithm,
                regret=chosen.seconds / best.seconds,
                predicted_order=tuple(m.algorithm for _, m in scored),
            )
        )
    return rows


def summarize_regret(rows: Sequence[RegretRow]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "median": math.nan,
            "mean": math.nan,
            "p90": math.nan,
            "worst": math.nan,
            "hit_rate": math.nan,
        }
    r = sorted(row.regret for row in rows)
    n = len(r)

    def q(frac: float) -> float:
        pos = frac * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        return r[lo] + (r[hi] - r[lo]) * (pos - lo)

    return {
        "n": float(n),
        "median": q(0.5),
        "mean": math.exp(sum(math.log(x) for x in r) / n),
        "p90": q(0.9),
        "worst": r[-1],
        "hit_rate": sum(1 for row in rows if row.chosen == row.best) / n,
    }


def cross_validate(
    measurements: Sequence[Measurement],
    *,
    device: str,
    folds: int = 5,
    expected_uses: int = 1,
    **fit_kwargs: Any,
) -> list[RegretRow]:
    """Regret of models fitted without each fold of groups (grouped, so one problem never trains and tests together)."""
    groups = sorted({m.group for m in measurements})
    rows: list[RegretRow] = []
    for k in range(max(2, folds)):
        held = {g for i, g in enumerate(groups) if i % max(2, folds) == k}
        train = [m for m in measurements if m.group not in held]
        test = [m for m in measurements if m.group in held]
        if not train or not test:
            continue
        model = fit_cost_model(train, device=device, **fit_kwargs)
        rows.extend(dispatch_regret(test, model, expected_uses=expected_uses))
    return rows
