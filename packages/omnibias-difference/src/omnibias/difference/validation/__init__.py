# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Data-driven refinement harness for omnibias-difference.

Pure-Python (core-only) probe utilities that turn each tier capability into an
*instrumented* experiment, per the ``omnibias-dev-empirical-validation`` gates:

* :func:`enclosure_soundness` -- the verified-primitive rule that an enclosure
  must contain a **dense deterministic grid AND a random sample** of true values.
* :func:`high_precision_derivative` / :func:`require_mpmath` -- the mpmath
  high-precision oracle adapter.
* :func:`baseline_compare` -- the best-in-class comparator against a named
  baseline (no baseline => no claim).
* :class:`Finding` / :class:`FindingsLedger` -- a structured ledger of the gaps,
  flaws, and bugs a probe exposes, written as JSON to a configurable output
  directory (``$OMNIBIAS_RUNS_DIR`` or a per-user temp dir by default) so large
  artifacts stay out of the repo tree.

This module is import-light and **optional**: it is never imported by
:mod:`omnibias.difference` itself, and ``mpmath`` is imported lazily so the
core-only import guard stays clean.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from omnibias.core.verified.interval import Interval

#: Severity tags, ascending: an ``info`` note, a missing capability (``gap``), a
#: correctness/tightness weakness (``flaw``), or an outright ``bug``.
SEVERITIES: tuple[str, ...] = ("info", "gap", "flaw", "bug")


@dataclass(frozen=True)
class Finding:
    """One gaps/flaws/bugs entry surfaced by a probe."""

    workstream: str
    severity: str
    summary: str
    detail: str = ""
    repro: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping of the finding."""
        return asdict(self)


class FindingsLedger:
    """A collection of :class:`Finding` records with JSON serialisation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.findings: list[Finding] = []

    def add(
        self, workstream: str, severity: str, summary: str, detail: str = "", **repro: Any
    ) -> Finding:
        """Record a finding and return it."""
        finding = Finding(
            workstream=workstream,
            severity=severity,
            summary=summary,
            detail=detail,
            repro=dict(repro),
        )
        self.findings.append(finding)
        return finding

    def extend(self, findings: Iterable[Finding]) -> None:
        """Absorb findings from another probe."""
        self.findings.extend(findings)

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def counts(self) -> dict[str, int]:
        """Count of findings by severity (every severity present, possibly ``0``)."""
        out = {severity: 0 for severity in SEVERITIES}
        for finding in self.findings:
            out[finding.severity] += 1
        return out

    def to_json(self) -> str:
        """Canonical, deterministic JSON of the ledger."""
        return json.dumps(
            {
                "name": self.name,
                "counts": self.counts(),
                "findings": [finding.to_dict() for finding in self.findings],
            },
            indent=2,
            sort_keys=True,
        )

    def write(self, path: str | None = None) -> str:
        """Write the ledger JSON (default: the scratch findings dir); return the path."""
        if path is None:
            directory = default_ledger_dir()
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{self.name}.json")
        else:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        return path

    def summary(self) -> str:
        """A short human-readable digest for the CI smoke output."""
        counts = self.counts()
        head = f"[{self.name}] " + ", ".join(f"{k}={counts[k]}" for k in SEVERITIES)
        lines = [head]
        for finding in self.findings:
            lines.append(f"  - ({finding.severity}) [{finding.workstream}] {finding.summary}")
        return "\n".join(lines)


def default_ledger_dir() -> str:
    """Directory for findings JSON (kept out of the repo tree by default).

    Override with the ``OMNIBIAS_RUNS_DIR`` environment variable; otherwise a
    per-user temporary directory is used so large artifacts never land in the
    working tree.
    """
    base = os.environ.get("OMNIBIAS_RUNS_DIR")
    if base:
        return os.path.join(os.path.expanduser(base), "difference_refine")
    return os.path.join(tempfile.gettempdir(), "omnibias_runs", "difference_refine")


@dataclass(frozen=True)
class SoundnessReport:
    """Result of a grid-and-random enclosure soundness check."""

    sound: bool
    n_grid: int
    n_random: int
    max_escape: float
    failures: tuple[tuple[float, float], ...]

    @property
    def n_points(self) -> int:
        """Total number of oracle points checked."""
        return self.n_grid + self.n_random


def grid_and_random_points(
    box: Interval, *, grid: int = 25, random_samples: int = 40, seed: int = 0
) -> list[float]:
    """A dense deterministic grid over ``box`` plus a seeded random sample."""
    if grid < 0 or random_samples < 0:
        raise ValueError("grid and random_samples must be >= 0")
    lo, hi = box.lo, box.hi
    points: list[float] = []
    if grid == 1:
        points.append(0.5 * (lo + hi))
    elif grid > 1:
        step = (hi - lo) / (grid - 1)
        points.extend(lo + i * step for i in range(grid))
    rng = random.Random(seed)
    points.extend(rng.uniform(lo, hi) for _ in range(random_samples))
    return points


def enclosure_soundness(
    enclosure: Interval,
    oracle: Callable[[float], float],
    box: Interval,
    *,
    grid: int = 25,
    random_samples: int = 40,
    seed: int = 0,
) -> SoundnessReport:
    """Check that ``enclosure`` contains ``oracle`` over a grid-and-random sample of ``box``.

    This is the ``delta -> 0`` acceptance gate: the interval enclosure must
    contain a dense deterministic grid **and** a random sample of true values. A
    single escaping point makes the report ``sound=False`` -- a soundness bug,
    never something to be papered over by widening.
    """
    failures: list[tuple[float, float]] = []
    max_escape = 0.0
    for x in grid_and_random_points(box, grid=grid, random_samples=random_samples, seed=seed):
        value = oracle(x)
        if not (enclosure.lo <= value <= enclosure.hi):
            escape = max(enclosure.lo - value, value - enclosure.hi)
            max_escape = max(max_escape, escape)
            failures.append((x, value))
    return SoundnessReport(
        sound=not failures,
        n_grid=grid,
        n_random=random_samples,
        max_escape=max_escape,
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class BaselineComparison:
    """A best-in-class comparison of a candidate against a named baseline."""

    name: str
    candidate: float
    baseline: float
    lower_is_better: bool
    wins: bool
    ratio: float


def baseline_compare(
    name: str,
    candidate: float,
    baseline: float,
    *,
    lower_is_better: bool = True,
    tol: float = 0.0,
) -> BaselineComparison:
    """Compare a candidate metric to a named baseline (``lower_is_better`` by default)."""
    if lower_is_better:
        wins = candidate <= baseline + tol
    else:
        wins = candidate + tol >= baseline
    ratio = candidate / baseline if baseline != 0.0 else math.inf
    return BaselineComparison(
        name=name,
        candidate=candidate,
        baseline=baseline,
        lower_is_better=lower_is_better,
        wins=wins,
        ratio=ratio,
    )


def require_mpmath() -> Any:
    """Return the ``mpmath`` module or raise a helpful error (the probe oracle)."""
    try:
        return importlib.import_module("mpmath")
    except ImportError as exc:  # pragma: no cover - the [test] extra ships mpmath
        raise RuntimeError(
            "this probe needs mpmath: pip install 'omnibias-difference[test]'"
        ) from exc


def high_precision_derivative(
    func: Callable[[Any], Any], z0: float, order: int, *, dps: int = 50
) -> float:
    """High-precision ``func^(order)(z0)`` via ``mpmath.taylor`` (the numerical oracle)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    mp = require_mpmath()
    with mp.workdps(dps):
        taylor = mp.taylor(func, z0, order)
        return float(taylor[order] * math.factorial(order))


def seed_sweep(metric: Callable[[int], float], seeds: Iterable[int]) -> dict[str, float]:
    """Aggregate a per-seed metric across ``seeds`` (the anti-overfitting rule)."""
    values = [metric(seed) for seed in seeds]
    if not values:
        raise ValueError("seeds must be non-empty")
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "n": float(len(values)),
    }


__all__ = [
    "BaselineComparison",
    "Finding",
    "FindingsLedger",
    "SEVERITIES",
    "SoundnessReport",
    "baseline_compare",
    "default_ledger_dir",
    "enclosure_soundness",
    "grid_and_random_points",
    "high_precision_derivative",
    "require_mpmath",
    "seed_sweep",
]
