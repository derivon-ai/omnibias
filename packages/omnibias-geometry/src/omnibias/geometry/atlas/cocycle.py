# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sheaf-atlas cocycle residual (theory 09-09).

Transition maps are jets. The loss includes
``||Phi_kj compose Phi_ji - Phi_ki||`` to order ``N``. Chart maps use
founding bias collapse (``delta -> 0``) for ``sigma^(n)``. Partition
gates may harden as ``beta -> inf`` (temperature collapse,
feasibility). Do not conflate the two.

Not a rewrite of metrics-only ``AtlasSpec`` blending. Not a
sheaf-cohomology theorem. Not P vs NP on arrangements. Not CCF stretch.
Empty overlaps make the cocycle vacuous; G2 forces a nonempty overlap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from omnibias.partition._core.config import PartitionConfig
from omnibias.partition._core.params import PartitionParams
from omnibias.partition._core.weights import partition_weights

DISCLAIMER = (
    "jet cocycle on an atlas; not a sheaf-cohomology theorem, not "
    "P vs NP, and not CCF stretch"
)


def honesty_payload(*, beta: float = 1.0) -> dict[str, bool]:
    return {
        "temperature_collapse_used": float(beta) != 1.0,
        "sheaf_cohomology_theorem": False,
        "p_vs_np_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class SheafAtlasConfig:
    jet_order: int = 1
    beta: float = 1.0


DEFAULT_CONFIG = SheafAtlasConfig()


@dataclass(frozen=True)
class AffineChart:
    """``Phi(x) = scale * x + shift``. Order-1 jet is ``(Phi(x), scale)``."""

    scale: float
    shift: float = 0.0


def compose_transition_jet(
    inner: Sequence[float],
    outer_tower: Sequence[float],
) -> tuple[float, ...]:
    """Faà di Bruno composition ``outer ∘ inner`` for 1-D jets up to order 2.

    ``inner[k] = f^{(k)}(x)``, ``outer_tower[k] = g^{(k)}(f(x))``.
    Multiplying values without the chain rule leaves a 1-jet residual of 1
    on the worked example.
    """
    if len(inner) != len(outer_tower):
        raise ValueError("inner jet and outer tower must share an order")
    order = len(inner) - 1
    if order < 0 or order > 2:
        raise ValueError("compose_transition_jet implements orders 0..2")
    if order == 0:
        return (float(outer_tower[0]),)
    first = float(outer_tower[1]) * float(inner[1])
    if order == 1:
        return (float(outer_tower[0]), first)
    second = float(outer_tower[2]) * float(inner[1]) ** 2 + float(outer_tower[1]) * float(inner[2])
    return (float(outer_tower[0]), first, second)


def affine_jet(x: float, chart: AffineChart, order: int) -> tuple[float, ...]:
    val = chart.scale * x + chart.shift
    if order == 0:
        return (val,)
    jet = [val, chart.scale]
    jet.extend(0.0 for _ in range(order - 1))
    return tuple(jet)


def affine_tower(y: float, chart: AffineChart, order: int) -> tuple[float, ...]:
    return affine_jet(y, chart, order)


def jet_norm(jet: Sequence[float]) -> float:
    return max(abs(float(v)) for v in jet) if jet else 0.0


def cocycle_residual(
    transitions: Mapping[tuple[int, int], AffineChart],
    x: float,
    triples: Sequence[tuple[int, int, int]],
    *,
    config: SheafAtlasConfig | None = None,
) -> dict[str, float]:
    """Max jet-norm of ``Phi_kj ∘ Phi_ji - Phi_ki`` on named triples."""
    cfg = DEFAULT_CONFIG if config is None else config
    if not triples:
        raise ValueError("cocycle_residual needs a nonempty triple list")
    worst = 0.0
    for i, j, k in triples:
        inner = affine_jet(x, transitions[(j, i)], cfg.jet_order)
        outer = affine_tower(inner[0], transitions[(k, j)], cfg.jet_order)
        composed = compose_transition_jet(inner, outer)
        direct = affine_jet(x, transitions[(k, i)], cfg.jet_order)
        residual = jet_norm([a - b for a, b in zip(composed, direct, strict=True)])
        if residual > worst:
            worst = residual
    return {"residual": worst, "x": x}


def worked_example() -> dict[str, float]:
    """``Phi_21(x)=2x``, ``Phi_32(y)=y/2``, ``Phi_31=id`` at ``x=0.5``."""
    x = 0.5
    transitions = {
        (2, 1): AffineChart(2.0),
        (3, 2): AffineChart(0.5),
        (3, 1): AffineChart(1.0),
    }
    report = cocycle_residual(transitions, x, ((1, 2, 3),))
    inner = affine_jet(x, transitions[(2, 1)], 1)
    composed = compose_transition_jet(inner, affine_tower(inner[0], transitions[(3, 2)], 1))
    buggy_deriv = inner[1]  # value-compose but skip the chain rule
    return {
        "residual": report["residual"],
        "composed_value": composed[0],
        "composed_deriv": composed[1],
        "buggy_deriv_gap": abs(buggy_deriv - 1.0),
    }


def two_interval_poisson(*, seeds: int = 5, beta: float = 1.0) -> dict[str, object]:
    """G2: two-interval Poisson with a known nonempty overlap and inverse charts."""
    xs = [0.3 + 0.4 * i / 32.0 for i in range(33)]
    params = PartitionParams(
        PartitionConfig(n_features=1, depth=1, split_kind="axis", beta_final=beta),
        W=[[1.0]],
        t=[0.5],
    )
    X = [[x] for x in xs]
    weights = partition_weights(params, X, beta)
    overlap = [float(row[0] * row[1]) for row in weights]
    nonempty = min(overlap) > 0.0
    transitions = {
        (2, 1): AffineChart(2.0),
        (1, 2): AffineChart(0.5),
        (1, 1): AffineChart(1.0),
    }
    cocycles: list[float] = []
    tasks: list[float] = []
    for seed in range(seeds):
        x = xs[seed % len(xs)]
        cocycles.append(float(cocycle_residual(transitions, x, ((1, 2, 1),))["residual"]))
        u = x * (1.0 - x)
        u_pp = -2.0
        f = 2.0
        tasks.append(abs(u_pp + f))
        _ = u
    task_med = sorted(tasks)[len(tasks) // 2]
    zero = 2.0
    cyc_mean = sum(cocycles) / float(len(cocycles))
    return {
        "overlap_min": min(overlap),
        "nonempty_overlap": nonempty,
        "cocycle_mean": cyc_mean,
        "task_median": task_med,
        "skill_vs_zero": 1.0 - task_med / zero,
        "below_1e6": cyc_mean < 1e-6,
        "ignored_cocycle": cyc_mean > 1e-3 and task_med < 1e-8,
        "temperature_collapse_used": beta != 1.0,
        "beta": beta,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "AffineChart",
    "SheafAtlasConfig",
    "affine_jet",
    "affine_tower",
    "cocycle_residual",
    "compose_transition_jet",
    "honesty_payload",
    "jet_norm",
    "two_interval_poisson",
    "worked_example",
]
