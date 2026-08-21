# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Taylor-model neuron (theory 09-05).

A unit whose output is a :class:`TaylorModel`, not a float. Affine
image plus ``sigma`` via the closed-form tower (founding bias collapse
``delta -> 0``) and a Lagrange remainder. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.

Sound enclosure of a shallow monotone cell. Not a deep-net
certificate, not ImageNet, not CCF stretch, and not Navier–Stokes
regularity. ``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import ast
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from omnibias.core.verified.interval import Interval, hull
from omnibias.core.verified.sigma import sigma_tower_interval, sigma_value_interval
from omnibias.core.verified.taylor_model import TaylorModel

DISCLAIMER = (
    "TM hidden state; sound remainder, not a deep-net certificate, not ImageNet, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "imagenet_claim": False,
        "stretch_claim": False,
        "deep_net_certificate": False,
        "ns_regularity": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class TMNeuronSpec:
    order: int = 1
    activation: str = "sigmoid"


DEFAULT_SPEC = TMNeuronSpec()


def _even_pow(iv: Interval, power: int) -> Interval:
    raw = iv.pow_int(power)
    if power % 2 == 0:
        return Interval(max(raw.lo, 0.0), raw.hi)
    return raw


def input_tm(lo: float, hi: float, order: int) -> TaylorModel:
    """Identity Taylor model on ``[lo, hi]``."""
    if hi < lo:
        raise ValueError(f"empty box: [{lo}, {hi}]")
    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    return TaylorModel.identity(center, radius, order)


def compose_sigma_tm(tm: TaylorModel, name: str, order: int) -> TaylorModel:
    """``sigma`` of a 1-D TM: polynomial tower plus Lagrange remainder."""
    if order < 1:
        raise ValueError("TM-sigma needs order >= 1")
    mid = tm.coeffs[0].mid
    tower = sigma_tower_interval(name, Interval.point(mid), order)
    shifted = tm - mid
    poly = TaylorModel.constant(tower[0], tm.center, tm.radius, tm.order)
    power = TaylorModel.constant(1.0, tm.center, tm.radius, tm.order)
    fact = 1
    for k in range(1, order + 1):
        fact *= k
        power = power * shifted
        poly = poly + power * (tower[k] * Interval.from_rational(Fraction(1, fact)))
    rng = hull([tm.bound(), Interval.point(mid)])
    tail = sigma_tower_interval(name, rng, order + 1)[order + 1]
    h_range = tm.bound() - Interval.point(mid)
    lagrange = tail * Interval.from_rational(Fraction(1, math.factorial(order + 1))) * _even_pow(
        h_range, order + 1
    )
    result = TaylorModel(tm.center, tm.radius, poly.coeffs, poly.remainder + lagrange)
    direct = sigma_value_interval(name, tm.bound())
    poly_bound = result._poly_bound()
    cap = direct - poly_bound
    new_lo = max(result.remainder.lo, cap.lo)
    new_hi = min(result.remainder.hi, cap.hi)
    if new_lo <= new_hi:
        return TaylorModel(tm.center, tm.radius, result.coeffs, Interval(new_lo, new_hi))
    return result


def tm_dense(
    tm_in: TaylorModel,
    weight: float,
    bias: float,
    spec: TMNeuronSpec | None = None,
) -> TaylorModel:
    """Affine ``w * x + b`` then TM-sigma. Pure Python; no torch / jax."""
    cfg = DEFAULT_SPEC if spec is None else spec
    pre = tm_in * Interval.point(weight) + Interval.point(bias)
    return compose_sigma_tm(pre, cfg.activation, cfg.order)


def worked_example() -> dict[str, float | bool]:
    """Order-1 sigmoid TM on ``[0, 0.2]`` about ``0.1``."""
    spec = TMNeuronSpec(order=1, activation="sigmoid")
    tm = tm_dense(input_tm(0.0, 0.2, spec.order), 1.0, 0.0, spec)
    box = tm.bound()
    return {
        "lo": box.lo,
        "hi": box.hi,
        "width": box.width,
        "remainder_width": tm.remainder.width,
        "vacuous": box.width >= 1e6,
    }


def nest_layers(
    weights: Sequence[float],
    biases: Sequence[float],
    *,
    lo: float = 0.0,
    hi: float = 0.2,
    spec: TMNeuronSpec | None = None,
) -> TaylorModel:
    cfg = DEFAULT_SPEC if spec is None else spec
    tm = input_tm(lo, hi, cfg.order)
    for weight, bias in zip(weights, biases, strict=True):
        tm = tm_dense(tm, weight, bias, cfg)
    return tm


def depth_honesty(*, layers: int = 6) -> dict[str, float | bool | int]:
    """G3: a 6-layer random-sign nest. Explosion is allowed and recorded."""
    rng = random.Random(20260821)
    weights = [rng.uniform(-3.0, 3.0) for _ in range(layers)]
    biases = [rng.uniform(-0.5, 0.5) for _ in range(layers)]
    tm = nest_layers(weights, biases)
    width = tm.remainder.width
    exploded = width > 1e3
    return {
        "layers": layers,
        "remainder_width": width,
        "bound_width": tm.bound().width,
        "enclosure_exploded": exploded,
    }


def contains_grid_and_sample(
    tm: TaylorModel,
    fn: str,
    lo: float,
    hi: float,
    *,
    n_grid: int = 65,
    n_sample: int = 32,
    seed: int = 20260821,
) -> bool:
    """Core verified contract: enclosure holds a grid and a random sample."""
    box = tm.bound()
    xs = [lo + (hi - lo) * i / (n_grid - 1) for i in range(n_grid)]
    rng = random.Random(seed)
    xs.extend(rng.uniform(lo, hi) for _ in range(n_sample))
    for x in xs:
        if fn == "sigmoid":
            ez = math.exp(-x) if x >= 0.0 else math.exp(x)
            true = 1.0 / (1.0 + ez) if x >= 0.0 else ez / (1.0 + ez)
        elif fn == "tanh":
            true = math.tanh(x)
        else:
            raise ValueError(f"unsupported sample activation {fn!r}")
        if not box.contains(true):
            return False
    return True


def source_imports_no_backend(path: Path | None = None) -> bool:
    """G4: the authored module must not import torch / jax / tensorflow / keras."""
    src = Path(path) if path is not None else Path(__file__)
    tree = ast.parse(src.read_text(encoding="utf-8"))
    blocked = {"torch", "jax", "tensorflow", "keras"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in blocked:
                    return False
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in blocked:
                return False
    return True


__all__ = [
    "DEFAULT_SPEC",
    "DISCLAIMER",
    "TMNeuronSpec",
    "compose_sigma_tm",
    "contains_grid_and_sample",
    "depth_honesty",
    "honesty_payload",
    "input_tm",
    "nest_layers",
    "source_imports_no_backend",
    "tm_dense",
    "worked_example",
]
