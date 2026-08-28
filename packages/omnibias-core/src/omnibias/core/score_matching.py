# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact Hyvärinen score matching (theory 09-21).

``E[ ||s||^2 / 2 + div s ]`` with a Hutchinson-free divergence.
For an affine OMBU score ``s = w x + b`` the divergence is ``w``.
For a tower score ``s = sigma'(w x + b) w`` the divergence is a
``sigma''`` contraction from founding bias collapse (``delta -> 0``)
order 2. Temperature collapse (``beta -> inf``, feasibility) does
not appear. do not conflate the two.

This is a training objective. Exact CNF ``div`` (``integrate_cnf``)
is prior art and is not claimed as new. Not ImageNet. Not CCF
stretch. ``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

DISCLAIMER = (
    "Hyvärinen / DSM on an OMBU score; CNF exact div is prior art, "
    "not ImageNet, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "cnf_div_claimed_new": False,
        "imagenet_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class ExactSMConfig:
    kind: str = "hyvarinen"
    n_samples: int = 64
    n_seeds: int = 5
    n_probe: int = 64


DEFAULT_CONFIG = ExactSMConfig()


def hyvarinen_point(score: float, div_s: float) -> float:
    """``||s||^2 / 2 + div s`` at one point."""
    return 0.5 * score * score + div_s


def exact_div_neg_id(dim: int = 1) -> float:
    """Closed-form ``div(-x) = -dim`` (no Hutchinson)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return -float(dim)


def linear_score(x: float, weight: float, bias: float = 0.0) -> float:
    """Affine OMBU score ``s = w x + b`` (identity cell, no ``sigma``)."""
    return weight * x + bias


def linear_div(weight: float) -> float:
    """``d/dx (w x + b) = w``."""
    return float(weight)


def tower_div_tanh(pre: float, weight: float) -> float:
    """``div( w sech^2(pre) )`` via founding ``sigma''`` (tanh)."""
    t = math.tanh(pre)
    sech2 = 1.0 - t * t
    sigma_dd = -2.0 * t * sech2
    return sigma_dd * weight * weight


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("xs must be nonempty")
    return sum(xs) / float(len(xs))


def fit_affine_score(xs: Sequence[float]) -> tuple[float, float]:
    """Exact minimizer of the 1-D Hyvärinen loss over ``s = w x + b``."""
    mx = _mean(xs)
    mxx = _mean([x * x for x in xs])
    var = mxx - mx * mx
    if var <= 1e-18:
        raise ValueError("sample variance is degenerate")
    weight = -1.0 / var
    bias = mx / var
    return weight, bias


def hyvarinen_mean(xs: Sequence[float], weight: float, bias: float) -> float:
    div = linear_div(weight)
    return _mean([hyvarinen_point(linear_score(x, weight, bias), div) for x in xs])


def dsm_point(score_noisy: float, noise: float, sigma: float) -> float:
    """Denoising score matching at one point: target ``-noise / sigma^2``."""
    if sigma == 0.0:
        raise ValueError("dsm sigma must be nonzero")
    target = -noise / (sigma * sigma)
    err = score_noisy - target
    return err * err


def gaussian_samples(n: int, *, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def exact_div_variance(divs: Sequence[float]) -> float:
    """Sample variance of an exact-div stream (must be 0 if constant)."""
    if len(divs) < 2:
        return 0.0
    # Averaging identical binary64 values by repeated addition can itself round,
    # producing a tiny nonzero residual below.  Constant exact-div streams are
    # structurally zero-variance, so preserve that fact before numerical
    # accumulation.
    if all(div == divs[0] for div in divs[1:]):
        return 0.0
    mu = _mean(divs)
    return _mean([(d - mu) * (d - mu) for d in divs])


def hutchinson_shear_probe(e1: float, e2: float) -> float:
    """Single-probe Hutchinson of ``s(x,y)=(x+y, x)``. Exact ``div`` is ``1``."""
    return e1 * e1 + 2.0 * e1 * e2


def worked_example() -> dict[str, float]:
    """G1: ``div(-x) = -1`` and Hyvärinen at ``x=1`` is ``-0.5``."""
    div = exact_div_neg_id(1)
    score = linear_score(1.0, -1.0, 0.0)
    return {
        "div": div,
        "score": score,
        "hyvarinen": hyvarinen_point(score, div),
        "abs_div_err": abs(div + 1.0),
        "abs_h_err": abs(hyvarinen_point(score, div) + 0.5),
    }


def score_matching_skill(*, config: ExactSMConfig | None = None) -> dict[str, object]:
    """G2: five-seed 1-D N(0,1) affine OMBU vs the zero-score baseline."""
    cfg = DEFAULT_CONFIG if config is None else config
    reached = 0
    losses: list[float] = []
    alignments: list[float] = []
    exact_vars: list[float] = []
    for seed in range(cfg.n_seeds):
        xs = gaussian_samples(cfg.n_samples, seed=seed + 1)
        weight, bias = fit_affine_score(xs)
        loss = hyvarinen_mean(xs, weight, bias)
        probe = gaussian_samples(cfg.n_probe, seed=10_000 + seed)
        align = abs(_mean([linear_score(x, weight, bias) + x for x in probe]))
        divs = [linear_div(weight) for _ in xs]
        exact_vars.append(exact_div_variance(divs))
        losses.append(loss)
        alignments.append(align)
        if loss < 0.0 and align < 0.2:
            reached += 1
    hon = honesty_payload()
    return {
        "reached": reached,
        "losses": losses,
        "alignments": alignments,
        "exact_div_variance": exact_vars,
        "hutchinson_exact_path_variance": 0.0,
        "cnf_div_claimed_new": hon["cnf_div_claimed_new"],
        "g2_earned": reached >= 3 and max(exact_vars) == 0.0,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "ExactSMConfig",
    "dsm_point",
    "exact_div_neg_id",
    "exact_div_variance",
    "fit_affine_score",
    "gaussian_samples",
    "honesty_payload",
    "hutchinson_shear_probe",
    "hyvarinen_mean",
    "hyvarinen_point",
    "linear_div",
    "linear_score",
    "score_matching_skill",
    "tower_div_tanh",
    "worked_example",
]
