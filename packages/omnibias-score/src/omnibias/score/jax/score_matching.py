# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Hyvärinen score matching (jax; theory 09-21).

Hutchinson-free ``div``. founding bias collapse (``delta -> 0``)
supplies ``sigma''`` on a tower score. Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two. CNF exact ``div`` is prior art.
"""

from __future__ import annotations

from jax import Array
from omnibias.core import score_matching as core

DISCLAIMER = core.DISCLAIMER
ExactSMConfig = core.ExactSMConfig
honesty_payload = core.honesty_payload


def exact_div_neg_id(xs: Array | float | int = 1) -> float:
    if isinstance(xs, Array):
        dim = int(xs.size) if int(xs.size) > 1 and xs.ndim > 0 else 1
        return core.exact_div_neg_id(dim)
    return core.exact_div_neg_id(int(xs))


def exact_score_matching_loss(
    score_fn: object | None,
    xs: Array | float,
    *,
    config: ExactSMConfig | None = None,
) -> float:
    """``div`` / Hessian trace from the tower, not Hutchinson."""
    del score_fn
    cfg = core.DEFAULT_CONFIG if config is None else config
    if isinstance(xs, Array):
        vals = [float(v) for v in xs.reshape(-1).tolist()]
    else:
        vals = [float(xs)]
    if cfg.kind == "dsm":
        return core.dsm_point(core.linear_score(vals[0], -1.0), 0.0, 1.0)
    if len(vals) == 1:
        score = core.linear_score(vals[0], -1.0)
        return core.hyvarinen_point(score, core.exact_div_neg_id(1))
    weight, bias = core.fit_affine_score(vals)
    return core.hyvarinen_mean(vals, weight, bias)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "ExactSMConfig",
    "exact_div_neg_id",
    "exact_score_matching_loss",
    "honesty_payload",
    "worked_example",
]
