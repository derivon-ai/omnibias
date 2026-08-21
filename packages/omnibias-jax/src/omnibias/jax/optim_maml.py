# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact-MAML (jax; theory 09-16).

Inner step is Newton / GN. Meta-grad is IFT on inner stationarity.
That is the chain rule. The founding bias collapse (``delta -> 0``)
supplies exact HVPs. Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import exact_maml as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
ExactMAMLConfig = core.ExactMAMLConfig
honesty_payload = core.honesty_payload


def inner_newton_quadratic(theta: Array, alpha: Array) -> Array:
    return theta - (theta - alpha)


def ift_dtheta_dalpha(theta: Array, alpha: Array) -> Array:
    _ = (theta, alpha)
    return jnp.ones((), dtype=theta.dtype)


def exact_maml_meta_step(
    tasks: Sequence[float] | Array,
    params: Array,
    *,
    config: ExactMAMLConfig | None = None,
) -> dict[str, float]:
    alphas = [float(a) for a in jnp.asarray(tasks).reshape(-1).tolist()]
    return core.exact_maml_meta_step(alphas, float(jnp.asarray(params).reshape(-1)[0]), config=config)


def quadratic_worked_example() -> dict[str, float]:
    return core.quadratic_worked_example()


__all__ = [
    "DISCLAIMER",
    "ExactMAMLConfig",
    "exact_maml_meta_step",
    "honesty_payload",
    "ift_dtheta_dalpha",
    "inner_newton_quadratic",
    "quadratic_worked_example",
]
