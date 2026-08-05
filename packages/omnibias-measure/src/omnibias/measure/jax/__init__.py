# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX backend for omnibias-measure.

``ops`` holds the differentiable functional primitives (bit-identical twin of
``omnibias.measure._core`` and ``omnibias.measure.torch``); ``layers`` holds the
functional / equinox-style pytree layers ``LebesgueIntegral`` /
``ExpectationLayer`` / ``LayerCakeIntegral``; ``pointprocess`` builds
temporal-point-process / survival likelihoods on the measure integral (exact
compensator via quadrature or the closed-form antiderivative window).
"""

from __future__ import annotations

from omnibias.measure.jax import integraleq, layers, ops, pointprocess
from omnibias.measure.jax.integraleq import (
    degenerate_kernel_solve,
    fredholm_residual,
    neumann_series,
    nystrom_solve,
    volterra_solve,
)
from omnibias.measure.jax.layers import (
    ExpectationLayer,
    LayerCakeIntegral,
    LebesgueIntegral,
)
from omnibias.measure.jax.ops import (
    importance_expectation,
    layer_cake_integral,
    lebesgue_integral,
    simple_function_approx,
    superlevel_measure,
)
from omnibias.measure.jax.pointprocess import (
    TemporalPointProcess,
    closed_form_compensator,
    compensator,
    poisson_nll,
    survival_nll,
)

__all__ = [
    "ExpectationLayer",
    "LayerCakeIntegral",
    "LebesgueIntegral",
    "TemporalPointProcess",
    "closed_form_compensator",
    "compensator",
    "degenerate_kernel_solve",
    "fredholm_residual",
    "importance_expectation",
    "integraleq",
    "layer_cake_integral",
    "layers",
    "lebesgue_integral",
    "neumann_series",
    "nystrom_solve",
    "ops",
    "pointprocess",
    "poisson_nll",
    "simple_function_approx",
    "superlevel_measure",
    "survival_nll",
    "volterra_solve",
]
