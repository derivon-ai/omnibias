# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch backend for omnibias-measure.

``ops`` holds the differentiable functional primitives (bit-identical twin of
``omnibias.measure._core``); ``layers`` holds the trainable ``nn.Module``
wrappers ``LebesgueIntegral`` / ``ExpectationLayer`` / ``LayerCakeIntegral``;
``pointprocess`` builds temporal-point-process / survival likelihoods on the
measure integral (exact compensator via quadrature or the closed-form
antiderivative window); ``integraleq`` solves Fredholm / Volterra integral
equations of the second kind on that same quadrature.
"""

from __future__ import annotations

from omnibias.measure.torch import integraleq, layers, ops, pointprocess
from omnibias.measure.torch.integraleq import (
    degenerate_kernel_solve,
    fredholm_residual,
    neumann_series,
    nystrom_solve,
    volterra_solve,
)
from omnibias.measure.torch.layers import (
    ExpectationLayer,
    LayerCakeIntegral,
    LebesgueIntegral,
)
from omnibias.measure.torch.ops import (
    importance_expectation,
    layer_cake_integral,
    lebesgue_integral,
    simple_function_approx,
    superlevel_measure,
)
from omnibias.measure.torch.pointprocess import (
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
