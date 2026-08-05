# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic measure substrate (pure Python + numpy).

The :class:`Measure` abstraction and the numpy reference implementations of the
measure-integral primitives. The torch / jax twins in
``omnibias.measure.<backend>`` consume the same :class:`Measure` (nodes and
weights generated once in float64), so they are bit-identical by construction.
"""

from __future__ import annotations

from omnibias.measure._core.integraleq import (
    SINGULAR_RCOND,
    KernelFn,
    NeumannResult,
    SourceFn,
    cumulative_trapezoid_matrix,
    degenerate_kernel_solve,
    fredholm_residual,
    neumann_series,
    nystrom_solve,
    solvability_margin,
    volterra_solve,
)
from omnibias.measure._core.integrate import (
    IntegrandFn,
    SimpleFunctionApprox,
    importance_expectation,
    layer_cake_integral,
    lebesgue_integral,
    simple_function_approx,
    superlevel_measure,
)
from omnibias.measure._core.measure import (
    DensityFn,
    Measure,
    counting,
    dirac,
    empirical,
    from_quadrature,
    gaussian,
    lebesgue,
    uniform_mc,
)

__all__ = [
    "DensityFn",
    "IntegrandFn",
    "KernelFn",
    "Measure",
    "NeumannResult",
    "SINGULAR_RCOND",
    "SimpleFunctionApprox",
    "SourceFn",
    "counting",
    "cumulative_trapezoid_matrix",
    "degenerate_kernel_solve",
    "dirac",
    "empirical",
    "fredholm_residual",
    "from_quadrature",
    "gaussian",
    "importance_expectation",
    "layer_cake_integral",
    "lebesgue",
    "lebesgue_integral",
    "neumann_series",
    "nystrom_solve",
    "simple_function_approx",
    "solvability_margin",
    "superlevel_measure",
    "uniform_mc",
    "volterra_solve",
]
