# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-boolean: differentiable Boolean algebra.

The pure-Python :mod:`omnibias.boolean._core` provides exact truth-table, ANF /
Reed-Muller, Walsh / Fourier, Boolean-differential-calculus, equation-solving and
multilinear-extension primitives. The differentiable backends live under
``omnibias.boolean.torch`` and ``omnibias.boolean.jax`` (soft gates, the
jet-based spectrum engine, the beta-annealed soft-gate system solver, and design
losses); import them explicitly so the core stays dependency-free.

The exact core is ground truth; the backends are differentiable *relaxations*
(heuristics, no completeness guarantee). See the package README for the
discrete <-> continuous derivative bridge and the honesty guardrails.

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.boolean._core import (
    BooleanAntiderivative,
    BooleanSolution,
    GeneralSolution,
    GF2Solution,
    TruthTable,
    absolute_indicator_iv,
    algebraic_degree,
    all_assignments,
    anf_from_multilinear_coeffs,
    anf_from_truth_table,
    anf_monomials,
    anf_to_string,
    assignment,
    autocorrelation_iv,
    bit_to_spin,
    boolean_derivative,
    boolean_derivative_reduced,
    boolean_derivative_set,
    boolean_integral,
    check_truth_table,
    constraint_from_predicate,
    constraints_are_linear,
    differential_bias_iv,
    eliminant,
    equation_from_callables,
    fourier_coeffs,
    fourier_coeffs_iv,
    fourier_influences,
    gf2_solve,
    index_of,
    influences,
    is_independent_of,
    is_satisfiable,
    linear_bias_iv,
    linear_system_rows,
    linearity_iv,
    max_linear_bias_iv,
    mixed_partial,
    mobius_iv,
    multilinear_coeffs,
    multilinear_eval,
    multilinear_eval_from_coeffs,
    nonlinearity_iv,
    num_vars,
    parseval_defect,
    parseval_defect_iv,
    pm1_values,
    reduced_index,
    restrict,
    solution_set,
    solve_for,
    solve_system,
    spin_to_bit,
    system_constraint,
    total_influence,
    truth_table_from_anf,
    truth_table_from_callable,
    truth_table_to_callable,
    values_from_multilinear_coeffs,
    verify_assignment,
    walsh_at_iv,
    walsh_hadamard,
    walsh_hadamard_iv,
    walsh_spectrum,
    walsh_spectrum_iv,
)

try:
    __version__ = _pkg_version("omnibias-boolean")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "both"

__all__ = [
    "BooleanAntiderivative",
    "BooleanSolution",
    "GF2Solution",
    "GeneralSolution",
    "TruthTable",
    "__lineage__",
    "__version__",
    "absolute_indicator_iv",
    "algebraic_degree",
    "all_assignments",
    "anf_from_multilinear_coeffs",
    "anf_from_truth_table",
    "anf_monomials",
    "anf_to_string",
    "assignment",
    "autocorrelation_iv",
    "bit_to_spin",
    "boolean_derivative",
    "boolean_derivative_reduced",
    "boolean_derivative_set",
    "boolean_integral",
    "check_truth_table",
    "constraint_from_predicate",
    "constraints_are_linear",
    "differential_bias_iv",
    "eliminant",
    "equation_from_callables",
    "fourier_coeffs",
    "fourier_coeffs_iv",
    "fourier_influences",
    "gf2_solve",
    "index_of",
    "influences",
    "is_independent_of",
    "is_satisfiable",
    "linear_bias_iv",
    "linear_system_rows",
    "linearity_iv",
    "max_linear_bias_iv",
    "mixed_partial",
    "mobius_iv",
    "multilinear_coeffs",
    "multilinear_eval",
    "multilinear_eval_from_coeffs",
    "nonlinearity_iv",
    "num_vars",
    "parseval_defect",
    "parseval_defect_iv",
    "pm1_values",
    "reduced_index",
    "restrict",
    "solution_set",
    "solve_for",
    "solve_system",
    "spin_to_bit",
    "system_constraint",
    "total_influence",
    "truth_table_from_anf",
    "truth_table_from_callable",
    "truth_table_to_callable",
    "values_from_multilinear_coeffs",
    "verify_assignment",
    "walsh_at_iv",
    "walsh_hadamard",
    "walsh_hadamard_iv",
    "walsh_spectrum",
    "walsh_spectrum_iv",
]
