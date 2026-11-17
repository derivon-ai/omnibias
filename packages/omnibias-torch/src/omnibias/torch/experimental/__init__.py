# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Experimental graph transformations for Taylor-normalized derivatives.

The compiler uses ``torch.export`` at construction time and emits an ordinary
FX module. It is tested with PyTorch 2.14 and supports only its documented
primitive rules; unsupported code raises a source-located error.
"""

from omnibias.core.apolar import ApolarAlgebra, apolar_algebra
from omnibias.core.demand import (
    Quantities,
    QuantitySpec,
    Residual,
    legacy_full_gradient,
    legacy_projected,
)
from omnibias.core.jet_algebra import JetBudget, JetSpec
from omnibias.core.operator_spec import OperatorPlanningBudget, OperatorSpec
from omnibias.torch.biharmonic import (
    BiharmonicState,
    biharmonic_activation,
    biharmonic_add,
    biharmonic_affine,
    biharmonic_multiply,
    biharmonic_seed,
    mlp_biharmonic,
)

from ._chain import Skip, fourier_feature_layer
from .compiler import (
    JetCompileError,
    JetProgram,
    coefficients_to_derivatives,
    compile_jet,
    directional_seed,
    extract_derivative,
    register_jet_rule,
    seed_jet,
)
from .demand import streams_quantities, streams_residual
from .operator import OperatorProgram, compile_operator
from .projected import (
    input_factor,
    projected_deep_divergence,
    projected_deep_laplacian,
    projected_deep_logdet,
)
from .projected_polylaplacian import projected_deep_bilaplacian, projected_deep_trilaplacian
from .residual_gram import (
    deep_residual_gauss_newton,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
)
from .woodbury import residual_inverse_newton, residual_solve, residual_solve_transpose

__all__ = [
    "ApolarAlgebra",
    "BiharmonicState",
    "CompiledOperator",
    "CompiledResidual",
    "JetBudget",
    "JetCompileError",
    "JetProgram",
    "JetSpec",
    "OperatorPlanningBudget",
    "OperatorProgram",
    "OperatorSpec",
    "Quantities",
    "QuantitySpec",
    "Residual",
    "Skip",
    "apolar_algebra",
    "biharmonic_activation",
    "biharmonic_add",
    "biharmonic_affine",
    "biharmonic_multiply",
    "biharmonic_seed",
    "coefficients_to_derivatives",
    "compile_jet",
    "compile_operator",
    "compile_planned_operator",
    "compile_residual",
    "deep_residual_gauss_newton",
    "deep_residual_gram",
    "deep_residual_ntk_diagonal",
    "directional_seed",
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
    "extract_derivative",
    "fourier_feature_layer",
    "input_factor",
    "legacy_full_gradient",
    "legacy_projected",
    "mlp_biharmonic",
    "projected_deep_bilaplacian",
    "projected_deep_divergence",
    "projected_deep_laplacian",
    "projected_deep_logdet",
    "projected_deep_trilaplacian",
    "register_jet_rule",
    "residual_inverse_newton",
    "residual_solve",
    "residual_solve_transpose",
    "seed_jet",
    "streams_quantities",
    "streams_residual",
]


def __getattr__(name: str) -> object:
    """Names promoted to :mod:`omnibias.torch.operators` still resolve here, on the schedule of ``omnibias.core.deprecation``."""
    import importlib

    from omnibias.core.deprecation import lookup, resolve
    from omnibias.torch import __version__

    importlib.import_module("omnibias.torch.operators")  # registers the aliases
    old = f"{__name__}.{name}"
    if lookup(old) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return resolve(old, __version__)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
