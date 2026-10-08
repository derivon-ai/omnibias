# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Experimental graph transformations for Taylor-normalized derivatives.

The compiler uses ``torch.export`` at construction time and emits an ordinary
FX module. It is tested with PyTorch 2.14 and supports only its documented
primitive rules; unsupported code raises a source-located error.
"""

from omnibias.core.apolar import ApolarAlgebra, apolar_algebra
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
from .operator import OperatorProgram, compile_operator

__all__ = [
    "ApolarAlgebra",
    "BiharmonicState",
    "JetBudget",
    "JetCompileError",
    "JetProgram",
    "JetSpec",
    "OperatorPlanningBudget",
    "OperatorProgram",
    "OperatorSpec",
    "apolar_algebra",
    "biharmonic_activation",
    "biharmonic_add",
    "biharmonic_affine",
    "biharmonic_multiply",
    "biharmonic_seed",
    "coefficients_to_derivatives",
    "compile_jet",
    "compile_operator",
    "directional_seed",
    "extract_derivative",
    "mlp_biharmonic",
    "register_jet_rule",
    "seed_jet",
]
