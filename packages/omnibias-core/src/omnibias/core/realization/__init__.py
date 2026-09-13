# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Realization-map descriptions with explicit observation and parameter scope."""

from omnibias.core.realization.algebraic import AlgebraicNumber, RealAlgebraicField, root_count
from omnibias.core.realization.membership import (
    MembershipReport,
    classify_binary_quadratic,
    classify_linear,
    classify_scalar_quadratic,
)
from omnibias.core.realization.polynomial import (
    AlgebraBudget,
    AlgebraBudgetExceeded,
    PolynomialNetworkSpec,
    PolynomialRealizationMap,
    SparsePolynomial,
    compile_coefficient_map,
)
from omnibias.core.realization.rank import (
    ExactRankWitness,
    GenericRankReport,
    certify_generic_rank,
    certify_matrix_rank,
)
from omnibias.core.realization.schema import (
    IntegralKind,
    LayerSpec,
    ObservationKind,
    ObservationSpec,
    OperatorRole,
    ParameterBlock,
    ParameterLayout,
    RealizationScope,
    RealizationSpec,
)
from omnibias.core.realization.witness import (
    AlgebraicRealizationWitness,
    LaurentClosureWitness,
    LaurentPolynomial,
    RationalRealizationWitness,
    rational_realization_witness,
)

__all__ = [
    "AlgebraBudget", "AlgebraBudgetExceeded", "AlgebraicNumber", "AlgebraicRealizationWitness",
    "ExactRankWitness", "GenericRankReport", "IntegralKind", "LaurentClosureWitness", "LaurentPolynomial",
    "LayerSpec", "MembershipReport", "ObservationKind", "ObservationSpec", "OperatorRole",
    "ParameterBlock", "ParameterLayout", "PolynomialNetworkSpec", "PolynomialRealizationMap",
    "RationalRealizationWitness", "RealAlgebraicField", "RealizationScope", "RealizationSpec",
    "SparsePolynomial", "certify_generic_rank", "certify_matrix_rank", "classify_binary_quadratic",
    "classify_linear", "classify_scalar_quadratic", "compile_coefficient_map",
    "rational_realization_witness", "root_count",
]
