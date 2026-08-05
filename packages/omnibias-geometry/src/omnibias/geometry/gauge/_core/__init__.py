# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic gauge-theory schemas and numeric kernels (pure Python)."""

from __future__ import annotations

from omnibias.geometry.gauge._core.connection import (
    EUCLIDEAN_4D,
    MINKOWSKI_4D,
    GaugeConnectionSpec,
    connection_component_names,
    gauge_connection_spec,
)
from omnibias.geometry.gauge._core.forms import (
    LieAlgebraValuedForm,
    hodge_star_flat,
    levi_civita_symbol,
    permutation_sign,
    sorted_index_sets,
    wedge,
)
from omnibias.geometry.gauge._core.lie_algebra import (
    LieAlgebra,
    as_lie_algebra,
    su,
    u1,
)
from omnibias.geometry.gauge._core.representation import (
    Irrep,
    adjoint,
    adjoint_rep_matrices,
    antisymmetric_power,
    antisymmetric_power_rep_matrices,
    branching_to_subalgebra,
    character,
    dimension,
    dual_coxeter_number,
    dynkin_index,
    fundamental,
    irrep,
    quadratic_casimir,
    su2_spin_matrices,
    symmetric_power,
    symmetric_power_rep_matrices,
    tensor_product_decomposition,
    trivial,
    weight_multiplicities,
)

__all__ = [
    "EUCLIDEAN_4D",
    "GaugeConnectionSpec",
    "Irrep",
    "LieAlgebra",
    "LieAlgebraValuedForm",
    "MINKOWSKI_4D",
    "adjoint",
    "adjoint_rep_matrices",
    "antisymmetric_power",
    "antisymmetric_power_rep_matrices",
    "as_lie_algebra",
    "branching_to_subalgebra",
    "character",
    "connection_component_names",
    "dimension",
    "dual_coxeter_number",
    "dynkin_index",
    "fundamental",
    "gauge_connection_spec",
    "hodge_star_flat",
    "irrep",
    "levi_civita_symbol",
    "permutation_sign",
    "quadratic_casimir",
    "sorted_index_sets",
    "su",
    "su2_spin_matrices",
    "symmetric_power",
    "symmetric_power_rep_matrices",
    "tensor_product_decomposition",
    "trivial",
    "u1",
    "wedge",
    "weight_multiplicities",
]
