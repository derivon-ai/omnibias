# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.geometry.gauge: non-abelian gauge theory on the omnibias substrate.

Lie algebras ``su(N)`` / ``u(1)`` (generators, structure constants ``f^{abc}``,
symmetric ``d^{abc}``), Lie-algebra-valued differential forms, gauge connections
``A_mu^a``, the field strength ``F = dA + g[A, A]``, the gauge-covariant
derivative and divergence (the Yang-Mills operator), the flat signature-aware
Hodge star on ``R^4``, the Yang-Mills action, topological charge, and gauge
transformations -- with closed-form connection derivatives from the
``omnibias-fields`` substrate and cross-backend (torch + jax) parity.

The pure-Python schemas (:class:`LieAlgebra`, :class:`LieAlgebraValuedForm`,
:class:`GaugeConnectionSpec`) live in :mod:`omnibias.geometry.gauge._core`; the backend
ops live in ``omnibias.geometry.gauge.torch`` and ``omnibias.geometry.gauge.jax``.

Two further submodules are imported by path rather than re-exported here:
``omnibias.geometry.gauge.lattice`` (the SU(2) lattice Monte-Carlo engine and its
statistical mass-gap estimators) and ``omnibias.geometry.gauge.transfer``
(rigorously enclosed transfer matrices and their **certified** spectral gaps, with
sealed certificates registered through ``omnibias.geometry.gauge.proofmachine``).

Maturity: this is an **alpha** submodule (folded in from the former standalone
``omnibias-gauge`` package) shipped inside the Beta ``omnibias-geometry``
distribution. It is the non-abelian extension of ``omnibias-geometry``'s
Riemannian / abelian exterior calculus, on the same ``omnibias-fields`` substrate;
its gauge-bundle API may still change while the rest of ``omnibias-geometry`` is Beta.
"""

from __future__ import annotations

from omnibias.geometry.gauge._core import (
    ADJOINT_BIANCHI,
    ADJOINT_YM,
    EUCLIDEAN_4D,
    LEGAL_ADJOINT_1FORM_ATOMS,
    LEGAL_SINGLET_ATOMS,
    MINKOWSKI_4D,
    SELF_DUAL_ACTION_OVER_TOPOLOGICAL,
    SINGLET_BIANCHI_SQ,
    SINGLET_SELF_DUAL_SQ,
    SINGLET_TR_F2,
    SINGLET_TR_F_FTILDE,
    SINGLET_YM_SQ,
    GaugeConnectionSpec,
    GaugeCovariantJet,
    Irrep,
    LieAlgebra,
    LieAlgebraValuedForm,
    adjoint,
    adjoint_rep_matrices,
    antisymmetric_power,
    antisymmetric_power_rep_matrices,
    as_lie_algebra,
    assert_library_gauge_legal,
    bpst_instanton_arrays,
    branching_to_subalgebra,
    character,
    connection_component_names,
    covariant_singlet_columns,
    dimension,
    dual_coxeter_number,
    dynkin_index,
    evaluate_gauge_law_gate,
    fundamental,
    gauge_connection_spec,
    gauge_equivariance_defect,
    global_gauge_transform_connection,
    hodge_star_flat,
    irrep,
    levi_civita_symbol,
    permutation_sign,
    quadratic_casimir,
    random_special_unitary,
    sorted_index_sets,
    su,
    su2_spin_matrices,
    symmetric_power,
    symmetric_power_rep_matrices,
    tensor_product_decomposition,
    thooft_eta,
    trivial,
    u1,
    wedge,
    weight_multiplicities,
)

__all__ = [
    "ADJOINT_BIANCHI",
    "ADJOINT_YM",
    "EUCLIDEAN_4D",
    "GaugeConnectionSpec",
    "GaugeCovariantJet",
    "Irrep",
    "LEGAL_ADJOINT_1FORM_ATOMS",
    "LEGAL_SINGLET_ATOMS",
    "LieAlgebra",
    "LieAlgebraValuedForm",
    "MINKOWSKI_4D",
    "SELF_DUAL_ACTION_OVER_TOPOLOGICAL",
    "SINGLET_BIANCHI_SQ",
    "SINGLET_SELF_DUAL_SQ",
    "SINGLET_TR_F2",
    "SINGLET_TR_F_FTILDE",
    "SINGLET_YM_SQ",
    "adjoint",
    "adjoint_rep_matrices",
    "antisymmetric_power",
    "antisymmetric_power_rep_matrices",
    "as_lie_algebra",
    "assert_library_gauge_legal",
    "bpst_instanton_arrays",
    "branching_to_subalgebra",
    "character",
    "connection_component_names",
    "covariant_singlet_columns",
    "dimension",
    "dual_coxeter_number",
    "dynkin_index",
    "evaluate_gauge_law_gate",
    "fundamental",
    "gauge_connection_spec",
    "gauge_equivariance_defect",
    "global_gauge_transform_connection",
    "hodge_star_flat",
    "irrep",
    "levi_civita_symbol",
    "permutation_sign",
    "quadratic_casimir",
    "random_special_unitary",
    "sorted_index_sets",
    "su",
    "su2_spin_matrices",
    "symmetric_power",
    "symmetric_power_rep_matrices",
    "tensor_product_decomposition",
    "thooft_eta",
    "trivial",
    "u1",
    "wedge",
    "weight_multiplicities",
]
