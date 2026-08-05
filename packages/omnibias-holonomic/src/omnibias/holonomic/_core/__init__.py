# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python core of the omnibias holonomic engine (no backend imports)."""

from __future__ import annotations

from omnibias.holonomic._core.asymptotics import (
    AsymptoticEstimate,
    certified_asymptotic,
    empirical_rate,
    precursive_asymptotics,
)
from omnibias.holonomic._core.certify import (
    HolonomicProof,
    prove_hypergeometric_identity,
    prove_identity_zeilberger,
)
from omnibias.holonomic._core.dfinite import (
    DFinite,
    PRecursive,
    dfinite_add,
    dfinite_cauchy,
    dfinite_hadamard,
)
from omnibias.holonomic._core.factor import (
    rational_roots,
    roots_with_multiplicity,
    square_free,
)
from omnibias.holonomic._core.gosper import (
    GosperResult,
    gosper_normal_form,
    gosper_sum,
)
from omnibias.holonomic._core.guess import (
    guess_algebraic,
    guess_dfinite,
    guess_recurrence,
    recurrence_to_operator,
)
from omnibias.holonomic._core.hyperterm import (
    ProperTerm,
    binomial_nk,
    geometric_k,
)
from omnibias.holonomic._core.ore import (
    OreAlgebra,
    OrePolynomial,
    diff_algebra,
    shift_algebra,
)
from omnibias.holonomic._core.oreops import (
    OreDivision,
    gcrd,
    lclm,
    ore_divmod,
    symmetric_product,
)
from omnibias.holonomic._core.petkovsek import (
    hyper,
    term_ratio_annihilates,
)
from omnibias.holonomic._core.qholonomic import (
    QGosperResult,
    QRecurrence,
    q_gosper,
    q_gosper_definite_sum,
    q_gosper_normal_form,
    q_shift_algebra,
    q_zeilberger,
)
from omnibias.holonomic._core.rational_poly import (
    Poly,
    dispersion_set,
    peval,
    pgcd,
    to_poly,
)
from omnibias.holonomic._core.transforms import (
    dfinite_compose_poly,
    dfinite_derivative,
    dfinite_integral,
    dfinite_to_precursive,
    precursive_to_dfinite,
)
from omnibias.holonomic._core.zeilberger import (
    Summand,
    Telescoper,
    ZeilbergerCertificate,
    creative_telescoping,
    gosper_definite_sum,
    summand_sum,
    wz_certificate,
    wz_pair,
    zeilberger,
)

__all__ = [
    "AsymptoticEstimate",
    "DFinite",
    "GosperResult",
    "HolonomicProof",
    "OreAlgebra",
    "OreDivision",
    "OrePolynomial",
    "PRecursive",
    "Poly",
    "ProperTerm",
    "QGosperResult",
    "QRecurrence",
    "Summand",
    "Telescoper",
    "ZeilbergerCertificate",
    "binomial_nk",
    "certified_asymptotic",
    "creative_telescoping",
    "dfinite_add",
    "dfinite_cauchy",
    "dfinite_compose_poly",
    "dfinite_derivative",
    "dfinite_hadamard",
    "dfinite_integral",
    "dfinite_to_precursive",
    "diff_algebra",
    "dispersion_set",
    "empirical_rate",
    "gcrd",
    "geometric_k",
    "gosper_definite_sum",
    "gosper_normal_form",
    "gosper_sum",
    "guess_algebraic",
    "guess_dfinite",
    "guess_recurrence",
    "hyper",
    "lclm",
    "ore_divmod",
    "peval",
    "pgcd",
    "precursive_asymptotics",
    "precursive_to_dfinite",
    "prove_hypergeometric_identity",
    "prove_identity_zeilberger",
    "q_gosper",
    "q_gosper_definite_sum",
    "q_gosper_normal_form",
    "q_shift_algebra",
    "q_zeilberger",
    "rational_roots",
    "recurrence_to_operator",
    "roots_with_multiplicity",
    "shift_algebra",
    "square_free",
    "summand_sum",
    "symmetric_product",
    "term_ratio_annihilates",
    "to_poly",
    "wz_certificate",
    "wz_pair",
    "zeilberger",
]
