# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-sos: certified universal positivity by optimization.

An SOS decomposition ``p = z(x)^T Q z(x)`` with positive-semidefinite ``Q``
certifies ``p(x) >= 0`` for every ``x``. Some nonnegative polynomials are not
sums of squares. A floating-point
semidefinite program *proposes* the Gram matrix ``Q``; the *proof* is a rigorous,
outward-rounded interval ``LDL^T`` positive-definiteness certificate from
:mod:`omnibias.core.verified` -- the same finite obligation the Mathlib-free Lean
kernel re-checks, so a sealed certificate can earn ``theorem_prover_verified``.

The optimizer only proposes; the interval algebra and the Lean kernel prove.
Everything is *sound by construction*: a failed rounding or positive-definite
margin yields an **inconclusive** verdict, never a false positivity claim.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.sos.certify import (
    DEFAULT_DENOMINATORS,
    certify_sos,
    certify_sos_rational,
    degree_reduction_report,
    is_sos,
    named_adapted_problems,
    rational_gram,
)
from omnibias.sos.formal import is_theorem_prover_verified, lean_available, lean_check_sos
from omnibias.sos.honesty import (
    FINITE_DIM_SYSTEM,
    GALERKIN_TRUNCATION,
    GLOBAL_POLYNOMIAL,
    SOSScope,
    honesty_labels,
    seal_sos_certificate,
)
from omnibias.sos.inequality import PolynomialInequalityBackend
from omnibias.sos.monomials import (
    MonomialBasis,
    SOSProblem,
    arrangement_adapted_basis,
    gram_products,
    gram_to_poly,
    monomial_basis,
)
from omnibias.sos.positivstellensatz import (
    PositivstellensatzCertificate,
    SOSMultiplier,
    certify_nonneg_on_set,
    is_nonneg_on_set,
    seal_positivstellensatz_certificate,
)
from omnibias.sos.problem import (
    Exponent,
    Polynomial,
    RationalPolynomial,
    SOSCertificate,
)

try:
    __version__ = _pkg_version("omnibias-sos")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Limit family exposed as package metadata.
__lineage__ = "exempt: infrastructure"

__all__ = [
    "DEFAULT_DENOMINATORS",
    "Exponent",
    "FINITE_DIM_SYSTEM",
    "GALERKIN_TRUNCATION",
    "GLOBAL_POLYNOMIAL",
    "MonomialBasis",
    "Polynomial",
    "PolynomialInequalityBackend",
    "PositivstellensatzCertificate",
    "RationalPolynomial",
    "SOSCertificate",
    "SOSMultiplier",
    "SOSProblem",
    "SOSScope",
    "__lineage__",
    "__version__",
    "arrangement_adapted_basis",
    "certify_nonneg_on_set",
    "certify_sos",
    "certify_sos_rational",
    "degree_reduction_report",
    "gram_products",
    "gram_to_poly",
    "honesty_labels",
    "is_nonneg_on_set",
    "is_sos",
    "is_theorem_prover_verified",
    "lean_available",
    "lean_check_sos",
    "monomial_basis",
    "named_adapted_problems",
    "rational_gram",
    "seal_positivstellensatz_certificate",
    "seal_sos_certificate",
]
