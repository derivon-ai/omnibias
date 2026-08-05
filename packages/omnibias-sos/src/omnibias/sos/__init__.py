# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-sos: certified universal positivity by optimization.

A polynomial ``p(x) >= 0`` for **all** ``x`` iff it has an SOS decomposition
``p = z(x)^T Q z(x)`` with ``Q`` positive semidefinite.  A floating-point
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

from omnibias.sos.auxiliary import (
    DEFAULT_SLACKS,
    AuxiliaryBoundCertificate,
    PolynomialSystem,
    certify_time_average_bound,
    energy_conserving_triad_system,
    energy_observable,
    seal_auxiliary_bound,
)
from omnibias.sos.certify import (
    DEFAULT_DENOMINATORS,
    certify_sos,
    certify_sos_rational,
    is_sos,
    rational_gram,
)
from omnibias.sos.formal import (
    drive_sos_obligation,
    is_theorem_prover_verified,
    lean_available,
    lean_check_sos,
)
from omnibias.sos.honesty import (
    FINITE_DIM_SYSTEM,
    GALERKIN_TRUNCATION,
    GLOBAL_POLYNOMIAL,
    SOSScope,
    honesty_labels,
    seal_sos_certificate,
)
from omnibias.sos.monomials import (
    MonomialBasis,
    SOSProblem,
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
from omnibias.sos.proofmachine import (
    SOS_GLOBAL_NONNEG,
    SOS_NONNEG_ON_SET,
    SOS_TIME_AVERAGE_BOUND,
    build_sos_machine,
    replay_sos_certificate,
    sos_certificate_schema_errors,
    sos_provers,
)

try:
    __version__ = _pkg_version("omnibias-sos")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "exempt: infrastructure"

__all__ = [
    "AuxiliaryBoundCertificate",
    "DEFAULT_DENOMINATORS",
    "DEFAULT_SLACKS",
    "Exponent",
    "FINITE_DIM_SYSTEM",
    "GALERKIN_TRUNCATION",
    "GLOBAL_POLYNOMIAL",
    "MonomialBasis",
    "Polynomial",
    "PolynomialSystem",
    "PositivstellensatzCertificate",
    "RationalPolynomial",
    "SOSCertificate",
    "SOSMultiplier",
    "SOSProblem",
    "SOSScope",
    "SOS_GLOBAL_NONNEG",
    "SOS_NONNEG_ON_SET",
    "SOS_TIME_AVERAGE_BOUND",
    "__lineage__",
    "__version__",
    "build_sos_machine",
    "certify_nonneg_on_set",
    "certify_sos",
    "certify_sos_rational",
    "certify_time_average_bound",
    "drive_sos_obligation",
    "energy_conserving_triad_system",
    "energy_observable",
    "gram_products",
    "gram_to_poly",
    "honesty_labels",
    "is_nonneg_on_set",
    "is_sos",
    "is_theorem_prover_verified",
    "lean_available",
    "lean_check_sos",
    "monomial_basis",
    "rational_gram",
    "replay_sos_certificate",
    "seal_auxiliary_bound",
    "seal_positivstellensatz_certificate",
    "seal_sos_certificate",
    "sos_certificate_schema_errors",
    "sos_provers",
]
