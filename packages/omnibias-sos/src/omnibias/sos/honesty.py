# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Scope labels and tamper-evident sealing for SOS certificates.

An SOS proof is a genuine ``for all x`` statement -- but only about what it
actually covers.  This module records that scope honestly and seals a proved
:class:`~omnibias.sos.problem.SOSCertificate` into a canonical, hash-sealed v1
``positive_definite`` certificate (via :mod:`omnibias.core.proof.certificate`),
so it can flow through the repo's schema / replay / Lean gates.

Every sealed certificate carries ``unproven_claim = False`` **by construction** --
there is no parameter, anywhere, that can set it ``True``.  The scope is one of:

* ``global_polynomial`` -- ``p(x) >= 0`` for all real ``x`` (a true universal
  statement about a polynomial);
* ``finite_dim_system`` -- a bound holding for all data of a finite-dimensional
  system (e.g. a polynomial ODE);
* ``galerkin_truncation`` -- a bound for a Galerkin-truncated PDE at a stated
  ``truncation_order`` -- explicitly **not** the continuum PDE and **not** a
  regularity / global-regularity statement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from omnibias.core.proof.certificate import Cert, positive_definite_certificate
from omnibias.core.verified.interval import Interval
from omnibias.sos.problem import SOSCertificate

GLOBAL_POLYNOMIAL = "global_polynomial"
FINITE_DIM_SYSTEM = "finite_dim_system"
GALERKIN_TRUNCATION = "galerkin_truncation"
_VALID_SCOPES = frozenset({GLOBAL_POLYNOMIAL, FINITE_DIM_SYSTEM, GALERKIN_TRUNCATION})


@dataclass(frozen=True)
class SOSScope:
    """What an SOS / auxiliary-functional certificate actually claims.

    ``kind`` must be one of :data:`GLOBAL_POLYNOMIAL`, :data:`FINITE_DIM_SYSTEM`,
    or :data:`GALERKIN_TRUNCATION`.  ``truncation_order`` records the Galerkin
    order (``None`` outside a truncation); ``system`` is a free-text label.
    """

    kind: str = GLOBAL_POLYNOMIAL
    truncation_order: int | None = None
    system: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _VALID_SCOPES:
            raise ValueError(f"scope kind must be one of {sorted(_VALID_SCOPES)}, got {self.kind!r}")

    @property
    def finite_dimensional(self) -> bool:
        """``True`` for finite-dimensional / Galerkin scopes (not the global polynomial)."""
        return self.kind != GLOBAL_POLYNOMIAL


def honesty_labels(scope: SOSScope) -> dict[str, bool]:
    """The boolean honesty flags for ``scope``; ``unproven_claim`` is always ``False``.

    None of these can be flipped by any caller-supplied value -- soundness of the
    scope labelling does not depend on trust in the caller.
    """
    return {
        "unproven_claim": False,
        "continuum_pde_claim": False,
        "regularity_claim": False,
        "finite_dimensional": scope.finite_dimensional,
    }


def seal_sos_certificate(
    certificate: SOSCertificate,
    *,
    claim: str,
    scope: SOSScope | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    r"""Seal a **proved** SOS certificate into a v1 ``positive_definite`` certificate.

    The interval ``LDL^T`` pivots become the certificate payload the Lean bridge
    turns into the ``allPivotsPos`` obligation.  Raises :class:`ValueError` on an
    inconclusive certificate -- there is nothing sound to seal.
    """
    if not certificate.certified:
        raise ValueError("cannot seal an inconclusive SOS certificate (no proof to record)")
    if scope is None:
        scope = SOSScope(GLOBAL_POLYNOMIAL)

    pivots = [Interval(lo, hi) for lo, hi in certificate.pivots]
    sealed_meta: dict[str, Any] = {
        "generator": "omnibias-sos",
        "sos": {
            "scope": scope.kind,
            "truncation_order": scope.truncation_order,
            "system": scope.system,
            "n_vars": certificate.n_vars,
            "basis_size": len(certificate.basis),
            "pd_margin": certificate.pd_margin,
            "gram": [list(row) for row in certificate.gram] if certificate.gram else None,
        },
    }
    if meta is not None:
        sealed_meta.update(meta)

    return positive_definite_certificate(
        claim,
        pivots,
        honesty=honesty_labels(scope),
        meta=sealed_meta,
    )


__all__ = [
    "FINITE_DIM_SYSTEM",
    "GALERKIN_TRUNCATION",
    "GLOBAL_POLYNOMIAL",
    "SOSScope",
    "honesty_labels",
    "seal_sos_certificate",
]
