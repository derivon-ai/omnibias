# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact 2x2 admissible-stress cone membership (theory 07-10).

Lemma 4.5 / Appendix C: ``T`` lies in the interior of
``cone(v1, v2)`` iff ``det(v1, v2) != 0`` and both Cramer weights are
strictly positive. Boundary membership uses ``>=``. A parallel pair
(``det = 0``) is ``BLOCKED`` and named. No LP solver.

Kernel emission reuses ``allRatLt`` (``0 < lambda_i``). Parent flags
stay false: this is not a Clay (C)/(D) reproof and not (A)/(B).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.catalog import CatalogEntry, register_catalog
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.obligations.rational_stencil import Obligation

PAYLOAD_CONE = "stress_cone"
_INT_CAP = 10**18

Sense = Literal["gt", "ge"]
ClaimStrength = Literal["BLOCKED", "PROVED"]
Vec2 = tuple[Fraction, Fraction]


def _as_frac(x: Fraction | int | str) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def _rat_pair(q: Fraction) -> list[str]:
    return [str(q.numerator), str(q.denominator)]


def _pair_to_frac(raw: Any) -> Fraction | None:
    if isinstance(raw, Fraction):
        return raw
    if isinstance(raw, int) and not isinstance(raw, bool):
        return Fraction(raw)
    if isinstance(raw, str):
        return Fraction(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) == 2:
        try:
            return Fraction(int(raw[0]), int(raw[1]))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _vec(raw: Sequence[Any] | Vec2) -> Vec2:
    if len(raw) != 2:
        raise ValueError("vectors must be length 2")
    a = _pair_to_frac(raw[0])
    b = _pair_to_frac(raw[1])
    if a is None or b is None:
        raise ValueError("vector entries must be rational")
    return (a, b)


def det2(left: Vec2, right: Vec2) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


@dataclass(frozen=True)
class ConeQuery:
    """``T`` against generators ``v1``, ``v2`` with interior or closed sense."""

    v1: Vec2
    v2: Vec2
    T: Vec2
    sense: Sense = "gt"
    name: str = "locked_interior"

    def __post_init__(self) -> None:
        object.__setattr__(self, "v1", _vec(self.v1))
        object.__setattr__(self, "v2", _vec(self.v2))
        object.__setattr__(self, "T", _vec(self.T))
        if self.sense not in {"gt", "ge"}:
            raise ValueError(f"unknown cone sense {self.sense!r}")
        object.__setattr__(self, "name", str(self.name))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "v1": [_rat_pair(self.v1[0]), _rat_pair(self.v1[1])],
            "v2": [_rat_pair(self.v2[0]), _rat_pair(self.v2[1])],
            "T": [_rat_pair(self.T[0]), _rat_pair(self.T[1])],
            "sense": self.sense,
            "name": self.name,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ConeQuery:
        def _read(key: str) -> Vec2:
            item = raw.get(key)
            if not isinstance(item, Sequence) or isinstance(item, str | bytes):
                raise ValueError(f"ConeQuery.{key} must be a length-2 sequence")
            if len(item) != 2:
                raise ValueError(f"ConeQuery.{key} must be a length-2 sequence")
            return _vec(item)

        sense_raw = str(raw.get("sense", "gt"))
        if sense_raw not in {"gt", "ge"}:
            raise ValueError(f"unknown cone sense {sense_raw!r}")
        return cls(
            v1=_read("v1"),
            v2=_read("v2"),
            T=_read("T"),
            sense=sense_raw,  # type: ignore[arg-type]
            name=str(raw.get("name", "")),
        )


@dataclass(frozen=True)
class ConeReport:
    holds: bool
    strength: ClaimStrength
    reason: str
    lambda1: Fraction | None
    lambda2: Fraction | None
    det: Fraction

    def to_mapping(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "strength": self.strength,
            "reason": self.reason,
            "lambda1": None if self.lambda1 is None else _rat_pair(self.lambda1),
            "lambda2": None if self.lambda2 is None else _rat_pair(self.lambda2),
            "det": _rat_pair(self.det),
        }


@dataclass(frozen=True)
class ConeCertificateReport:
    certificate: Mapping[str, Any]
    obligation: Obligation
    report: ConeReport
    theorem_prover_verified: bool
    mathlib_verified: bool
    lean: LeanCheckResult | None
    wall_seconds: float | None


def check_cone(query: ConeQuery) -> ConeReport:
    """Exact Cramer weights. ``BLOCKED`` is named, never silent."""
    delta = det2(query.v1, query.v2)
    if delta == 0:
        return ConeReport(
            holds=False,
            strength="BLOCKED",
            reason="degenerate_generators",
            lambda1=None,
            lambda2=None,
            det=delta,
        )
    lam1 = det2(query.T, query.v2) / delta
    lam2 = det2(query.v1, query.T) / delta
    if query.sense == "gt":
        interior = lam1 > 0 and lam2 > 0
        if not interior:
            reason = "not_interior"
            if lam1 <= 0 and lam2 <= 0:
                reason = "opposite_cone"
            return ConeReport(False, "BLOCKED", reason, lam1, lam2, delta)
        return ConeReport(True, "PROVED", "interior", lam1, lam2, delta)
    closed = lam1 >= 0 and lam2 >= 0
    if not closed:
        reason = "outside_closed_cone"
        if lam1 <= 0 and lam2 <= 0 and not (lam1 == 0 and lam2 == 0):
            reason = "opposite_cone"
        return ConeReport(False, "BLOCKED", reason, lam1, lam2, delta)
    return ConeReport(True, "PROVED", "closed_cone", lam1, lam2, delta)


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "collapse_limit_in_lean": False,
        "function_class_in_lean": False,
        "mathlib_path_used": False,
        "navier_stokes_proof_claim": False,
        "forced_blowup_reproof_claim": False,
        "yang_mills_mass_gap_claim": False,
        "continuum_pde_claim": False,
        "continuum_claim": False,
    }


def cone_obligation(query: ConeQuery) -> Obligation:
    report = check_cone(query)
    lambdas: list[dict[str, Any]] = []
    max_abs = 1
    if report.lambda1 is not None and report.lambda2 is not None:
        for value in (report.lambda1, report.lambda2):
            lambdas.append({"lambda": _rat_pair(value)})
            max_abs = max(max_abs, abs(value.numerator), abs(value.denominator))
    payload: dict[str, Any] = {
        "type": PAYLOAD_CONE,
        "kind": PAYLOAD_CONE,
        "name": query.name,
        "holds": report.holds,
        "strength": report.strength,
        "reason": report.reason,
        "query": query.to_mapping(),
        "lambdas": lambdas,
        "det": _rat_pair(report.det),
        "external_premises": [
            "analytic classes of the slow base and the high-frequency packets",
            "construction of each particular / signed correction",
            "PDE residual estimates that the cone treats as given",
        ],
        "parent": "Navier-Stokes forced blowup (Clay C/D)",
    }
    return Obligation(PAYLOAD_CONE, payload, report.holds, max_abs)


def seal_cone_certificate(
    query: ConeQuery,
    *,
    run_lean: bool = True,
) -> ConeCertificateReport:
    """Seal the v1 certificate. ``mathlib_verified`` stays false here."""
    report = check_cone(query)
    obligation = cone_obligation(query)
    if obligation.max_abs_int > _INT_CAP:
        raise ValueError(
            f"obligation integers exceed cap {_INT_CAP}: {obligation.max_abs_int}"
        )
    cert = make_certificate(
        claim="admissible-stress cone membership is a rational identity",
        payload=obligation.payload,
        honesty=honesty_payload(),
    )
    lean: LeanCheckResult | None = None
    verified = False
    wall: float | None = None
    if run_lean:
        t0 = time.perf_counter()
        lean = check_certificate(cert)
        wall = time.perf_counter() - t0
        verified = bool(lean.verified)
    return ConeCertificateReport(
        certificate=cert,
        obligation=obligation,
        report=report,
        theorem_prover_verified=verified,
        mathlib_verified=False,
        lean=lean,
        wall_seconds=wall,
    )


def replay_cone_certificate(certificate: Mapping[str, Any]) -> bool | None:
    payload = certificate.get("payload")
    if not isinstance(payload, Mapping) or payload.get("type") != PAYLOAD_CONE:
        return None
    query_raw = payload.get("query")
    if not isinstance(query_raw, Mapping):
        return False
    try:
        query = ConeQuery.from_mapping(query_raw)
    except ValueError:
        return False
    report = check_cone(query)
    return report.holds == bool(payload.get("holds")) and report.reason == str(
        payload.get("reason", "")
    )


def digest_is_valid(cert: Mapping[str, Any]) -> bool:
    return verify_certificate_digest(cert)


def locked_interior_cone() -> ConeQuery:
    """``v1 = e1``, ``v2 = e2``, ``T = (1, 1)`` so ``lambda_i = 1``."""
    return ConeQuery(
        v1=(Fraction(1), Fraction(0)),
        v2=(Fraction(0), Fraction(1)),
        T=(Fraction(1), Fraction(1)),
        sense="gt",
        name="locked_interior",
    )


def parallel_generators_cone() -> ConeQuery:
    return ConeQuery(
        v1=(Fraction(1), Fraction(0)),
        v2=(Fraction(2), Fraction(0)),
        T=(Fraction(1), Fraction(1)),
        sense="gt",
        name="parallel_generators",
    )


def opposite_cone_query() -> ConeQuery:
    return ConeQuery(
        v1=(Fraction(1), Fraction(0)),
        v2=(Fraction(0), Fraction(1)),
        T=(Fraction(-1), Fraction(-1)),
        sense="gt",
        name="opposite_cone",
    )


def boundary_cone_query() -> ConeQuery:
    return ConeQuery(
        v1=(Fraction(1), Fraction(0)),
        v2=(Fraction(0), Fraction(1)),
        T=(Fraction(1), Fraction(0)),
        sense="ge",
        name="boundary_closed",
    )


def _register() -> None:
    register_catalog(
        CatalogEntry(
            kind=PAYLOAD_CONE,
            obligation="exact 2x2 cone membership of an admissible stress",
            parent="Navier-Stokes forced blowup (Clay C/D)",
            parent_status="already_true",
            package="omnibias.core.proof.obligations.stress_cone",
            mode="exact_replay",
            complete=True,
            existential=False,
        ),
        locked_interior_cone,
    )


_register()


__all__ = [
    "ConeCertificateReport",
    "ConeQuery",
    "ConeReport",
    "PAYLOAD_CONE",
    "boundary_cone_query",
    "check_cone",
    "cone_obligation",
    "det2",
    "digest_is_valid",
    "honesty_payload",
    "locked_interior_cone",
    "opposite_cone_query",
    "parallel_generators_cone",
    "replay_cone_certificate",
    "seal_cone_certificate",
]
