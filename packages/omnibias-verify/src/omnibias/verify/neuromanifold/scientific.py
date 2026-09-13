# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Scientific consumers with explicit finite/box scope and bound operands.

Interval providers are mathematical contracts: their callbacks must enclose
the declared model and derivatives on the entire input box. Rational quantum
geometry is recomputed from a complete *declared finite distribution*; sampled
Monte Carlo estimates are never accepted as exact score operands.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    positive_definite_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lift import integer_null_space
from omnibias.core.verified.eig_operator import interval_ldlt_pivots
from omnibias.core.verified.interval import Interval

Box = tuple[Interval, ...]
IntervalMatrix = Sequence[Sequence[Interval]]
JacobianProvider = Callable[[Box], IntervalMatrix]
ErrorProvider = Callable[[Box], Mapping[int, Interval]]
Status = Literal["proved", "inconclusive"]
Q = Fraction


def _box(box: Sequence[Interval]) -> Box:
    out = tuple(box)
    if not out or any(not isinstance(x, Interval) or not math.isfinite(x.lo + x.hi) for x in out):
        raise ValueError("nonempty finite interval domain required")
    return out


def _encode_box(box: Box) -> list[list[float]]:
    return [[x.lo, x.hi] for x in box]


@dataclass(frozen=True)
class IdentifiabilityCertificate:
    status: Status
    gram: tuple[tuple[Interval, ...], ...]
    eigenvalue_floor: float
    certificate: Cert
    assumptions: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return self.status == "proved"


def certify_identifiability(
    whitened_jacobian: JacobianProvider,
    parameter_box: Sequence[Interval],
    *,
    physical: Sequence[int],
    nuisance: Sequence[int] = (),
    eigenvalue_floor: float = 0.0,
    provider_assumption: str,
) -> IdentifiabilityCertificate:
    """Sufficient uniform first-order identifiability with named nuisance axes.

    Prove J_selected.T J_selected > floor*I on the box. Full column rank of
    physical+nuisance columns implies positive profiled physical information.
    This sufficient test is inconclusive for redundant nuisance coordinates;
    it does not claim global injectivity or statistical parameter coverage.
    """
    domain = _box(parameter_box)
    indices = tuple(physical) + tuple(nuisance)
    if (
        not physical
        or len(set(indices)) != len(indices)
        or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices)
    ):
        raise ValueError("nonempty physical and distinct nonnegative column indices required")
    if (
        not provider_assumption.strip()
        or eigenvalue_floor < 0
        or not math.isfinite(eigenvalue_floor)
    ):
        raise ValueError("explicit provider assumption and nonnegative finite floor required")
    raw = [list(row) for row in whitened_jacobian(domain)]
    if (
        not raw
        or not raw[0]
        or any(len(row) != len(raw[0]) for row in raw)
        or max(indices) >= len(raw[0])
    ):
        raise ValueError("rectangular nonempty Jacobian with selected columns required")
    if any(
        not isinstance(v, Interval) or not math.isfinite(v.lo + v.hi) for row in raw for v in row
    ):
        raise TypeError("Jacobian provider must return finite sound Interval entries")
    selected = [[row[i] for i in indices] for row in raw]
    n = len(indices)
    gram = [
        [sum((row[i] * row[j] for row in selected), Interval.point(0.0)) for j in range(n)]
        for i in range(n)
    ]
    shifted = [
        [v - eigenvalue_floor if i == j else v for j, v in enumerate(row)]
        for i, row in enumerate(gram)
    ]
    pivots = interval_ldlt_pivots(shifted)
    passed = pivots is not None and all(p.lo > 0 for p in pivots)
    status: Status = "proved" if passed else "inconclusive"
    assumptions = (
        provider_assumption,
        "C1 model and fixed positive observation covariance on the declared box",
    )
    meta = {
        "kind": "scientific_identifiability",
        "domain": _encode_box(domain),
        "physical": list(physical),
        "nuisance": list(nuisance),
        "whitened_jacobian": [_encode_box(tuple(row)) for row in raw],
        "floor": eigenvalue_floor,
        "status": status,
        "assumptions": list(assumptions),
        "scope": "uniform selected first-order information; global injectivity and sampling coverage unproved",
    }
    if passed:
        assert pivots is not None
        cert = positive_definite_certificate(
            "selected physical+nuisance information exceeds the declared floor",
            pivots,
            matrix=shifted,
            meta=meta,
        )
    else:
        cert = make_certificate(
            claim="identifiability test is inconclusive",
            payload={"type": "scientific_inconclusive", "status": status},
            meta=meta,
        )
    return IdentifiabilityCertificate(
        status, tuple(tuple(r) for r in gram), eigenvalue_floor, cert, assumptions
    )


@dataclass(frozen=True)
class ReductionCertificate:
    """Errors bound accepted regions only; unresolved regions prevent certification."""

    status: Status
    accepted_regions: tuple[Box, ...]
    unresolved_regions: tuple[Box, ...]
    errors: tuple[tuple[int, float], ...]
    certificate: Cert
    assumptions: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return self.status == "proved"


def certify_reduction(
    error_enclosure: ErrorProvider,
    domain: Sequence[Interval],
    *,
    orders: Sequence[int] = (0,),
    error_budget: float = 1e-6,
    max_boxes: int = 256,
    provider_assumption: str,
) -> ReductionCertificate:
    """Uniform value/derivative error coverage by adaptive interval subdivision.

    Provider maps each box to full-minus-reduced derivative intervals for every
    requested order. Nonuniform/singular regions remain unresolved. A limit is
    not inferred from a shrinking sequence of sampled errors.
    """
    box = _box(domain)
    requested = tuple(orders)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not isinstance(i, int) or i < 0 for i in requested)
    ):
        raise ValueError("distinct nonnegative derivative orders required")
    if (
        error_budget < 0
        or not math.isfinite(error_budget)
        or max_boxes < 1
        or not provider_assumption.strip()
    ):
        raise ValueError(
            "finite error budget, positive box budget, explicit provider assumption required"
        )
    pending = [box]
    accepted = []
    unresolved = []
    bounds = {i: 0.0 for i in requested}
    records = []
    evaluated = 0
    while pending and evaluated < max_boxes:
        region = pending.pop()
        evaluated += 1
        errors: dict[int, Interval] | None
        try:
            errors = dict(error_enclosure(region))
        except (ZeroDivisionError, OverflowError):
            errors = None
        if errors is not None and any(
            i not in errors or not isinstance(errors[i], Interval) for i in requested
        ):
            raise TypeError("provider must return an Interval for each requested derivative order")
        passed = errors is not None and all(
            math.isfinite(errors[i].mag) and errors[i].mag <= error_budget for i in requested
        )
        if passed:
            assert errors is not None
            accepted.append(region)
            for i in requested:
                bounds[i] = max(bounds[i], errors[i].mag)
            records.append(
                {
                    "box": _encode_box(region),
                    "errors": [[i, errors[i].lo, errors[i].hi] for i in requested],
                }
            )
            continue
        axis = max(range(len(region)), key=lambda i: region[i].width)
        iv = region[axis]
        mid = iv.mid
        if mid <= iv.lo or mid >= iv.hi:
            unresolved.append(region)
            continue
        left = list(region)
        right = list(region)
        left[axis] = Interval(iv.lo, mid)
        right[axis] = Interval(mid, iv.hi)
        pending.extend((tuple(right), tuple(left)))
    unresolved.extend(pending)
    status: Status = "proved" if not unresolved else "inconclusive"
    assumptions = (
        provider_assumption,
        "callbacks enclose the named value/derivative errors on every full box",
    )
    cert = make_certificate(
        claim="uniform effective-model error coverage"
        if status == "proved"
        else "effective-model error coverage is incomplete",
        payload={
            "type": "scientific_reduction",
            "status": status,
            "domain": _encode_box(box),
            "orders": list(requested),
            "budget": error_budget,
            "accepted": records,
            "unresolved": [_encode_box(r) for r in unresolved],
        },
        meta={
            "assumptions": list(assumptions),
            "evaluated_boxes": evaluated,
            "scope": "declared box and derivative orders",
        },
    )
    return ReductionCertificate(
        status, tuple(accepted), tuple(unresolved), tuple(bounds.items()), cert, assumptions
    )


def _rational(value: object) -> Q:
    if isinstance(value, bool) or not isinstance(value, (int, Q)):
        raise TypeError("exact int/Fraction operands required; Monte Carlo floats are not a proof")
    return Q(value)


def _rows(values: Sequence[Sequence[object]], width: int | None = None) -> list[list[Q]]:
    rows = [[_rational(v) for v in row] for row in values]
    if not rows:
        return []
    n = len(rows[0]) if width is None else width
    if n == 0 or any(len(row) != n for row in rows):
        raise ValueError("rectangular rational rows required")
    return rows


def _kernel(rows: Sequence[Sequence[Q]]) -> list[list[Q]]:
    integer = []
    for row in rows:
        scale = math.lcm(*(v.denominator for v in row))
        integer.append([int(v * scale) for v in row])
    return [[Q(v) for v in row] for row in integer_null_space(integer)]


def _rank(rows: Sequence[Sequence[Q]], columns: int) -> int:
    return 0 if not rows else columns - len(_kernel(rows))


def _strings(rows: Sequence[Sequence[Q]]) -> list[list[str]]:
    return [[str(v) for v in row] for row in rows]


@dataclass(frozen=True)
class QuantumGeometryCertificate:
    status: Status
    covariance: tuple[tuple[Q, ...], ...]
    rank: int
    null_basis: tuple[tuple[Q, ...], ...]
    physical_basis: tuple[tuple[Q, ...], ...]
    projected: tuple[tuple[Q, ...], ...]
    complete_null_space: bool
    certificate: Cert

    @property
    def certified(self) -> bool:
        return self.status == "proved"


def certify_quantum_geometry(
    scores: Sequence[Sequence[object]],
    weights: Sequence[object],
    *,
    imaginary_scores: Sequence[Sequence[object]] | None = None,
    null_basis: Sequence[Sequence[object]] | None = None,
    physical_basis: Sequence[Sequence[object]] | None = None,
    sampling_kind: str = "exact_finite_distribution",
) -> QuantumGeometryCertificate:
    """Exact real QGT, complete kernel and physical quotient for a rational table.

    Rows enumerate the declared finite distribution with exact rational
    nonnegative weights. Scores may have rational real/imaginary components.
    This certifies that finite table, not an unprovided population or model
    derivative identity. A physical basis is a list of P-component vectors;
    together with the complete kernel it must span all parameter directions.
    """
    if sampling_kind != "exact_finite_distribution":
        raise ValueError("only exact finite distributions can earn this certificate")
    real = _rows(scores)
    if not real:
        raise ValueError("nonempty finite score table required")
    n, p = len(real), len(real[0])
    w = [_rational(v) for v in weights]
    if len(w) != n or any(v < 0 for v in w) or sum(w) <= 0:
        raise ValueError("nonnegative exact weight per row with positive total required")
    total = sum(w, Q(0))
    w = [v / total for v in w]
    imaginary = (
        [[Q(0)] * p for _ in range(n)] if imaginary_scores is None else _rows(imaginary_scores, p)
    )
    if len(imaginary) != n:
        raise ValueError("imaginary score table must match real table")
    centered = []
    for table in (real, imaginary):
        mean = [sum((w[k] * table[k][i] for k in range(n)), Q(0)) for i in range(p)]
        centered.append([[table[k][i] - mean[i] for i in range(p)] for k in range(n)])
    cov = [
        [
            sum(
                (
                    w[k]
                    * (
                        centered[0][k][i] * centered[0][k][j]
                        + centered[1][k][i] * centered[1][k][j]
                    )
                    for k in range(n)
                ),
                Q(0),
            )
            for j in range(p)
        ]
        for i in range(p)
    ]
    exact_kernel = _kernel(cov)
    rank = p - len(exact_kernel)
    null = exact_kernel if null_basis is None else _rows(null_basis, p)
    valid_null = all(
        all(sum(cov[i][j] * v[j] for j in range(p)) == 0 for i in range(p)) for v in null
    )
    complete = valid_null and len(null) == p - rank and _rank(null, p) == p - rank
    if physical_basis is None:
        physical = []
        span = list(exact_kernel)
        current = len(exact_kernel)
        for i in range(p):
            e = [Q(int(i == j)) for j in range(p)]
            if _rank([*span, e], p) > current:
                physical.append(e)
                span.append(e)
                current += 1
    else:
        physical = _rows(physical_basis, p)
    covers = complete and len(physical) == rank and _rank([*null, *physical], p) == p
    projected = [
        [sum((u[i] * cov[i][j] * v[j] for i in range(p) for j in range(p)), Q(0)) for v in physical]
        for u in physical
    ]
    # Exact rational LDL (no numerical threshold). Positivity on the quotient
    # plus a complete null basis distinguishes redundancy from lost directions.
    work = [row.copy() for row in projected]
    pivots = []
    positive = True
    for k in range(len(work)):
        pivot = work[k][k]
        pivots.append(pivot)
        if pivot <= 0:
            positive = False
            break
        for i in range(k + 1, len(work)):
            for j in range(k + 1, len(work)):
                work[i][j] -= work[i][k] * work[k][j] / pivot
    status: Status = "proved" if covers and positive else "inconclusive"
    payload = {
        "type": "rational_quantum_geometry",
        "status": status,
        "scores": _strings(real),
        "imaginary_scores": _strings(imaginary),
        "weights": [str(v) for v in w],
        "null_basis": _strings(null),
        "physical_basis": _strings(physical),
        "covariance": _strings(cov),
        "projected": _strings(projected),
        "pivots": [str(v) for v in pivots],
        "rank": rank,
        "complete_null_space": complete,
    }
    cert = make_certificate(
        claim="exact finite-distribution quantum covariance and quotient",
        payload=payload,
        meta={
            "scope": "declared rational score table only",
            "sampling_kind": sampling_kind,
            "assumptions": [
                "provided scores are the intended model's log derivatives; table is the declared finite distribution"
            ],
        },
    )
    return QuantumGeometryCertificate(
        status,
        tuple(map(tuple, cov)),
        rank,
        tuple(map(tuple, null)),
        tuple(map(tuple, physical)),
        tuple(map(tuple, projected)),
        complete,
        cert,
    )


def replay_quantum_geometry(certificate: Mapping[str, Any]) -> bool:
    """Recompute operand-bound covariance/kernel claims even after resealing."""
    if not verify_certificate_digest(certificate):
        return False
    try:
        payload = certificate["payload"]
        if payload["type"] != "rational_quantum_geometry":
            return False

        def parse(name: str) -> list[list[Q]]:
            return [[Q(v) for v in row] for row in payload[name]]

        result = certify_quantum_geometry(
            parse("scores"),
            [Q(v) for v in payload["weights"]],
            imaginary_scores=parse("imaginary_scores"),
            null_basis=parse("null_basis"),
            physical_basis=parse("physical_basis"),
        )
        meta = certificate["meta"]
        expected = result.certificate
        return bool(
            expected["payload"] == payload
            and expected["claim"] == certificate["claim"]
            and expected["honesty"] == certificate["honesty"]
            and all(
                meta.get(key) == expected["meta"][key]
                for key in ("scope", "sampling_kind", "assumptions")
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


__all__ = [
    "IdentifiabilityCertificate",
    "QuantumGeometryCertificate",
    "ReductionCertificate",
    "certify_identifiability",
    "certify_quantum_geometry",
    "certify_reduction",
    "replay_quantum_geometry",
]
