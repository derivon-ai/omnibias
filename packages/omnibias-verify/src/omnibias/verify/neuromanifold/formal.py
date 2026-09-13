# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite operand replay for geometric certificates, with earned build flags."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.confluence import initialize_moments
from omnibias.core.proof.certificate import Cert, decode_interval, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.realization_algebra_replay import moment_identity_replay_certificate
from omnibias.core.proof.realization_replay import (
    interval_error_budget_certificate,
    interval_ldlt_replay_certificate,
    krawczyk_replay_certificate,
    polynomial_evaluation_certificate,
    polynomial_inequality_certificate,
    rank_replay_certificate,
    verify_replay_certificate,
)
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.realization.rank import certify_matrix_rank
from omnibias.core.realization.transition import TransitionProposal
from omnibias.core.verified.interval import Interval
from omnibias.geometry.neuromanifold import RegularQuotientChart
from omnibias.verify._core.newton import float_inverse

from .confluence import ConfluenceCertificate, replay_confluence_certificate
from .minima import MinimumCertificate


@dataclass(frozen=True)
class FormalGeometryReplay:
    certificates: tuple[Cert, ...]
    kernel: tuple[LeanCheckResult, ...]
    mathlib: tuple[LeanCheckResult, ...]
    dependencies: tuple[str, ...]

    @property
    def theorem_prover_verified(self) -> bool:
        return bool(self.kernel) and all(result.verified for result in self.kernel)

    @property
    def mathlib_verified(self) -> bool:
        return bool(self.mathlib) and all(result.verified for result in self.mathlib)


def replay_finite_geometry(
    certificates: Sequence[Cert],
    *,
    mathlib: bool = False,
    dependencies: tuple[str, ...] = (),
    timeout: int = 120,
) -> FormalGeometryReplay:
    """Serial actual builds. Flags cover the emitted finite obligations only."""
    if not certificates or any(not verify_replay_certificate(c) for c in certificates):
        raise ValueError("valid operand-bound finite replay certificates are required")
    kernel_results = tuple(check_certificate(c, timeout=timeout) for c in certificates)
    mathlib_results: list[LeanCheckResult] = []
    if mathlib:
        try:
            from omnibias.formal.mathlib_check import check_certificate as mathlib_check
        except ImportError:
            mathlib_results = [
                LeanCheckResult(False, False, "", "optional formal package unavailable")
            ]
        else:
            for certificate in certificates:
                result = mathlib_check(certificate, timeout=timeout)
                mathlib_results.append(
                    LeanCheckResult(
                        result.verified, result.available, result.obligation, result.detail
                    )
                )
    return FormalGeometryReplay(
        tuple(certificates), kernel_results, tuple(mathlib_results), dependencies
    )


def minimum_replay_certificates(result: MinimumCertificate) -> tuple[Cert, ...]:
    """Recompute Krawczyk products and interval LDL from the recorded operands."""
    if result.status != "proved" or not verify_certificate_digest(result.certificate):
        raise ValueError("a sealed successful minimum result is required")
    certificate: Mapping[str, Any] = result.certificate
    if (result.claim not in ("slice_minimum", "quotient_minimum", "morse_bott_minimum")
            or certificate["meta"].get("kind") != result.claim
            or certificate["meta"].get("status") != "proved"):
        raise ValueError("minimum claim or status differs from the sealed source")
    chart_certificates: tuple[Cert, ...] = ()
    if result.claim != "slice_minimum":
        chart_meta = certificate["meta"]
        chart = RegularQuotientChart(
            tuple(tuple(Fraction(v) for v in row) for row in chart_meta["source"]),
            tuple(chart_meta["independent_columns"]),
            tuple(tuple(Fraction(v) for v in row) for row in chart_meta["projection"]),
            tuple(tuple(Fraction(v) for v in row) for row in chart_meta["kernel"]),
            chart_meta["observation_scope"],
        )
        if not chart.verify() or chart.dim != len(result.search_box):
            raise ValueError("invalid source quotient factorization")
        witness = certify_matrix_rank(chart.source)
        witness = replace(witness,
                          left=tuple(tuple(row[j] for j in chart.independent_columns)
                                     for row in chart.source), right=chart.projection)
        chart_certificates = (rank_replay_certificate(witness),)
        certificate = certificate["meta"]["reduced_certificate"]
    if not verify_certificate_digest(certificate):
        raise ValueError("invalid dependent reduced certificate")
    meta = certificate["meta"]
    if (meta.get("kind") != "slice_minimum" or meta.get("status") != "proved"
            or len(meta.get("parameter_names", ())) != len(result.search_box)):
        raise ValueError("dependent reduced objective has incompatible scope or dimension")
    if (
        [[x.lo, x.hi] for x in result.search_box] != meta["search_box"]
        or [[[x.lo, x.hi] for x in row] for row in result.hessian] != meta["hessian"]
        or result.stationary_box is None
        or [[x.lo, x.hi] for x in result.stationary_box] != meta["stationary_box"]
    ):
        raise ValueError("result operands differ from the dependent source certificate")
    box = tuple(tuple(Fraction(x) for x in row) for row in meta["search_box"])
    hessian = tuple(
        tuple(tuple(Fraction(x) for x in entry) for entry in row) for row in meta["hessian"]
    )
    gradient = tuple(tuple(Fraction(x) for x in row) for row in meta["gradient_at_center"])
    # Match the floating preconditioner used by the interval producer. Its
    # entries are then exact dyadics; the proof needs no exact matrix inverse.
    original_box = result.search_box
    center = tuple(Fraction(x.mid) for x in original_box)
    inverse = float_inverse([[entry.mid for entry in row] for row in result.hessian])
    if inverse is None:
        raise ValueError("singular recorded preconditioner")
    preconditioner = tuple(tuple(Fraction(x) for x in row) for row in inverse)
    return chart_certificates + (
        krawczyk_replay_certificate(
            center, gradient, hessian, preconditioner, box, require_contraction=False,
            stationary_box=tuple((Fraction(x.lo), Fraction(x.hi)) for x in result.stationary_box),
        ),
        interval_ldlt_replay_certificate(hessian),
    )


def formalize_minimum(
    result: MinimumCertificate, *, mathlib: bool = False, timeout: int = 120
) -> FormalGeometryReplay:
    """Build finite inclusion/curvature checks; analytic hypotheses stay explicit."""
    return replay_finite_geometry(
        minimum_replay_certificates(result),
        mathlib=mathlib,
        dependencies=result.dependencies,
        timeout=timeout,
    )


def _transition_coordinate_replays(meta: Mapping[str, Any]) -> tuple[Cert, ...]:
    """Bind affine biases, centered moments and stored-coordinate rounding."""
    if meta["kind"] != "confluence_transition":
        return ()
    snapshot = meta["source"]
    proposal = TransitionProposal.from_json(meta["proposal"])
    if proposal.kind not in ("centered_pair", "duplicate_merge", "cluster_to_moments"):
        return ()
    ids = [int(v) for v in snapshot["slot_ids"]["values"]]
    slots = tuple(ids.index(i) for i in proposal.slot_ids)
    count = len(slots)
    updates = {(u.name, u.index): u.value for u in proposal.updates}

    def original(name: str) -> tuple[Fraction, ...]:
        return tuple(Fraction(snapshot[name]["values"][i]) for i in slots)

    coefficients, centers, scales = original("weights"), original("centers"), original("scales")
    offsets = tuple(-scale * center for scale, center in zip(scales, centers, strict=True))
    center = sum(offsets, Fraction()) / count
    pair = proposal.kind in ("centered_pair", "duplicate_merge")
    raw_order = 1 if pair else updates["orders", slots[0]]
    if isinstance(raw_order, tuple):
        raise ValueError("malformed proposed order")
    order = int(raw_order)
    initialization = initialize_moments(offsets, coefficients, center=center, order=order)
    stored_center_raw = updates["centers", slots[0]]
    if isinstance(stored_center_raw, tuple):
        raise ValueError("malformed proposed center")
    stored_center = Fraction(stored_center_raw)
    stored_rho: tuple[Fraction, ...]
    if pair:
        raw_moments = (updates["weights", slots[0]], updates["odd_moments", slots[0]])
        if any(isinstance(v, tuple) for v in raw_moments):
            raise ValueError("malformed proposed pair moments")
        stored_moments = tuple(Fraction(v) for v in raw_moments if not isinstance(v, tuple))
        rho_raw = updates["rho", slots[0]]
        if isinstance(rho_raw, tuple):
            raise ValueError("malformed proposed spread")
        stored_rho = (Fraction(rho_raw),)
    else:
        row = updates["moment_coefficients", slots[0]]
        if not isinstance(row, tuple):
            raise ValueError("malformed proposed moment row")
        stored_moments, stored_rho = tuple(Fraction(v) for v in row), ()
    point = coefficients + centers + scales + (stored_center,) + stored_moments + stored_rho
    variables = tuple(SparsePolynomial.variable(len(point), i) for i in range(len(point)))
    biases = tuple(-variables[2 * count + i] * variables[count + i] for i in range(count))
    mean = sum(biases, SparsePolynomial.constant(len(point), 0)) * Fraction(1, count)
    moments = []
    from math import factorial

    for k in range(order + 1):
        moment = SparsePolynomial.constant(len(point), 0)
        for i in range(count):
            moment += variables[i] * (biases[i] - mean) ** k * Fraction(1, factorial(k))
        moments.append(moment)
    common_scales = tuple(variables[2 * count + i] - variables[2 * count] for i in range(1, count))
    identities = biases + (mean,) + common_scales + tuple(moments)
    expected = offsets + (center,) + (Fraction(0),) * (count - 1) + initialization.exact_moments
    differences = [mean + variables[2 * count] * variables[3 * count]]
    differences.extend(moment - variables[3 * count + 1 + k] for k, moment in enumerate(moments))
    # The target bank stores a fixed-capacity moment row. Unused higher terms
    # must remain exactly zero, rather than being silently omitted from replay.
    for k in range(order + 1, len(stored_moments)):
        identities += (variables[3 * count + 1 + k],)
        expected += (Fraction(0),)
    if pair:
        rho = (biases[0] - biases[1]) ** 2 * Fraction(1, 4)
        identities += (rho,)
        expected += ((offsets[0] - offsets[1]) ** 2 / 4,)
        differences.append(rho - variables[-1])
    inequalities: list[SparsePolynomial] = []
    for difference in differences:
        value = difference.evaluate(point)
        enclosure = Interval.from_rational(value)
        inequalities.extend(
            (difference - Fraction(enclosure.lo), -difference + Fraction(enclosure.hi))
        )
    relation: Literal["ge"] = "ge"
    relations = (relation,) * len(inequalities)
    return (
        polynomial_evaluation_certificate(identities, point, expected),
        moment_identity_replay_certificate(
            offsets, coefficients, center=center, expected_moments=initialization.exact_moments
        ),
        polynomial_inequality_certificate(inequalities, point, relations),
    )


def formalize_confluence(
    result: ConfluenceCertificate, *, mathlib: bool = False, timeout: int = 120
) -> FormalGeometryReplay:
    """Build source-coordinate identities, conversion arithmetic and error budgets.

    Taylor bounds and their propagation are rigorous analytic dependencies;
    finite coordinate identities are recomputed from the stored source operands.
    """
    if not result.accepted or not replay_confluence_certificate(result.certificate):
        raise ValueError("a sealed successful confluence result is required")
    meta = result.certificate["meta"]
    maximum = max(error.hi for _, error in result.errors)
    if (
        result.budget != meta["error_budget"]
        or maximum != decode_interval(result.certificate["payload"]["interval"]).hi
        or [[n, e.lo, e.hi] for n, e in result.errors] != meta["errors"]
    ):
        raise ValueError("result budget or errors differ from the source certificate")
    certificates = _transition_coordinate_replays(meta) + tuple(
        interval_error_budget_certificate(
            (Fraction(error.lo), Fraction(error.hi)), Fraction(result.budget)
        )
        for _, error in result.errors
    )
    return replay_finite_geometry(
        certificates,
        mathlib=mathlib,
        dependencies=result.dependencies
        + ("rigorous Taylor and conversion-error bound propagation",),
        timeout=timeout,
    )

__all__ = [
    "FormalGeometryReplay",
    "formalize_confluence",
    "formalize_minimum",
    "minimum_replay_certificates",
    "replay_finite_geometry",
]
