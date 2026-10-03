# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform-format barrier for the direct H3 LN/exp representation.

On the corrected kill sequence

``epsilon=1/n, sep=exp(-n**2)``,

the natural logarithmic matching data contain

``tau=epsilon*log(1/sep)=n`` and
``log(W_max/W_e)=2*n``

for the existing ``h_max=epsilon**3`` section.  Every finite truncation has
an exact two-function LN chain, but its chain-function norm is ``3*n`` and
its cell format also grows with the outer radius.  Thus this direct
representation is not uniformly bounded-format.

This is not a representation-independent impossibility theorem.  An exact,
positive normalization that preserves displacement zeros could evade the
barrier, but no such normalization of the physical return family is
constructed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.log_noetherian import (
    LNCell,
    LNChain,
    LNFiber,
    NamedPolynomial,
    ln_format,
    seal_ln_certificate,
    verify_ln_certificate,
    verify_ln_chain,
)
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "LNFormatBarrierReport",
    "TruncatedLNFormat",
    "identity_verdicts",
    "report",
    "residual_kill_log_w",
    "residual_kill_norm",
    "residual_kill_tau",
    "truncated_matching_chain",
    "truncated_matching_format",
]


def residual_kill_tau(tau: Fraction, n: Fraction) -> Fraction:
    """Exact residual for ``epsilon*log(1/sep)=n``."""
    return tau - n


def residual_kill_log_w(log_w_ratio: Fraction, n: Fraction) -> Fraction:
    """Exact residual for the existing ``epsilon**3`` section."""
    return log_w_ratio - 2 * n


def residual_kill_norm(norm: Fraction, n: Fraction) -> Fraction:
    """Exact residual for ``sup|tau| + sup|log W ratio| = 3*n``."""
    return norm - 3 * n


def _verdict(residual: Fraction) -> str:
    box = (
        Interval.point(0.0)
        if residual == 0
        else Interval.from_rational(residual)
    )
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    n = Fraction(7)
    return {
        "ln_kill_tau": _verdict(residual_kill_tau(n, n)),
        "ln_kill_log_w": _verdict(residual_kill_log_w(2 * n, n)),
        "ln_kill_norm": _verdict(residual_kill_norm(3 * n, n)),
    }


def truncated_matching_chain(n: int) -> LNChain:
    """Return the exact direct-matching chain on ``1 < |u| < n``."""
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer at least two")
    cell = LNCell(
        (LNFiber("Annulus", (Fraction(1), Fraction(n))),),
        real_part=True,
    )
    u = SparsePolynomial.variable(1, 0)
    tau = NamedPolynomial("tau", u)
    log_w = NamedPolynomial("log_W_ratio", 2 * u)
    tau_ref = SparsePolynomial.variable(2, 0)
    log_w_ref = SparsePolynomial.variable(2, 1)
    return LNChain(
        cell=cell,
        functions=(tau, log_w),
        closure_matrix=((tau_ref,), (log_w_ref,)),
        claimed_derivatives=((u,), (2 * u,)),
    )


@dataclass(frozen=True)
class TruncatedLNFormat:
    """Certified finite format data for one bounded truncation."""

    n: int
    chain_length: int
    closure_degree: int
    coefficient_norm: Fraction
    domain_inner: Fraction
    domain_outer: Fraction
    chain_sup_norm: Fraction
    combinatorial_format: Fraction
    total_format_upper: Fraction
    chain_valid: bool
    certificate_valid: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "n": self.n,
            "chain_length": self.chain_length,
            "closure_degree": self.closure_degree,
            "coefficient_norm": [
                self.coefficient_norm.numerator,
                self.coefficient_norm.denominator,
            ],
            "analytic_domain": {
                "kind": "annulus",
                "inner": [self.domain_inner.numerator, self.domain_inner.denominator],
                "outer": [self.domain_outer.numerator, self.domain_outer.denominator],
            },
            "chain_sup_norm": [
                self.chain_sup_norm.numerator,
                self.chain_sup_norm.denominator,
            ],
            "combinatorial_format": [
                self.combinatorial_format.numerator,
                self.combinatorial_format.denominator,
            ],
            "total_format_upper": [
                self.total_format_upper.numerator,
                self.total_format_upper.denominator,
            ],
            "chain_valid": self.chain_valid,
            "certificate_valid": self.certificate_valid,
        }


def truncated_matching_format(n: int) -> TruncatedLNFormat:
    """Certify all four finite-format data on one truncation."""
    chain = truncated_matching_chain(n)
    sup_norm = Fraction(3 * n)
    supplied_sup = Interval.from_rational(sup_norm)
    format_report = ln_format(
        chain,
        cell_format=Fraction(n + 1),
        sup_bound=supplied_sup,
    )
    certificate = seal_ln_certificate(
        chain,
        cell_format=Fraction(n + 1),
        sup_bound=supplied_sup,
    )
    total_upper = format_report.combinatorial_part + sup_norm
    return TruncatedLNFormat(
        n=n,
        chain_length=2,
        closure_degree=1,
        coefficient_norm=Fraction(2),
        domain_inner=Fraction(1),
        domain_outer=Fraction(n),
        chain_sup_norm=sup_norm,
        combinatorial_format=format_report.combinatorial_part,
        total_format_upper=total_upper,
        chain_valid=verify_ln_chain(chain),
        certificate_valid=verify_ln_certificate(certificate),
    )


@dataclass(frozen=True)
class LNFormatBarrierReport:
    """Finite truncations plus the exact failure of a uniform direct format."""

    identities: Mapping[str, str]
    truncations: Sequence[TruncatedLNFormat]
    finite_truncations_certified: bool
    direct_chain_length_uniform: bool
    direct_coefficients_uniform: bool
    direct_domain_uniform: bool
    direct_norm_uniform: bool
    direct_bounded_format: bool
    normalized_zero_equivalent_route_open: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-ln-format-barrier-v1",
            "identities": dict(self.identities),
            "truncations": [item.to_payload() for item in self.truncations],
            "finite_truncations_certified": self.finite_truncations_certified,
            "direct_chain_length_uniform": self.direct_chain_length_uniform,
            "direct_coefficients_uniform": self.direct_coefficients_uniform,
            "direct_domain_uniform": self.direct_domain_uniform,
            "direct_norm_uniform": self.direct_norm_uniform,
            "direct_bounded_format": self.direct_bounded_format,
            "actual_return_ln_membership_proved": False,
            "normalized_zero_equivalent_route_open": (
                self.normalized_zero_equivalent_route_open
            ),
            "g3_passed": False,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact obstruction to the direct W/tau bounded-format chain "
                "on the corrected kill sequence. Not an impossibility theorem "
                "for normalized LN representations, not G3, and not Hilbert XVI."
            ),
        }


def report(
    truncation_ns: Sequence[int] = (4, 8, 16, 32),
) -> LNFormatBarrierReport:
    """Replay the direct-format growth on an increasing exact-Q sequence."""
    if (
        not truncation_ns
        or any(type(n) is not int or n < 2 for n in truncation_ns)
        or any(
            left >= right
            for left, right in zip(truncation_ns, truncation_ns[1:], strict=False)
        )
    ):
        raise ValueError("truncation_ns must be strictly increasing integers >= 2")
    truncations = tuple(truncated_matching_format(n) for n in truncation_ns)
    finite = all(
        item.chain_valid
        and item.certificate_valid
        and item.chain_sup_norm == 3 * item.n
        for item in truncations
    )
    lengths = {item.chain_length for item in truncations}
    degrees = {item.closure_degree for item in truncations}
    coefficients = {item.coefficient_norm for item in truncations}
    norm_growth = all(
        left.chain_sup_norm < right.chain_sup_norm
        for left, right in zip(truncations, truncations[1:], strict=False)
    )
    domain_growth = all(
        left.domain_outer < right.domain_outer
        for left, right in zip(truncations, truncations[1:], strict=False)
    )
    direct_length_uniform = lengths == {2}
    direct_coefficients_uniform = degrees == {1} and coefficients == {Fraction(2)}
    direct_domain_uniform = not domain_growth
    direct_norm_uniform = not norm_growth
    direct_bounded = (
        finite
        and direct_length_uniform
        and direct_coefficients_uniform
        and direct_domain_uniform
        and direct_norm_uniform
    )
    honesty = build_honesty(
        direct_ln_format_barrier=finite and norm_growth and domain_growth,
        actual_return_ln_membership_proved=False,
        normalized_ln_return_member=False,
        g3_passed=False,
        full_hilbert16_solved=False,
    )
    return LNFormatBarrierReport(
        identities=identity_verdicts(),
        truncations=truncations,
        finite_truncations_certified=finite,
        direct_chain_length_uniform=direct_length_uniform,
        direct_coefficients_uniform=direct_coefficients_uniform,
        direct_domain_uniform=direct_domain_uniform,
        direct_norm_uniform=direct_norm_uniform,
        direct_bounded_format=direct_bounded,
        normalized_zero_equivalent_route_open=True,
        honesty=honesty,
    )
