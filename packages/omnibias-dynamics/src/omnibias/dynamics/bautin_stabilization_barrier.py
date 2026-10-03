# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Finite-jet barrier for an all-orders Bautin stabilization claim.

The exact focal engine can derive a finite prefix
``V_1, ..., V_N`` from a polynomial vector field, and Buchberger witnesses
can prove that every *computed* later value lies in
``(V_1, V_2, V_3)``.  That is a finite theorem.  It does not imply that every
uncomputed focal value lies in the ideal: a finite ascending ideal chain can
plateau and enlarge later unless a recurrence or termination theorem controls
the whole tail.

This module tests that distinction on Bautin's normalized quadratic family.
It derives through ``V_4`` (degree ten), certifies ``V_4`` membership exactly,
and constructs an exact formal continuation with the same finite prefix whose
next coefficient is outside the ideal.  The continuation is a logical
counterexample to a *jet-only inference*, not a claim that the adversarial
coefficient is realized by a quadratic vector field.  Proving that it cannot
occur is precisely the missing all-orders theorem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.realization_replay import source_digest
from omnibias.dynamics.bautin import (
    bautin_quadratic_family,
    center_membership,
)
from omnibias.dynamics.focal import LyapunovQuantities, lyapunov_quantities
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.holonomic._core.groebner import (
    IdealMembership,
    verify_ideal_membership,
)
from omnibias.holonomic._core.poly_n import PolyN

__all__ = [
    "FiniteBautinAudit",
    "H5BautinReport",
    "certify_finite_bautin_audit",
    "report",
    "verify_finite_bautin_audit",
]


def _poly_payload(poly: PolyN) -> list[list[object]]:
    return [
        [list(monomial), [coefficient.numerator, coefficient.denominator]]
        for monomial, coefficient in sorted(poly.terms.items())
    ]


def _membership_payload(
    degree: int,
    candidate: PolyN,
    membership: IdealMembership,
) -> dict[str, object]:
    return {
        "degree": degree,
        "candidate": _poly_payload(candidate),
        "is_member": membership.is_member,
        "cofactors": [_poly_payload(item) for item in membership.cofactors],
        "remainder": _poly_payload(membership.remainder),
    }


@dataclass(frozen=True)
class FiniteBautinAudit:
    """Exact-Q witnesses for a derived finite focal prefix and its limit."""

    quantities: LyapunovQuantities
    reference_generators: tuple[PolyN, ...]
    checked_memberships: tuple[tuple[int, PolyN, IdealMembership], ...]
    adversarial_degree: int
    adversarial_candidate: PolyN
    adversarial_nonmembership: IdealMembership
    source_digest: str
    seal: Cert

    @property
    def finite_order_stabilization_verified(self) -> bool:
        return bool(self.checked_memberships) and all(
            membership.is_member
            for _, _, membership in self.checked_memberships
        )


def certify_finite_bautin_audit(*, order: int = 10) -> FiniteBautinAudit:
    """Derive and seal finite stabilization for the classical quadratic family.

    ``order=10`` computes ``V_1`` through ``V_4``.  Higher even orders are
    accepted, but they remain finite and therefore never earn the all-orders
    honesty flag.
    """
    if type(order) is not int or order < 10 or order % 2:
        raise ValueError("order must be an even integer at least ten")
    p, q = bautin_quadratic_family()
    quantities = lyapunov_quantities(p, q, order=order)
    references = tuple(value for _, value in quantities.quantities[:3])
    checked = tuple(
        (
            degree,
            value,
            center_membership(value, references, radical=False),
        )
        for degree, value in quantities.quantities[3:]
    )
    if not checked or not all(item[2].is_member for item in checked):
        raise ValueError("the computed higher focal prefix did not stabilize")
    adversarial_degree = quantities.quantities[-1][0] + 2
    adversarial = PolyN.var(quantities.param_nvars, 0)
    nonmembership = center_membership(
        adversarial,
        references,
        radical=False,
    )
    if nonmembership.is_member:
        raise ArithmeticError("the chosen continuation coefficient lies in the ideal")
    payload = {
        "type": "finite_bautin_stabilization_audit",
        "family": "bautin_normalized_quadratic",
        "computed_order": order,
        "reference_degrees": [
            degree for degree, _ in quantities.quantities[:3]
        ],
        "reference_generators": [_poly_payload(item) for item in references],
        "checked_memberships": [
            _membership_payload(degree, value, membership)
            for degree, value, membership in checked
        ],
        "formal_continuation_counterexample": _membership_payload(
            adversarial_degree,
            adversarial,
            nonmembership,
        ),
        "scope": (
            "finite exact-Q focal prefix and formal jet-continuation barrier; "
            "the adversarial continuation is not asserted realizable by a "
            "quadratic vector field"
        ),
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim=(
            "Finite exact-Q Bautin stabilization through the computed order, "
            "plus a counterexample to inferring the all-orders tail from that "
            "finite jet alone."
        ),
        payload=payload,
        honesty={
            "finite_order_stabilization_verified": True,
            "bautin_ideal_stabilization_proved": False,
            "all_orders_focal_recurrence_proved": False,
            "actual_singular_return_map_derived": False,
            "g2_passed": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return FiniteBautinAudit(
        quantities,
        references,
        checked,
        adversarial_degree,
        adversarial,
        nonmembership,
        digest,
        seal,
    )


def verify_finite_bautin_audit(certificate: FiniteBautinAudit) -> bool:
    """Replay all membership identities and the exact nonmembership decision."""
    if not isinstance(certificate, FiniteBautinAudit):
        return False
    payload = certificate.seal.get("payload", {})
    if (
        payload.get("type") != "finite_bautin_stabilization_audit"
        or source_digest(payload) != certificate.source_digest
        or not verify_certificate_digest(certificate.seal)
    ):
        return False
    for _, value, membership in certificate.checked_memberships:
        if (
            not membership.is_member
            or not verify_ideal_membership(
                value,
                certificate.reference_generators,
                membership,
            )
        ):
            return False
    nonmembership = certificate.adversarial_nonmembership
    if (
        nonmembership.is_member
        or not verify_ideal_membership(
            certificate.adversarial_candidate,
            certificate.reference_generators,
            nonmembership,
        )
    ):
        return False
    replay = center_membership(
        certificate.adversarial_candidate,
        certificate.reference_generators,
        radical=False,
    )
    return replay == nonmembership


@dataclass(frozen=True)
class H5BautinReport:
    """H5 result: a derived finite win, but no all-orders stabilization."""

    computed_order: int
    reference_degrees: tuple[int, ...]
    checked_higher_degrees: tuple[int, ...]
    finite_order_stabilization_verified: bool
    formal_jet_counterexample_verified: bool
    all_orders_focal_recurrence_proved: bool
    actual_singular_return_map_derived: bool
    bautin_ideal_stabilization_proved: bool
    g2_passed: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-bautin-stabilization-barrier-v1",
            "computed_order": self.computed_order,
            "reference_degrees": list(self.reference_degrees),
            "checked_higher_degrees": list(self.checked_higher_degrees),
            "finite_order_stabilization_verified": (
                self.finite_order_stabilization_verified
            ),
            "formal_jet_counterexample_verified": (
                self.formal_jet_counterexample_verified
            ),
            "all_orders_focal_recurrence_proved": (
                self.all_orders_focal_recurrence_proved
            ),
            "actual_singular_return_map_derived": (
                self.actual_singular_return_map_derived
            ),
            "bautin_ideal_stabilization_proved": (
                self.bautin_ideal_stabilization_proved
            ),
            "g2_passed": self.g2_passed,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Finite exact-Q stabilization for a mechanically derived "
                "quadratic focal prefix. No recurrence for the infinite tail, "
                "no arbitrary singular return map, no G2, and no Hilbert XVI."
            ),
        }


def report(
    *,
    order: int = 10,
    audit: FiniteBautinAudit | None = None,
) -> H5BautinReport:
    """Test whether finite exact jets can honestly earn H5."""
    if audit is None:
        audit = certify_finite_bautin_audit(order=order)
    elif audit.quantities.order != order:
        raise ValueError("audit order does not match the requested report order")
    verified = verify_finite_bautin_audit(audit)
    finite = verified and audit.finite_order_stabilization_verified
    counterexample = (
        verified
        and not audit.adversarial_nonmembership.is_member
        and not audit.adversarial_nonmembership.remainder.is_zero()
    )
    all_orders = False
    actual_singular = False
    honesty = build_honesty(
        finite_order_stabilization_verified=finite,
        finite_jet_all_orders_barrier=counterexample,
        all_orders_focal_recurrence_proved=all_orders,
        actual_singular_return_map_derived=actual_singular,
        bautin_ideal_stabilization_proved=all_orders,
        g2_passed=False,
        full_hilbert16_solved=False,
    )
    return H5BautinReport(
        computed_order=order,
        reference_degrees=tuple(
            degree for degree, _ in audit.quantities.quantities[:3]
        ),
        checked_higher_degrees=tuple(
            degree for degree, _, _ in audit.checked_memberships
        ),
        finite_order_stabilization_verified=finite,
        formal_jet_counterexample_verified=counterexample,
        all_orders_focal_recurrence_proved=all_orders,
        actual_singular_return_map_derived=actual_singular,
        bautin_ideal_stabilization_proved=all_orders,
        g2_passed=False,
        honesty=honesty,
    )
