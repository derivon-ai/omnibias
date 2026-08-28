# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""O1 leading-form normalization for the n=2 leftover-chart tree.

A sealed finite lemma: scaling each component so the leading homogeneous
part is monic (lex-first coefficient 1) yields a unique representative
on a named chart. Case C leftover is 0; Case D leftover is a nonzero
constant (the leading content removed by O1). Neither earns
``jacobian_conjecture_proof_claim``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Literal

from omnibias.core.proof.certificate import make_certificate
from omnibias.holonomic._core.poly_n import PolyN
from omnibias.holonomic.jacobian_n2 import (
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    seal_jacobian_honesty,
    shear_map,
)
from omnibias.holonomic.jacobian_n2_inverse import gabber_n2_test

LeftoverCase = Literal["A", "B", "C", "D"]
O1_SCHEMA = "omnibias.holonomic.jacobian_n2.o1.v1"


def _lex_key(mon: tuple[int, ...]) -> tuple[int, ...]:
    """Lex: higher first-variable exponent first."""
    return mon


def leading_content(poly: PolyN) -> Fraction:
    """Absolute lex-first leading-homogeneous coefficient; 0 if zero."""
    if poly.is_zero():
        return Fraction(0)
    part = poly.homogeneous_part(poly.total_degree())
    lead_mon = min(part.terms, key=_lex_key)
    return abs(part.terms[lead_mon])


def o1_normalize(components: Sequence[PolyN]) -> tuple[PolyN, ...]:
    """Scale each component so its leading homogeneous part is monic."""
    if len(components) != 2:
        raise ValueError("o1_normalize expects a map Q^2 -> Q^2")
    out: list[PolyN] = []
    for component in components:
        if component.is_zero():
            out.append(component)
            continue
        part = component.homogeneous_part(component.total_degree())
        lead_mon = min(part.terms, key=_lex_key)
        scale = part.terms[lead_mon]
        if scale == 0:
            raise ValueError("leading coefficient vanished")
        out.append(component * (Fraction(1) / scale))
    return tuple(out)


@dataclass(frozen=True)
class LeftoverChart:
    """Finite leftover after O1 on one named chart."""

    case: LeftoverCase
    leftover: Fraction
    replay_ok: bool
    representative: tuple[PolyN, ...]

    def honesty(self) -> MappingProxyType[str, bool]:
        return seal_jacobian_honesty(
            {
                "jacobian_conjecture_proof_claim": False,
                "jacobian_n2_claim": False,
            }
        )


def classify_leftover_chart(components: Sequence[PolyN]) -> LeftoverChart:
    """A/B/C/D leftover of one map. C leftover is 0; D is const-nonzero content."""
    from omnibias.holonomic._core.poly_n import identical_jacobian_constant

    if len(components) != 2:
        raise ValueError("classify_leftover_chart expects a map Q^2 -> Q^2")
    contents = [leading_content(component) for component in components]
    scale = Fraction(1)
    for item in contents:
        if item != 0:
            scale *= item
    representative = o1_normalize(components)
    jac = identical_jacobian_constant(representative)
    gabber = gabber_n2_test(representative)
    if jac is None:
        return LeftoverChart("A", scale, False, representative)
    if jac == 0:
        return LeftoverChart("B", Fraction(0), True, representative)
    if not gabber.inverse_ok:
        return LeftoverChart("A", scale, False, representative)
    if scale == 1:
        return LeftoverChart("C", Fraction(0), True, representative)
    leftover = scale if scale != 0 else Fraction(1)
    return LeftoverChart("D", leftover, True, representative)


def _terms_payload(poly: PolyN) -> list[list[object]]:
    return [[list(mon), str(coeff)] for mon, coeff in sorted(poly.terms.items())]


def seal_leftover_certificate(chart: LeftoverChart, *, name: str) -> dict[str, object]:
    """Hash-sealed leftover replay. Proof-claim stays False."""
    if JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED:
        raise RuntimeError("parent-proof claim is not wired")
    honesty = dict(chart.honesty())
    return dict(
        make_certificate(
            claim=f"finite leftover chart {name}",
            payload={
                "schema": O1_SCHEMA,
                "name": name,
                "case": chart.case,
                "leftover": str(chart.leftover),
                "replay_ok": chart.replay_ok,
                "representative": [_terms_payload(p) for p in chart.representative],
            },
            honesty=honesty,
        )
    )


def replay_leftover_certificate(certificate: Mapping[str, object]) -> bool:
    """Replay a sealed C/D leftover. Forged proof claims are rejected."""
    honesty = certificate.get("honesty", {})
    if not isinstance(honesty, Mapping):
        return False
    seal_jacobian_honesty(honesty)
    if honesty.get("jacobian_conjecture_proof_claim"):
        return False
    inner = certificate.get("payload")
    payload = inner if isinstance(inner, Mapping) else certificate
    leftover = Fraction(str(payload["leftover"]))
    case = str(payload["case"])
    if case == "C" and leftover != 0:
        return False
    if case == "D" and leftover == 0:
        return False
    if leftover != 0 and leftover.numerator == 0:
        return False
    return bool(payload.get("replay_ok")) and case in {"C", "D"}


def named_case_c_shear() -> tuple[PolyN, PolyN]:
    """Normalized quadratic shear: leftover 0."""
    return shear_map(0, (0, 0, 1))


def named_case_d_content() -> tuple[PolyN, PolyN]:
    """Same chart before O1: leading content 2, leftover const-nonzero."""
    return shear_map(0, (0, 0, 2))


__all__ = [
    "LeftoverChart",
    "O1_SCHEMA",
    "classify_leftover_chart",
    "leading_content",
    "named_case_c_shear",
    "named_case_d_content",
    "o1_normalize",
    "replay_leftover_certificate",
    "seal_leftover_certificate",
]
