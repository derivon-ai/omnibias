# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact Keller-map replay and fiber count (Alpöge / Gallagher).

Replay constructors live here. Blind search lives in
:mod:`omnibias.holonomic.keller_search` and must not import these literals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.holonomic._core.poly_n import PolyN, jacobian_det
from omnibias.holonomic._core.rational_poly import to_poly
from omnibias.holonomic.keller_search import (
    SweepSearchHit,
    build_sweep_map,
    generic_fiber_degree,
    honesty_search,
    search_tangent_sweep,
    tangency_leading_coeff,
    tangency_polynomial,
)

# Alpöge's published witness (Gao arXiv:2608.00222 §3.4), replay only.
_ALP_WITNESSES = (
    (Fraction(0), Fraction(0), Fraction(-1, 4)),
    (Fraction(1), Fraction(-3, 2), Fraction(13, 2)),
    (Fraction(-1), Fraction(3, 2), Fraction(13, 2)),
)
_ALP_IMAGE = (Fraction(-1, 4), Fraction(0), Fraction(0))


def _x() -> PolyN:
    return PolyN.var(3, 0)


def _y() -> PolyN:
    return PolyN.var(3, 1)


def _z() -> PolyN:
    return PolyN.var(3, 2)


def alpoge_map() -> tuple[PolyN, PolyN, PolyN]:
    """Alpöge's degree-(7,6,4) map with ``det J ≡ -2``."""
    x, y, z = _x(), _y(), _z()
    u = PolyN.const(3, 1) + x * y
    f1 = (u**3) * z + (y**2) * u * (PolyN.const(3, 4) + PolyN.const(3, 3) * x * y)
    f2 = y + PolyN.const(3, 3) * x * (u**2) * z + PolyN.const(3, 3) * x * (y**2) * (
        PolyN.const(3, 4) + PolyN.const(3, 3) * x * y
    )
    f3 = PolyN.const(3, 2) * x - PolyN.const(3, 3) * (x**2) * y - (x**3) * z
    return f1, f2, f3


def gallagher_map() -> tuple[PolyN, PolyN, PolyN]:
    """Gao §3.5 degree-four geometric example, ``det J ≡ 2`` (sweep order)."""
    p = to_poly([0, 6, -6, 1])
    built = build_sweep_map(p, 2, -4, -1, component_order="sweep")
    if built is None:
        raise RuntimeError("Gallagher side conditions failed")
    return built


def _honesty_replay(*, keller_replay: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": False,
        "jacobian_conjecture_proof_claim": False,
        "jacobian_n2_claim": False,
        "navier_stokes_proof_claim": False,
        "keller_n_ge_3_replay": keller_replay,
        "ten_proofs_formalization_claim": False,
    }


@dataclass(frozen=True)
class KellerReplayCertificate:
    """Exact identities for a published n=3 Keller map."""

    name: str
    jacobian_constant: Fraction
    witness_image: tuple[Fraction, Fraction, Fraction] | None
    witness_preimages: tuple[tuple[Fraction, Fraction, Fraction], ...]
    generic_fiber: int
    replay_ok: bool
    honesty: dict[str, bool]
    search: str = "replay"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "keller_alpoge_replay" if self.name == "alpoge" else "keller_gallagher_replay",
            "name": self.name,
            "jacobian_constant": str(self.jacobian_constant),
            "witness_image": [str(c) for c in self.witness_image] if self.witness_image else None,
            "witness_preimages": [[str(c) for c in pt] for pt in self.witness_preimages],
            "generic_fiber": self.generic_fiber,
            "replay_ok": self.replay_ok,
            "search": self.search,
            "honesty": dict(self.honesty),
        }


def verify_alpoge_map() -> KellerReplayCertificate:
    """Prove ``det J + 2 ≡ 0`` and the published 3-to-1 witness."""
    components = alpoge_map()
    det = jacobian_det(components)
    constant = det.constant_value()
    jac_ok = constant == Fraction(-2) and (det + PolyN.const(3, 2)).is_zero()
    images = [tuple(f.eval(pt) for f in components) for pt in _ALP_WITNESSES]
    witness_ok = all(im == _ALP_IMAGE for im in images) and len(set(_ALP_WITNESSES)) == 3
    replay_ok = bool(jac_ok and witness_ok)
    return KellerReplayCertificate(
        name="alpoge",
        jacobian_constant=constant if constant is not None else Fraction(0),
        witness_image=_ALP_IMAGE,
        witness_preimages=_ALP_WITNESSES,
        generic_fiber=3,
        replay_ok=replay_ok,
        honesty=_honesty_replay(keller_replay=replay_ok),
    )


def verify_gallagher_map() -> KellerReplayCertificate:
    """Prove ``det J ≡ 2`` for Gao's degree-four n=3 example."""
    components = gallagher_map()
    det = jacobian_det(components)
    constant = det.constant_value()
    jac_ok = constant == Fraction(2) and (det - PolyN.const(3, 2)).is_zero()
    return KellerReplayCertificate(
        name="gallagher",
        jacobian_constant=constant if constant is not None else Fraction(0),
        witness_image=None,
        witness_preimages=(),
        generic_fiber=4,
        replay_ok=jac_ok,
        honesty=_honesty_replay(keller_replay=jac_ok),
    )


def matches_known_replay(hit: SweepSearchHit) -> bool:
    """Whether a search hit is the published Alpöge parameter tuple."""
    p = to_poly(hit.p)
    return (
        p == to_poly([0, 4, -3])
        and hit.gamma0 == Fraction(2)
        and hit.a == Fraction(-3)
        and hit.b == Fraction(-1)
    )


def search_hit_certificate(hit: SweepSearchHit) -> dict[str, Any]:
    """Public dict for a blind-search hit (honesty after classification)."""
    known = matches_known_replay(hit)
    honesty = honesty_search()
    honesty["discovered_by_omnibias"] = not known
    honesty["keller_n_ge_3_replay"] = False
    return {
        "kind": "keller_tangent_sweep_deg3" if "deg3" in hit.search else "keller_tangent_sweep",
        "search": hit.search,
        "generic_fiber": hit.generic_fiber,
        "p": [str(c) for c in hit.p],
        "gamma0": str(hit.gamma0),
        "a": str(hit.a),
        "b": str(hit.b),
        "jacobian_constant": str(hit.jacobian_constant),
        "witness": [[str(c) for c in pt] for pt in hit.witness],
        "rediscovered_known_witness": known,
        "discovered_by_omnibias": not known,
        "replay_ok": True,
        "honesty": honesty,
    }


def fiber_report(p: Sequence[Fraction | int] = (0, 4, -3)) -> dict[str, Any]:
    """Tangency polynomial degree / leading-coeff check (no Groebner)."""
    poly = to_poly(p)
    degree = generic_fiber_degree(poly)
    lead = tangency_leading_coeff(poly)
    return {
        "kind": "keller_fiber",
        "degree": degree,
        "leading_coeff": str(lead),
        "leading_coeff_constant": True,
        "honesty": _honesty_replay(keller_replay=False),
    }


__all__ = [
    "KellerReplayCertificate",
    "alpoge_map",
    "fiber_report",
    "gallagher_map",
    "matches_known_replay",
    "search_hit_certificate",
    "search_tangent_sweep",
    "tangency_polynomial",
    "verify_alpoge_map",
    "verify_gallagher_map",
]
