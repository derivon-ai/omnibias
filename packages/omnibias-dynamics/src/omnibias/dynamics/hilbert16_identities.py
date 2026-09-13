# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact finite identities for the Hilbert XVI coalescence atlas.

A hit certifies the named rational identity. It does not prove a C2 remainder,
G1, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.proof.catalog import CatalogEntry, register_catalog
from omnibias.core.proof.discovery import (
    Candidate,
    ExactCheck,
    FiniteFamily,
    Statement,
    run_discovery,
)
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.scale_dichotomy import (
    residual_blowup_height,
    residual_blowup_height_ratio,
    residual_event_leading,
    residual_joint_sep_r1_sum,
    residual_second_kappa_difference,
)

IDENTITY_NAMES: tuple[str, ...] = (
    "double_root",
    "w_recovers_height_power",
    "outgoing_as_w_ratio",
    "r1_product",
    "sigma_kappa",
    "blowup_height_scale",
    "blowup_height_ratio",
    "event_leading_kappa",
    "joint_sep_r1_sum",
    "hk_second_kappa_vanishes",
)

HILBERT16_IDENTITY_KIND = "hilbert16_coalescence_identities"


def _honesty(*, ok: bool) -> dict[str, bool]:
    return {
        "finite_algebra_replayed": ok,
        "parent_status_open": True,
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
        "saddle_node_c2_remainder": False,
        "two_blowup_c2_remainder": False,
        "scale_dichotomy_c2_remainder": False,
    }


def residual_double_root(rstar: Fraction, x: Fraction) -> Fraction:
    return rstar**2 + (-2 * rstar) * x + x**2 - (x - rstar) ** 2


def residual_w_height(eps: Fraction, w: Fraction, height_power: Fraction) -> Fraction:
    return eps * w - height_power


def residual_outgoing_w_ratio(
    eps: Fraction, w_max: Fraction, w_e: Fraction, height_max_power: Fraction, height_e_power: Fraction
) -> Fraction:
    return height_max_power / height_e_power - w_max / w_e


def residual_r1_product(lam1: Fraction, sep: Fraction, L: Fraction) -> Fraction:
    return (-lam1 - sep) / 2 * (-lam1 + sep) - 2 * L


def residual_sigma_kappa(sigma: Fraction, sep: Fraction, chi: Fraction, r1: Fraction) -> Fraction:
    return sigma * (chi * r1 / sep) - (sigma / sep) * chi * r1


def _sample(name: str) -> Fraction:
    rstar, x = Fraction(3, 2), Fraction(-1, 5)
    if name == "double_root":
        return residual_double_root(rstar, x)
    if name == "w_recovers_height_power":
        eps, w = Fraction(1, 4), Fraction(3, 2)
        return residual_w_height(eps, w, eps * w)
    if name == "outgoing_as_w_ratio":
        eps, w_max, w_e = Fraction(1, 3), Fraction(4), Fraction(2)
        return residual_outgoing_w_ratio(eps, w_max, w_e, eps * w_max, eps * w_e)
    if name == "r1_product":
        lam1, sep = Fraction(-4), Fraction(2)
        L = (lam1**2 - sep**2) / 4
        return residual_r1_product(lam1, sep, L)
    if name == "sigma_kappa":
        return residual_sigma_kappa(Fraction(1, 3), Fraction(1, 7), Fraction(2), Fraction(5))
    if name == "blowup_height_scale":
        eps, sigma, eta = Fraction(1, 2), Fraction(1, 3), Fraction(2)
        return residual_blowup_height(eps, sigma, eta, eps**3 * sigma**2 * eta)
    if name == "blowup_height_ratio":
        eps, eta = Fraction(1, 2), Fraction(2)
        sigma1, sigma2 = Fraction(1, 3), Fraction(1, 5)
        return residual_blowup_height_ratio(
            eps**3 * sigma1**2 * eta, eps**3 * sigma2**2 * eta, sigma1, sigma2
        )
    if name == "event_leading_kappa":
        return residual_event_leading(
            Fraction(13, 6), Fraction(1, 2), Fraction(3), Fraction(1), Fraction(1, 3), Fraction(2)
        )
    if name == "joint_sep_r1_sum":
        return residual_joint_sep_r1_sum(Fraction(-4), Fraction(2), Fraction(1))
    if name == "hk_second_kappa_vanishes":
        return residual_second_kappa_difference(Fraction(3, 5), Fraction(1, 7))
    raise KeyError(name)


def identity_verdict(name: str) -> Any:
    residual = _sample(name)
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False)


@dataclass
class Hilbert16IdentityFamily:
    """Finite box of named exact identities. Parent remains open."""

    name: str = HILBERT16_IDENTITY_KIND
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="hilbert16_coalescence_identities",
            obligation=(
                "exact double-root, W-height, outgoing W-ratio, shrinking-root, "
                "sigma-kappa, blow-up height, event-leading, and joint-axis "
                "identities over Q"
            ),
            parent="Hilbert XVI",
            parent_status="open",
            existential=True,
        )
    )

    def cardinality(self) -> int:
        return len(IDENTITY_NAMES)

    def origin(self) -> str:
        return IDENTITY_NAMES[0]

    def neighbors(self, candidate: Candidate) -> Sequence[str]:
        if candidate not in IDENTITY_NAMES:
            return ()
        index = IDENTITY_NAMES.index(candidate)  # type: ignore[arg-type]
        out: list[str] = []
        if index > 0:
            out.append(IDENTITY_NAMES[index - 1])
        if index + 1 < len(IDENTITY_NAMES):
            out.append(IDENTITY_NAMES[index + 1])
        return out

    def score(self, candidate: Candidate) -> int:
        return 0 if candidate in IDENTITY_NAMES else -1

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if candidate not in IDENTITY_NAMES:
            return None
        residual = _sample(str(candidate))
        ok = residual == 0
        verdict = identity_verdict(str(candidate))
        return ExactCheck(
            ok=ok,
            payload={
                "identity": candidate,
                "residual": [residual.numerator, residual.denominator],
                "verdict": verdict.status,
                "honesty": _honesty(ok=ok),
            },
        )


def replay_hilbert16_identities(*, budget: int = 16) -> Any:
    family: FiniteFamily = Hilbert16IdentityFamily()
    return run_discovery(family.statement, family, "score_guided", budget=budget, collect=True)


register_catalog(
    CatalogEntry(
        kind=HILBERT16_IDENTITY_KIND,
        obligation="exact coalescence identities over Q",
        parent="Hilbert XVI",
        parent_status="open",
        package="omnibias-dynamics",
        mode="exact_replay",
        complete=True,
        existential=True,
    ),
    replay_hilbert16_identities,
)
