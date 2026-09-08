# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Inverse Keller CSP / soft-opt track for Jacobian ``n=2``.

Forward polarity (``jacobian_n2``) enumerates maps and checks the Keller
gate. This module goes the **opposite** direction:

1. Soft-optimize coefficient vectors so ``det JF`` is nearly a nonzero
   constant (Keller residual).
2. Soft-score non-injectivity on a finite rational grid (collision
   residual).
3. Rationalize proposals and **exact**-certify with
   ``identical_jacobian_constant`` + grid collision / Gabber.
4. Escalate only through ``escalate_n2_result`` on a genuine violator.

Honesty
    Soft losses are probes. They do not set ``jacobian_n2_claim`` or
    ``jacobian_conjecture_proof_claim``. The proof claim stays unearned
    until a sealed parent-proof gate exists
    (:data:`JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED`). A miss is not the
    parent; an elegant leftover closure might earn a proof later.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.proof.discovery import DiscoveryResult, ExactCheck
from omnibias.holonomic._core.poly_n import (
    PolyN,
    eval_map,
    identical_jacobian_constant,
    jacobian_det,
)
from omnibias.holonomic.jacobian_n2 import (
    JACOBIAN_N2_PARENT,
    default_collision_grid,
    escalate_n2_result,
    jacobian_n2_box_statement,
    jacobian_n2_honesty,
    n2_counterexample_earned,
    n2_violation_payload,
    plane_monomials,
    rational_grid_collision,
    reject_jacobian_proof_claim,
    seal_jacobian_honesty,
)
from omnibias.holonomic.jacobian_n2_inverse import gabber_n2_test

Rational = Fraction | int


def n_plane_coeffs(max_degree: int) -> int:
    """Number of float / rational coefficients for a degree-``d`` plane map."""

    return 2 * len(plane_monomials(max_degree))


def map_from_coeffs(
    coeffs: Sequence[Rational],
    *,
    max_degree: int,
) -> tuple[PolyN, PolyN]:
    """Build ``(P, Q)`` from concatenated monomial coefficients."""

    mons = plane_monomials(max_degree)
    need = 2 * len(mons)
    if len(coeffs) != need:
        raise ValueError(
            f"expected {need} coefficients for degree {max_degree}, got {len(coeffs)}"
        )
    terms_p: dict[tuple[int, ...], Fraction] = {}
    terms_q: dict[tuple[int, ...], Fraction] = {}
    for idx, (i, j) in enumerate(mons):
        cp = Fraction(coeffs[idx])
        cq = Fraction(coeffs[len(mons) + idx])
        if cp:
            terms_p[(i, j)] = cp
        if cq:
            terms_q[(i, j)] = cq
    return PolyN(2, terms_p), PolyN(2, terms_q)


def coeffs_from_map(
    components: Sequence[PolyN],
    *,
    max_degree: int,
) -> tuple[Fraction, ...]:
    """Inverse of :func:`map_from_coeffs` (missing monomials → 0)."""

    if len(components) != 2:
        raise ValueError("coeffs_from_map expects a plane map")
    mons = plane_monomials(max_degree)
    out: list[Fraction] = []
    for component in components:
        for i, j in mons:
            out.append(component.terms.get((i, j), Fraction(0)))
    return tuple(out)


def keller_residual_exact(components: Sequence[PolyN]) -> Fraction:
    """Sum of squares of non-constant coefficients of ``det JF``.

    Zero iff ``det JF`` is an identical constant (possibly zero).
    """

    det = jacobian_det(components)
    total = Fraction(0)
    for mon, coeff in det.terms.items():
        if any(e != 0 for e in mon):
            total += coeff * coeff
    return total


def keller_target_residual_exact(
    components: Sequence[PolyN],
    target: Rational = 1,
) -> Fraction:
    """``||det JF - target||^2`` over monomial coefficients (exact)."""

    det = jacobian_det(components)
    target_f = Fraction(target)
    const = Fraction(0)
    total = Fraction(0)
    for mon, coeff in det.terms.items():
        if all(e == 0 for e in mon):
            const = coeff
        else:
            total += coeff * coeff
    total += (const - target_f) * (const - target_f)
    return total


def soft_keller_loss(
    coeffs: Sequence[float],
    *,
    max_degree: int,
    target: float = 1.0,
) -> float:
    """Float proxy of :func:`keller_target_residual_exact` (probe only)."""

    rat = tuple(Fraction(c).limit_denominator(10_000_000) for c in coeffs)
    components = map_from_coeffs(rat, max_degree=max_degree)
    return float(
        keller_target_residual_exact(components, Fraction(target).limit_denominator())
    )


def soft_collision_loss(
    coeffs: Sequence[float],
    *,
    max_degree: int,
    grid: Sequence[Fraction] | None = None,
    eps: float = 1e-9,
) -> float:
    """Soft non-injectivity: ``sum_{p≠q} 1 / (||F(p)-F(q)||^2 + eps)``.

    Large when some pair nearly collides. Probe only.
    """

    axis = tuple(grid) if grid is not None else default_collision_grid()
    rat = tuple(Fraction(c).limit_denominator(10_000_000) for c in coeffs)
    components = map_from_coeffs(rat, max_degree=max_degree)
    points = [(x, y) for x in axis for y in axis]
    images = [eval_map(components, pt) for pt in points]
    score = 0.0
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            dx = float(images[i][0] - images[j][0])
            dy = float(images[i][1] - images[j][1])
            score += 1.0 / (dx * dx + dy * dy + eps)
    return score


def inverse_joint_loss(
    coeffs: Sequence[float],
    *,
    max_degree: int,
    target: float = 1.0,
    collision_weight: float = 1e-3,
    grid: Sequence[Fraction] | None = None,
) -> dict[str, float]:
    """Minimize Keller residual; lightly reward collisions."""

    keller = soft_keller_loss(coeffs, max_degree=max_degree, target=target)
    coll = soft_collision_loss(coeffs, max_degree=max_degree, grid=grid)
    joint = keller - collision_weight * math.log1p(coll)
    return {"keller": keller, "collision": coll, "joint": joint}


@dataclass(frozen=True)
class SoftProposal:
    """One soft-opt proposal before exact certification."""

    coeffs_float: tuple[float, ...]
    max_degree: int
    target: float
    losses: Mapping[str, float]
    steps: int
    seed: int


def anneal_keller_propose(
    *,
    max_degree: int = 2,
    target: float = 1.0,
    steps: int = 400,
    step_size: float = 0.15,
    seed: int = 0,
    collision_weight: float = 1e-3,
    init: Sequence[float] | None = None,
    grid: Sequence[Fraction] | None = None,
) -> SoftProposal:
    """Random-coordinate descent on the joint soft loss (pure Python).

    Starts near the identity ``(x, y)`` when ``init`` is omitted.
    """

    rng = random.Random(seed)
    dim = n_plane_coeffs(max_degree)
    mons = plane_monomials(max_degree)
    if init is not None:
        if len(init) != dim:
            raise ValueError(f"init needs {dim} entries, got {len(init)}")
        coeffs = [float(c) for c in init]
    else:
        coeffs = [0.0] * dim
        ix = mons.index((1, 0))
        iy = mons.index((0, 1))
        coeffs[ix] = 1.0
        coeffs[len(mons) + iy] = 1.0

    best = list(coeffs)
    best_loss = inverse_joint_loss(
        best,
        max_degree=max_degree,
        target=target,
        collision_weight=collision_weight,
        grid=grid,
    )["joint"]
    cur = list(coeffs)
    cur_loss = best_loss
    for t in range(steps):
        j = rng.randrange(dim)
        delta = rng.uniform(-step_size, step_size)
        delta *= max(0.05, 1.0 - t / max(steps, 1))
        trial = list(cur)
        trial[j] += delta
        losses = inverse_joint_loss(
            trial,
            max_degree=max_degree,
            target=target,
            collision_weight=collision_weight,
            grid=grid,
        )
        temperature = max(1e-4, 0.2 * (1.0 - t / max(steps, 1)))
        accept = losses["joint"] <= cur_loss or rng.random() < math.exp(
            (cur_loss - losses["joint"]) / temperature
        )
        if accept:
            cur = trial
            cur_loss = losses["joint"]
            if cur_loss < best_loss:
                best = list(cur)
                best_loss = cur_loss
    final_losses = inverse_joint_loss(
        best,
        max_degree=max_degree,
        target=target,
        collision_weight=collision_weight,
        grid=grid,
    )
    return SoftProposal(
        coeffs_float=tuple(best),
        max_degree=max_degree,
        target=target,
        losses=final_losses,
        steps=steps,
        seed=seed,
    )


def rationalize_coeffs(
    coeffs: Sequence[float],
    *,
    height: int = 8,
    denom_cap: int = 64,
) -> tuple[Fraction, ...]:
    """Nearest rational coeffs with bounded height / denominator."""

    if height < 0:
        raise ValueError(f"height must be >= 0, got {height}")
    out: list[Fraction] = []
    for c in coeffs:
        r = Fraction(c).limit_denominator(max(denom_cap, 1))
        num, den = r.numerator, r.denominator
        if abs(num) > height * den:
            sign = 1 if num >= 0 else -1
            r = Fraction(sign * height * den, den)
        if den > denom_cap:
            r = Fraction(max(-height, min(height, round(float(c)))))
        out.append(r)
    return tuple(out)


@dataclass(frozen=True)
class ExactCertifyResult:
    """Exact post-rationalization certificate attempt."""

    coeffs: tuple[Fraction, ...]
    max_degree: int
    keller_residual: Fraction
    jacobian_constant: Fraction | None
    jacobian_nonzero_constant: bool
    collision: bool
    gabber_fails: bool
    violator: bool
    payload: Mapping[str, Any]
    honesty: Mapping[str, bool]


def certify_rational_map(
    coeffs: Sequence[Rational],
    *,
    max_degree: int,
    grid: Sequence[Fraction] | None = None,
) -> ExactCertifyResult:
    """Exact Keller + collision / Gabber gate on a rational coefficient map."""

    axis = tuple(grid) if grid is not None else default_collision_grid()
    components = map_from_coeffs(coeffs, max_degree=max_degree)
    residual = keller_residual_exact(components)
    constant = identical_jacobian_constant(components)
    nonzero = constant is not None and constant != 0
    hit = rational_grid_collision(components, axis) if nonzero else None
    collision = hit is not None
    gabber = gabber_n2_test(components) if nonzero else None
    gabber_fails = bool(nonzero and gabber is not None and gabber.fails)
    if residual == 0 and nonzero:
        payload: dict[str, Any] = n2_violation_payload(
            components,
            axis=axis,
            candidate=[str(c) for c in coeffs],
        )
        violator = n2_counterexample_earned(payload)
        honesty = seal_jacobian_honesty(payload["honesty"])
        payload = {**payload, "honesty": dict(honesty)}
    else:
        violator = False
        honesty = jacobian_n2_honesty(discovered=False, n2_counterexample=False)
        payload = {
            "jacobian_identity": "identical" if residual == 0 else "nonconstant",
            "jacobian_constant": "" if constant is None else str(constant),
            "jacobian_nonzero_constant": bool(nonzero),
            "grid_collision": bool(collision),
            "gabber_fails": bool(gabber_fails),
            "keller_residual": str(residual),
            "parent": JACOBIAN_N2_PARENT,
            "parent_status": "open",
            "honesty": dict(honesty),
        }
    reject_jacobian_proof_claim(honesty)
    return ExactCertifyResult(
        coeffs=tuple(Fraction(c) for c in coeffs),
        max_degree=max_degree,
        keller_residual=residual,
        jacobian_constant=constant,
        jacobian_nonzero_constant=bool(nonzero),
        collision=bool(collision),
        gabber_fails=bool(gabber_fails),
        violator=bool(violator),
        payload=payload,
        honesty=dict(honesty),
    )


def propose_and_certify(
    *,
    max_degree: int = 2,
    target: float = 1.0,
    steps: int = 400,
    seed: int = 0,
    height: int = 8,
    collision_weight: float = 1e-3,
    init: Sequence[float] | None = None,
    grid: Sequence[Fraction] | None = None,
) -> dict[str, Any]:
    """One inverse-track tick: soft propose → rationalize → exact certify."""

    proposal = anneal_keller_propose(
        max_degree=max_degree,
        target=target,
        steps=steps,
        seed=seed,
        collision_weight=collision_weight,
        init=init,
        grid=grid,
    )
    rational = rationalize_coeffs(proposal.coeffs_float, height=height)
    certified = certify_rational_map(rational, max_degree=max_degree, grid=grid)
    sealed: dict[str, Any] | None = None
    if certified.violator:
        statement = jacobian_n2_box_statement(
            max_degree=max_degree,
            coeff_height=height,
            grid=grid,
        )
        fake = DiscoveryResult(
            status="DISPROVED",
            statement=statement,
            family=f"inverse_csp_d{max_degree}",
            proposer="anneal_keller_propose",
            budget=1,
            evaluated=1,
            candidate=None,
            check=ExactCheck(ok=True, payload=dict(certified.payload)),
            detail="inverse CSP exact violator",
            search_incomplete=True,
        )
        sealed = escalate_n2_result(fake)
    return {
        "track": "jacobian_n2_inverse_csp",
        "parent": JACOBIAN_N2_PARENT,
        "parent_status": "open" if not certified.violator else "false",
        "proposal": {
            "seed": proposal.seed,
            "steps": proposal.steps,
            "max_degree": proposal.max_degree,
            "target": proposal.target,
            "losses": dict(proposal.losses),
            "coeffs_float": list(proposal.coeffs_float),
        },
        "rational_coeffs": [str(c) for c in rational],
        "certified": {
            "keller_residual": str(certified.keller_residual),
            "jacobian_constant": (
                None
                if certified.jacobian_constant is None
                else str(certified.jacobian_constant)
            ),
            "jacobian_nonzero_constant": certified.jacobian_nonzero_constant,
            "collision": certified.collision,
            "gabber_fails": certified.gabber_fails,
            "violator": certified.violator,
            "honesty": dict(certified.honesty),
        },
        "escalate": sealed,
        "escalate_parent": bool(sealed and sealed.get("escalate_parent")),
        "note": (
            "soft propose is a probe; exact gate only; parent stays open unless violator"
            if not certified.violator
            else "exact n=2 violator earned via inverse CSP certify"
        ),
    }


__all__ = [
    "ExactCertifyResult",
    "SoftProposal",
    "anneal_keller_propose",
    "certify_rational_map",
    "coeffs_from_map",
    "inverse_joint_loss",
    "keller_residual_exact",
    "keller_target_residual_exact",
    "map_from_coeffs",
    "n_plane_coeffs",
    "propose_and_certify",
    "rationalize_coeffs",
    "soft_collision_loss",
    "soft_keller_loss",
]
