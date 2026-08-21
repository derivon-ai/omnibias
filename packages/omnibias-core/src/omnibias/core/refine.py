# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Adaptive pack refinement algebra (theory 03-13).

Three structural moves — birth, growth, and death — act on a bank of
tempered packs. Birth and growth are **exactly** zero-perturbation
(new outer weight ``c = 0``, or a higher-order sibling with ``c = 0``).
Death is bounded-perturbation: the API reports
``||c_g p_g|| / ||u||`` and never pretends the change is zero.

Scale ``alpha`` is the founding scaling law
``sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)``. That is a third
axis, not bias collapse (``delta -> 0``) and not temperature collapse
(``beta -> inf`` via :func:`omnibias.core.spec.tempered`).

Singularity and scale-flow indicators use the Domb-Sykes / ratio
estimators that spec 03-10 and 03-07 need. They are not the full Padé
tracker or the RG beta-function; those parent specs stay designed.

This module is backend-free. Tensor evaluation lives in the torch / jax
twins. ``GrowableOperatorMultiBiasUnit.grow`` remains the literal-``K``
growth move; pack-bank *p*-type growth is the collapsed sibling
(add ``sigma^(n+1)`` at the same centre with ``c = 0``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from omnibias.core.line_search import (
    lagrange_remainder_bound,
    taylor_coeffs_from_derivatives,
)
from omnibias.core.multipack import PackSpec

HpMove = Literal["h", "p", "none"]
HpRule = Literal["coefficient_decay", "always_h", "always_p"]


class Indicator(str, Enum):
    """Where-to-refine signal. Scale comes only from jet-aware members."""

    RESIDUAL = "residual"
    CERTIFIED_ERROR = "certified_error"
    SINGULARITY = "singularity"
    SCALE_FLOW = "scale_flow"


@dataclass(frozen=True)
class RefinedPack:
    """One tempered pack: ``c * alpha^n sigma^(n)(alpha (z - center))``.

    ``center`` is the physical location of the pack (spec 03-13 ``mu``).
    This is *not* :class:`~omnibias.core.multipack.PackSpec.mean`, which
    is added (``z + mean``). Convert with :meth:`from_pack_spec` /
    :meth:`to_pack_spec`.
    """

    order: int
    center: float
    weight: float = 0.0
    scale: float = 1.0
    age: int = 0

    def __post_init__(self) -> None:
        if int(self.order) < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if not math.isfinite(self.center):
            raise ValueError(f"center must be finite, got {self.center}")
        if not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite, got {self.weight}")
        if not math.isfinite(self.scale) or self.scale == 0.0:
            raise ValueError(f"scale must be finite and nonzero, got {self.scale}")
        if int(self.age) < 0:
            raise ValueError(f"age must be >= 0, got {self.age}")
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "center", float(self.center))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "age", int(self.age))

    @classmethod
    def from_pack_spec(cls, spec: PackSpec, *, scale: float = 1.0, age: int = 0) -> RefinedPack:
        """Map a Birkhoff :class:`PackSpec` (``z + mean``) onto a centred pack."""
        return cls(
            order=spec.order,
            center=-float(spec.mean),
            weight=float(spec.weight),
            scale=float(scale),
            age=int(age),
        )

    def to_pack_spec(self) -> PackSpec:
        """Inverse of :meth:`from_pack_spec` (drops ``scale`` and ``age``)."""
        return PackSpec(order=self.order, mean=-self.center, weight=self.weight)

    def with_weight(self, weight: float) -> RefinedPack:
        return RefinedPack(
            order=self.order,
            center=self.center,
            weight=float(weight),
            scale=self.scale,
            age=self.age,
        )

    def aged(self, steps: int = 1) -> RefinedPack:
        return RefinedPack(
            order=self.order,
            center=self.center,
            weight=self.weight,
            scale=self.scale,
            age=self.age + int(steps),
        )


@dataclass(frozen=True)
class RefinePolicy:
    """Discrete refinement policy. Decisions are not a smooth optimiser."""

    indicator: Indicator = Indicator.RESIDUAL
    birth_threshold: float = 0.1
    death_threshold: float = 0.01
    min_age: int = 100
    max_packs: int | None = None
    hp_rule: HpRule = "coefficient_decay"
    hysteresis: float = 2.0
    jet_order: int = 6
    default_order: int = 0
    min_center_separation: float = 0.0
    min_scale_ratio: float = 1.5
    debug_zero_perturbation: bool = False

    def __post_init__(self) -> None:
        if self.birth_threshold < 0.0:
            raise ValueError(f"birth_threshold must be >= 0, got {self.birth_threshold}")
        if self.death_threshold < 0.0:
            raise ValueError(f"death_threshold must be >= 0, got {self.death_threshold}")
        if self.min_age < 0:
            raise ValueError(f"min_age must be >= 0, got {self.min_age}")
        if self.max_packs is not None and int(self.max_packs) < 1:
            raise ValueError(f"max_packs must be >= 1 or None, got {self.max_packs}")
        if self.hysteresis < 1.0:
            raise ValueError(f"hysteresis must be >= 1, got {self.hysteresis}")
        if self.jet_order < 1:
            raise ValueError(f"jet_order must be >= 1, got {self.jet_order}")
        if self.default_order < 0:
            raise ValueError(f"default_order must be >= 0, got {self.default_order}")
        if self.min_center_separation < 0.0:
            raise ValueError(
                f"min_center_separation must be >= 0, got {self.min_center_separation}"
            )
        if self.min_scale_ratio < 1.0:
            raise ValueError(f"min_scale_ratio must be >= 1, got {self.min_scale_ratio}")
        if self.hp_rule not in ("coefficient_decay", "always_h", "always_p"):
            raise ValueError(f"unknown hp_rule {self.hp_rule!r}")
        object.__setattr__(self, "indicator", Indicator(self.indicator))


@dataclass(frozen=True)
class RefineProposal:
    """A single structural suggestion before it is applied."""

    move: HpMove
    center: float
    scale: float
    order: int
    parent_index: int | None = None
    indicator_peak: float = 0.0
    inherit_scale: bool = False


@dataclass(frozen=True)
class RefineReport:
    """What changed, plus the death bound that the caller must see."""

    born: tuple[RefinedPack, ...]
    grown: tuple[int, ...]
    died: tuple[int, ...]
    death_perturbation: float
    indicator_values: tuple[float, ...]
    proposed_center: float | None = None
    proposed_scale: float | None = None
    hp_move: HpMove = "none"


def pack_term_scalar(
    z: float,
    pack: RefinedPack,
    sigma_n: Callable[[float, int], float],
) -> float:
    """Scalar ``c * alpha^n sigma^(n)(alpha (z - center))``."""
    u = pack.scale * (float(z) - pack.center)
    return pack.weight * (pack.scale ** pack.order) * float(sigma_n(u, pack.order))


def successive_ratios(derivatives: Sequence[float]) -> list[float]:
    """Taylor-coefficient ratios ``a_k / a_{k-1}`` for ``k >= 1``."""
    coeffs = taylor_coeffs_from_derivatives(derivatives)
    if len(coeffs) < 2:
        raise ValueError("need at least two derivatives for a ratio")
    out: list[float] = []
    for k in range(1, len(coeffs)):
        prev = coeffs[k - 1]
        if prev == 0.0 or not math.isfinite(prev):
            out.append(float("nan"))
        else:
            out.append(coeffs[k] / prev)
    return out


def local_scale_from_derivatives(derivatives: Sequence[float]) -> float:
    r"""Signed generation scale from early jet ratios.

    For ``exp(alpha (z - z0))``, ``a_k / a_{k-1} = alpha / k``, so
    ``k (a_k / a_{k-1})`` is constantly ``alpha``. The estimator is the
    median of those products over finite ratios. The sign is the sign of
    ``a_1 / a_0`` (the first directional scale).
    """
    ratios = successive_ratios(derivatives)
    products: list[float] = []
    for k, ratio in enumerate(ratios, start=1):
        if math.isfinite(ratio):
            products.append(float(k) * ratio)
    if not products:
        raise ValueError("no finite successive ratio; cannot estimate scale")
    products.sort()
    mid = len(products) // 2
    if len(products) % 2 == 1:
        return products[mid]
    return 0.5 * (products[mid - 1] + products[mid])


def domb_sykes_fit(derivatives: Sequence[float]) -> tuple[float, float]:
    r"""Linear fit of ``a_k / a_{k-1}`` against ``1/k``.

    Returns ``(intercept, slope)``. For a singularity
    ``(1 - x/x_s)^{-p}`` the intercept is ``1/x_s``. An entire function
    (the worked ``exp(-100 x)`` example) has intercept ``0``.
    """
    ratios = successive_ratios(derivatives)
    xs: list[float] = []
    ys: list[float] = []
    for k, ratio in enumerate(ratios, start=1):
        if math.isfinite(ratio):
            xs.append(1.0 / float(k))
            ys.append(ratio)
    n = len(xs)
    if n < 2:
        raise ValueError("Domb-Sykes needs at least two finite ratios")
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denom = float(n) * sum_xx - sum_x * sum_x
    if denom == 0.0:
        raise ValueError("Domb-Sykes abscissae are degenerate")
    slope = (float(n) * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / float(n)
    return intercept, slope


def hp_decision(
    derivatives: Sequence[float],
    existing_scale: float,
    *,
    rule: HpRule = "coefficient_decay",
    scale_mismatch: float = 2.0,
    entire_intercept: float = 0.05,
) -> HpMove:
    """Choose *h* (new scale) or *p* (raise order) from jet decay.

    Classical *hp*: *p* where the solution is locally analytic at the
    current scale, *h* where the scale is wrong or a singularity is
    nearby. The worked boundary-layer jet is entire (intercept ~ 0) but
    the early ratios demand ``|alpha| ~ 100``, so the first move is *h*.
    """
    if rule == "always_h":
        return "h"
    if rule == "always_p":
        return "p"
    if existing_scale == 0.0 or not math.isfinite(existing_scale):
        raise ValueError(f"existing_scale must be finite and nonzero, got {existing_scale}")
    scale = local_scale_from_derivatives(derivatives)
    if abs(scale) / abs(existing_scale) >= float(scale_mismatch) or abs(existing_scale) / max(
        abs(scale), 1e-300
    ) >= float(scale_mismatch):
        return "h"
    intercept, _slope = domb_sykes_fit(derivatives)
    if math.isfinite(intercept) and abs(intercept) <= float(entire_intercept):
        return "p"
    return "h"


def certified_local_error(
    derivatives: Sequence[float],
    step: float,
) -> float:
    """Lagrange remainder of the jet using ``|phi^(N)|`` as ``M`` for order ``N-1``.

    Sound when ``|phi^(N)|`` bounds the next derivative on the segment of
    length ``|step|``. Uses the same identity as
    :func:`omnibias.core.line_search.lagrange_remainder_bound`.
    """
    if len(derivatives) < 2:
        raise ValueError("certified_local_error needs at least two derivatives")
    order = len(derivatives) - 2
    m = abs(float(derivatives[-1]))
    enclosure = lagrange_remainder_bound(m, float(step), order)
    return float(enclosure.hi)


def nearest_pack_index(
    packs: Sequence[RefinedPack],
    center: float,
) -> int | None:
    """Index of the pack whose centre is closest to ``center``."""
    if not packs:
        return None
    best = 0
    best_d = abs(packs[0].center - float(center))
    for i, pack in enumerate(packs[1:], start=1):
        d = abs(pack.center - float(center))
        if d < best_d:
            best = i
            best_d = d
    return best


def inherit_scale(packs: Sequence[RefinedPack], center: float, default: float = 1.0) -> float:
    """Scale of the nearest existing pack, or ``default`` if the bank is empty."""
    idx = nearest_pack_index(packs, center)
    if idx is None:
        return float(default)
    return packs[idx].scale


def _too_close(
    packs: Sequence[RefinedPack],
    center: float,
    scale: float,
    policy: RefinePolicy,
) -> bool:
    for pack in packs:
        if abs(pack.center - center) > policy.min_center_separation:
            continue
        ratio = abs(pack.scale) / max(abs(scale), 1e-300)
        if ratio < 1.0:
            ratio = 1.0 / ratio
        if ratio < policy.min_scale_ratio:
            return True
    return False


def propose_refinement(
    packs: Sequence[RefinedPack],
    *,
    policy: RefinePolicy,
    peak_location: float,
    peak_value: float,
    derivatives: Sequence[float] | None,
    birth_score: float,
) -> RefineProposal | None:
    """Turn an indicator peak into an *h* or *p* proposal, or ``None``.

    Residual / certified-error proposals inherit the nearest pack's
    scale (they do not carry a scale). Singularity and scale-flow read
    ``derivatives`` and propose that scale.
    """
    threshold = max(policy.birth_threshold, policy.hysteresis * policy.death_threshold)
    if birth_score < threshold:
        return None
    if policy.max_packs is not None and len(packs) >= int(policy.max_packs):
        return None

    inherit = policy.indicator in (Indicator.RESIDUAL, Indicator.CERTIFIED_ERROR)
    if inherit or derivatives is None:
        scale = inherit_scale(packs, peak_location)
        move: HpMove = "h" if policy.hp_rule != "always_p" else "p"
        if policy.hp_rule == "always_p":
            move = "p"
        elif policy.hp_rule == "always_h":
            move = "h"
        else:
            move = "h"
    else:
        existing = inherit_scale(packs, peak_location)
        scale = local_scale_from_derivatives(derivatives)
        if policy.indicator is Indicator.SCALE_FLOW and policy.hp_rule == "coefficient_decay":
            # Scale flow names the cutoff to add; that is an h-move.
            move = "h"
        else:
            move = hp_decision(derivatives, existing, rule=policy.hp_rule)

    parent = nearest_pack_index(packs, peak_location)
    if move == "p":
        if parent is None:
            move = "h"
        else:
            order = packs[parent].order + 1
            return RefineProposal(
                move="p",
                center=packs[parent].center,
                scale=packs[parent].scale,
                order=order,
                parent_index=parent,
                indicator_peak=float(peak_value),
                inherit_scale=True,
            )

    if _too_close(packs, float(peak_location), float(scale), policy):
        return None
    return RefineProposal(
        move="h",
        center=float(peak_location),
        scale=float(scale),
        order=int(policy.default_order),
        parent_index=parent,
        indicator_peak=float(peak_value),
        inherit_scale=inherit,
    )


def apply_birth(packs: Sequence[RefinedPack], proposal: RefineProposal) -> tuple[list[RefinedPack], RefinedPack]:
    """Append a zero-weight pack. The represented function is unchanged."""
    born = RefinedPack(
        order=proposal.order,
        center=proposal.center,
        weight=0.0,
        scale=proposal.scale,
        age=0,
    )
    return [*packs, born], born


def apply_growth(
    packs: Sequence[RefinedPack],
    proposal: RefineProposal,
) -> tuple[list[RefinedPack], RefinedPack, int]:
    """*p*-type: sibling at the same centre / scale, order ``n+1``, ``c = 0``."""
    if proposal.parent_index is None:
        raise ValueError("p-type growth requires a parent_index")
    parent = packs[proposal.parent_index]
    grown = RefinedPack(
        order=proposal.order,
        center=parent.center,
        weight=0.0,
        scale=parent.scale,
        age=0,
    )
    return [*packs, grown], grown, proposal.parent_index


def pack_contributions(
    terms: Sequence[float],
    field_norm: float,
) -> list[float]:
    """``||c_g p_g|| / ||u||`` from already-reduced per-pack norms."""
    if field_norm < 0.0:
        raise ValueError(f"field_norm must be >= 0, got {field_norm}")
    denom = float(field_norm)
    if denom == 0.0:
        return [0.0 for _ in terms]
    return [abs(float(t)) / denom for t in terms]


def select_deaths(
    packs: Sequence[RefinedPack],
    contributions: Sequence[float],
    policy: RefinePolicy,
    *,
    keep_at_least: int = 1,
) -> list[int]:
    """Indices that may die. Newly born packs are protected by ``min_age``."""
    if len(packs) != len(contributions):
        raise ValueError("contributions length must match packs")
    if keep_at_least < 0:
        raise ValueError(f"keep_at_least must be >= 0, got {keep_at_least}")
    candidates = [
        i
        for i, (pack, contrib) in enumerate(zip(packs, contributions, strict=True))
        if pack.age >= policy.min_age and contrib <= policy.death_threshold
    ]
    budget = len(packs) - keep_at_least
    if budget <= 0:
        return []
    candidates.sort(key=lambda i: contributions[i])
    return candidates[:budget]


def apply_deaths(
    packs: Sequence[RefinedPack],
    indices: Sequence[int],
    contributions: Sequence[float],
) -> tuple[list[RefinedPack], float]:
    """Remove packs. Reported perturbation is the sum of their contributions."""
    doomed = set(int(i) for i in indices)
    perturbation = sum(float(contributions[i]) for i in doomed)
    kept = [p for i, p in enumerate(packs) if i not in doomed]
    return kept, perturbation


def increment_ages(packs: Sequence[RefinedPack], steps: int = 1) -> list[RefinedPack]:
    return [p.aged(steps) for p in packs]


def residual_indicator(values: Sequence[float]) -> tuple[int, float]:
    """Argmax of ``|value|`` and that magnitude."""
    if not values:
        raise ValueError("residual_indicator needs at least one sample")
    best = 0
    best_v = abs(float(values[0]))
    for i, raw in enumerate(values[1:], start=1):
        v = abs(float(raw))
        if v > best_v:
            best = i
            best_v = v
    return best, best_v


def assert_zero_perturbation(
    before: Sequence[float],
    after: Sequence[float],
) -> None:
    """Birth and growth must be bit-identical, not 'within tolerance'."""
    if len(before) != len(after):
        raise AssertionError(
            f"zero-perturbation length mismatch: {len(before)} vs {len(after)}"
        )
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        if a != b:
            raise AssertionError(
                f"zero-perturbation violated at index {i}: {a!r} vs {b!r}"
            )


def death_bound_holds(
    measured_relative: float,
    reported: float,
    *,
    atol: float = 0.0,
) -> bool:
    """``measured ||Δu|| / ||u||`` must not exceed the reported bound."""
    if reported < 0.0:
        raise ValueError(f"reported perturbation must be >= 0, got {reported}")
    return float(measured_relative) <= float(reported) + float(atol)


def factorial_scale_term(scale: float, order: int) -> float:
    """``alpha^n`` with an integer order (used by the twins' host path)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    return float(scale) ** int(order)


def jet_order_needed(policy: RefinePolicy) -> int:
    """Derivatives ``0 .. N`` required for the configured indicator."""
    return max(int(policy.jet_order), 2)


def refine_bank(
    packs: Sequence[RefinedPack],
    policy: RefinePolicy,
    *,
    probes: Sequence[float],
    residuals: Sequence[float],
    contributions: Sequence[float],
    derivatives: Sequence[float] | None,
    certified_values: Sequence[float] | None = None,
    peak_location: float | None = None,
) -> tuple[list[RefinedPack], RefineReport]:
    """One discrete refine step on a host-side pack list.

    Age every pack, then birth or grow from the indicator peak, then die.
    ``died`` indexes the bank *after* appends (newborns sit at the end,
    age 0, so ``min_age`` protects them). Torch / jax twins apply this
    report to pre-allocated slots so optimizer moments of live packs
    stay put.
    """
    if len(probes) != len(residuals):
        raise ValueError("probes and residuals must have the same length")
    if len(contributions) != len(packs):
        raise ValueError("contributions length must match packs")
    if not probes:
        raise ValueError("refine_bank needs at least one probe")

    aged = increment_ages(packs, 1)
    if policy.indicator is Indicator.CERTIFIED_ERROR:
        if certified_values is None:
            raise ValueError("certified_error indicator requires certified_values")
        if len(certified_values) != len(probes):
            raise ValueError("certified_values length must match probes")
        values: Sequence[float] = certified_values
    else:
        values = residuals
    peak_i, peak_v = residual_indicator(values)
    loc = float(probes[peak_i]) if peak_location is None else float(peak_location)
    proposal = propose_refinement(
        aged,
        policy=policy,
        peak_location=loc,
        peak_value=peak_v,
        derivatives=derivatives,
        birth_score=peak_v,
    )
    working = list(aged)
    born: list[RefinedPack] = []
    grown: list[int] = []
    hp: HpMove = "none"
    proposed_center: float | None = None
    proposed_scale: float | None = None
    if proposal is not None:
        proposed_center = proposal.center
        proposed_scale = proposal.scale
        hp = proposal.move
        if proposal.move == "p":
            working, _sibling, parent = apply_growth(working, proposal)
            grown.append(parent)
        else:
            working, newborn = apply_birth(working, proposal)
            born.append(newborn)

    contrib_ext = [float(c) for c in contributions]
    while len(contrib_ext) < len(working):
        contrib_ext.append(0.0)
    died = select_deaths(working, contrib_ext, policy)
    working, perturbation = apply_deaths(working, died, contrib_ext)
    report = RefineReport(
        born=tuple(born),
        grown=tuple(grown),
        died=tuple(died),
        death_perturbation=perturbation,
        indicator_values=tuple(float(v) for v in values),
        proposed_center=proposed_center,
        proposed_scale=proposed_scale,
        hp_move=hp,
    )
    return working, report


__all__ = [
    "HpMove",
    "HpRule",
    "Indicator",
    "RefinePolicy",
    "RefineProposal",
    "RefineReport",
    "RefinedPack",
    "apply_birth",
    "apply_deaths",
    "apply_growth",
    "assert_zero_perturbation",
    "certified_local_error",
    "death_bound_holds",
    "domb_sykes_fit",
    "factorial_scale_term",
    "hp_decision",
    "increment_ages",
    "inherit_scale",
    "jet_order_needed",
    "local_scale_from_derivatives",
    "nearest_pack_index",
    "pack_contributions",
    "pack_term_scalar",
    "propose_refinement",
    "refine_bank",
    "residual_indicator",
    "select_deaths",
    "successive_ratios",
]
