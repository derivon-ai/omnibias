# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite rational convergence ledgers (theory 07-08).

A staged iterative construction (a wave-packet ladder, a cluster
expansion) closes if and only if a finite list of affine margin
inequalities holds on a rational (possibly unbounded) feasible
interval. That collapse is the whole trick, and it is decidable:
each residual is affine, so ``< 0`` is decided by endpoint values
plus slope sign. No LP solver, no floats.

This is the finite spine shared by OpenAI's Navier–Stokes
``ExponentLedger`` (Clay alternatives (C) and (D)) and the
Kotecký–Preiss polymer-coordination criterion behind strong-coupling
Yang–Mills. The checker certifies the **algebra**. It does not
construct a correction, define an analytic class, or prove a PDE.

``theorem_prover_verified`` is earned only by a genuine kernel
``lake build``. This module never writes that flag into a
certificate. ``mathlib_verified`` is a distinct tier and is never
set here. Parent-level honesty flags
(``navier_stokes_proof_claim``, ``yang_mills_mass_gap_claim``) are
**derived**: they become true only when every margin discharges
**and** ``external_premises`` is empty. They cannot be stamped by
hand.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, cast

from omnibias.core.proof.catalog import CatalogEntry, register_catalog
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.obligations.rational_stencil import Obligation

PAYLOAD_LEDGER = "convergence_ledger"

#: Largest absolute numerator/denominator the Lean emitter will accept.
_INT_CAP = 10**18

Sense = Literal["le", "lt", "ge", "gt"]
ClaimStrength = Literal["BLOCKED", "CONDITIONAL", "PROVED"]
_SENSE_FLIP: dict[Sense, Sense] = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt"}

NS_PARENT = "Navier-Stokes unforced regularity (Clay A/B)"
YM_PARENT = "Yang-Mills existence and mass gap (Clay)"

PARENT_CLAIM_KEYS: frozenset[str] = frozenset(
    {
        "navier_stokes_proof_claim",
        "yang_mills_mass_gap_claim",
    }
)

NS_EXTERNAL_PREMISES: tuple[str, ...] = (
    "analytic classes of the slow base and the high-frequency packets",
    "construction of each particular / signed correction",
    "PDE residual estimates that the ledger treats as given",
)

YM_EXTERNAL_PREMISES: tuple[str, ...] = (
    "continuum / thermodynamic limit of the lattice theory",
    "Osterwalder-Schrader reconstruction",
    "spacing-uniform spectral-gap lower bound",
)


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


def _coeffs_tuple(raw: Mapping[str, Any]) -> tuple[tuple[str, Fraction], ...]:
    items: list[tuple[str, Fraction]] = []
    for name, value in raw.items():
        parsed = _pair_to_frac(value)
        if parsed is None:
            raise ValueError(f"AffineForm coeff {name!r} must be a rational")
        if parsed != 0:
            items.append((str(name), parsed))
    items.sort(key=lambda pair: pair[0])
    return tuple(items)


@dataclass(frozen=True)
class AffineForm:
    """``const + sum_k coeffs[k] * x_k`` with exact rational coefficients."""

    const: Fraction
    coeffs: tuple[tuple[str, Fraction], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "const", _as_frac(self.const))
        cleaned = tuple(
            (str(name), _as_frac(value)) for name, value in self.coeffs if value != 0
        )
        object.__setattr__(self, "coeffs", tuple(sorted(cleaned, key=lambda pair: pair[0])))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AffineForm:
        const = _pair_to_frac(raw.get("const", 0))
        if const is None:
            raise ValueError("AffineForm.const must be a rational")
        coeffs_raw = raw.get("coeffs", {})
        if not isinstance(coeffs_raw, Mapping):
            raise ValueError("AffineForm.coeffs must be a mapping")
        return cls(const, _coeffs_tuple(cast(Mapping[str, Any], coeffs_raw)))

    @classmethod
    def constant(cls, value: Fraction | int | str) -> AffineForm:
        return cls(_as_frac(value))

    @classmethod
    def variable(cls, name: str, *, scale: Fraction | int | str = 1) -> AffineForm:
        return cls(Fraction(0), ((name, _as_frac(scale)),))

    @property
    def coeff_map(self) -> dict[str, Fraction]:
        return {name: value for name, value in self.coeffs}

    def coeff(self, name: str) -> Fraction:
        return self.coeff_map.get(name, Fraction(0))

    def names(self) -> frozenset[str]:
        return frozenset(self.coeff_map)

    def eval(self, assignment: Mapping[str, Fraction]) -> Fraction:
        acc = self.const
        for name, value in self.coeffs:
            if name not in assignment:
                raise KeyError(f"AffineForm missing assignment for {name!r}")
            acc += value * assignment[name]
        return acc

    def substitute(self, assignment: Mapping[str, Fraction]) -> AffineForm:
        """Fold known variables into the constant term."""
        acc = self.const
        leftover: list[tuple[str, Fraction]] = []
        for name, value in self.coeffs:
            if name in assignment:
                acc += value * assignment[name]
            else:
                leftover.append((name, value))
        return AffineForm(acc, tuple(leftover))

    def shift(self, name: str, step: Fraction) -> AffineForm:
        """Replace ``name`` by ``name + step``."""
        return AffineForm(self.const + self.coeff(name) * step, self.coeffs)

    def __add__(self, other: AffineForm | Fraction | int) -> AffineForm:
        if not isinstance(other, AffineForm):
            return AffineForm(self.const + _as_frac(other), self.coeffs)
        merged = dict(self.coeff_map)
        for name, value in other.coeffs:
            merged[name] = merged.get(name, Fraction(0)) + value
        return AffineForm(self.const + other.const, _coeffs_tuple(merged))

    def __sub__(self, other: AffineForm | Fraction | int) -> AffineForm:
        if not isinstance(other, AffineForm):
            return AffineForm(self.const - _as_frac(other), self.coeffs)
        return self + AffineForm(-other.const, tuple((n, -c) for n, c in other.coeffs))

    def __neg__(self) -> AffineForm:
        return AffineForm(-self.const, tuple((n, -c) for n, c in self.coeffs))

    def scale(self, factor: Fraction | int | str) -> AffineForm:
        k = _as_frac(factor)
        return AffineForm(self.const * k, tuple((n, c * k) for n, c in self.coeffs))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "const": _rat_pair(self.const),
            "coeffs": {name: _rat_pair(value) for name, value in self.coeffs},
        }


@dataclass(frozen=True)
class MinForm:
    """Pointwise minimum of finitely many affine forms."""

    parts: tuple[AffineForm, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("MinForm requires at least one part")
        object.__setattr__(self, "parts", tuple(self.parts))

    @classmethod
    def singleton(cls, form: AffineForm) -> MinForm:
        return cls((form,))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MinForm:
        parts_raw = raw.get("parts", ())
        if not isinstance(parts_raw, Sequence) or isinstance(parts_raw, str | bytes):
            raise ValueError("MinForm.parts must be a sequence")
        parts = []
        for item in parts_raw:
            if not isinstance(item, Mapping):
                raise ValueError("MinForm part must be a mapping")
            parts.append(AffineForm.from_mapping(item))
        return cls(tuple(parts))

    def eval(self, assignment: Mapping[str, Fraction]) -> Fraction:
        return min(part.eval(assignment) for part in self.parts)

    def binding_index(self, assignment: Mapping[str, Fraction]) -> int:
        values = [part.eval(assignment) for part in self.parts]
        best = min(values)
        return values.index(best)

    def to_mapping(self) -> dict[str, Any]:
        return {"parts": [part.to_mapping() for part in self.parts]}


@dataclass(frozen=True)
class StageMap:
    """``s_{n+1} = s_n + step`` with rational ``s_0``."""

    var: str
    step: Fraction
    initial: Fraction

    def __post_init__(self) -> None:
        if not self.var:
            raise ValueError("StageMap.var must be non-empty")
        object.__setattr__(self, "var", str(self.var))
        object.__setattr__(self, "step", _as_frac(self.step))
        object.__setattr__(self, "initial", _as_frac(self.initial))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StageMap:
        step = _pair_to_frac(raw.get("step", 0))
        initial = _pair_to_frac(raw.get("initial", 0))
        if step is None or initial is None:
            raise ValueError("StageMap step/initial must be rationals")
        return cls(str(raw.get("var", "s")), step, initial)

    def value(self, n: int) -> Fraction:
        if n < 0:
            raise ValueError("stage index must be >= 0")
        return self.initial + self.step * n

    def to_mapping(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "step": _rat_pair(self.step),
            "initial": _rat_pair(self.initial),
        }


@dataclass(frozen=True)
class SideCondition:
    """``form(assignment) ⋈ bound`` with a rational bound."""

    form: AffineForm
    sense: Sense
    bound: Fraction
    name: str = ""

    def __post_init__(self) -> None:
        if self.sense not in {"le", "lt", "ge", "gt"}:
            raise ValueError(f"unknown side-condition sense {self.sense!r}")
        object.__setattr__(self, "bound", _as_frac(self.bound))
        object.__setattr__(self, "name", str(self.name))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SideCondition:
        form_raw = raw.get("form")
        if not isinstance(form_raw, Mapping):
            raise ValueError("SideCondition.form must be a mapping")
        bound = _pair_to_frac(raw.get("bound", 0))
        if bound is None:
            raise ValueError("SideCondition.bound must be a rational")
        sense = str(raw.get("sense", "le"))
        if sense not in {"le", "lt", "ge", "gt"}:
            raise ValueError(f"unknown side-condition sense {sense!r}")
        return cls(
            AffineForm.from_mapping(form_raw),
            cast(Sense, sense),
            bound,
            name=str(raw.get("name", "")),
        )

    def holds(self, assignment: Mapping[str, Fraction]) -> bool:
        value = self.form.eval(assignment)
        if self.sense == "le":
            return value <= self.bound
        if self.sense == "lt":
            return value < self.bound
        if self.sense == "ge":
            return value >= self.bound
        return value > self.bound

    def to_mapping(self) -> dict[str, Any]:
        return {
            "form": self.form.to_mapping(),
            "sense": self.sense,
            "bound": _rat_pair(self.bound),
            "name": self.name,
        }


@dataclass(frozen=True)
class MarginObligation:
    """``next_quantity(s_{n+1}) < quantity(s_n) + gain(s_n)``.

    ``next_quantity`` defaults to ``quantity``, recovering the standard
    form ``q(s_{n+1}) < q(s_n) + gain(s_n)``. A cross-quantity comparison
    (``wave(s+step) < meanUpdate(s)``) sets ``next_quantity`` separately
    and uses a zero gain.
    """

    name: str
    quantity: AffineForm
    gain: MinForm
    next_quantity: AffineForm | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        if not self.name:
            raise ValueError("MarginObligation.name must be non-empty")

    @property
    def lhs(self) -> AffineForm:
        return self.quantity if self.next_quantity is None else self.next_quantity

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MarginObligation:
        qty_raw = raw.get("quantity")
        gain_raw = raw.get("gain")
        if not isinstance(qty_raw, Mapping) or not isinstance(gain_raw, Mapping):
            raise ValueError("MarginObligation quantity/gain must be mappings")
        next_raw = raw.get("next_quantity")
        next_qty = AffineForm.from_mapping(next_raw) if isinstance(next_raw, Mapping) else None
        return cls(
            str(raw.get("name", "")),
            AffineForm.from_mapping(qty_raw),
            MinForm.from_mapping(gain_raw),
            next_qty,
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "quantity": self.quantity.to_mapping(),
            "gain": self.gain.to_mapping(),
        }
        if self.next_quantity is not None:
            payload["next_quantity"] = self.next_quantity.to_mapping()
        return payload


@dataclass(frozen=True)
class ConvergenceLedger:
    """A stage-indexed rational budget plus named external premises."""

    name: str
    stage: StageMap
    parameters: Mapping[str, Fraction]
    side_conditions: tuple[SideCondition, ...]
    obligations: tuple[MarginObligation, ...]
    parent: str = ""
    external_premises: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConvergenceLedger.name must be non-empty")
        if not self.obligations:
            raise ValueError("ConvergenceLedger requires at least one obligation")
        params = {str(k): _as_frac(v) for k, v in dict(self.parameters).items()}
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "side_conditions", tuple(self.side_conditions))
        object.__setattr__(self, "obligations", tuple(self.obligations))
        object.__setattr__(self, "external_premises", tuple(self.external_premises))
        object.__setattr__(self, "parent", str(self.parent))
        object.__setattr__(self, "name", str(self.name))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ConvergenceLedger:
        stage_raw = raw.get("stage")
        if not isinstance(stage_raw, Mapping):
            raise ValueError("ConvergenceLedger.stage must be a mapping")
        params_raw = raw.get("parameters", {})
        if not isinstance(params_raw, Mapping):
            raise ValueError("ConvergenceLedger.parameters must be a mapping")
        params: dict[str, Fraction] = {}
        for key, value in params_raw.items():
            parsed = _pair_to_frac(value)
            if parsed is None:
                raise ValueError(f"parameter {key!r} must be a rational")
            params[str(key)] = parsed
        sides_raw = raw.get("side_conditions", ())
        obls_raw = raw.get("obligations", ())
        if not isinstance(sides_raw, Sequence) or isinstance(sides_raw, str | bytes):
            raise ValueError("side_conditions must be a sequence")
        if not isinstance(obls_raw, Sequence) or isinstance(obls_raw, str | bytes):
            raise ValueError("obligations must be a sequence")
        sides = []
        for item in sides_raw:
            if not isinstance(item, Mapping):
                raise ValueError("side condition must be a mapping")
            sides.append(SideCondition.from_mapping(item))
        obls = []
        for item in obls_raw:
            if not isinstance(item, Mapping):
                raise ValueError("obligation must be a mapping")
            obls.append(MarginObligation.from_mapping(item))
        premises_raw = raw.get("external_premises", ())
        if isinstance(premises_raw, str):
            premises: tuple[str, ...] = (premises_raw,)
        elif isinstance(premises_raw, Sequence):
            premises = tuple(str(item) for item in premises_raw)
        else:
            raise ValueError("external_premises must be a sequence of strings")
        return cls(
            name=str(raw.get("name", "")),
            stage=StageMap.from_mapping(stage_raw),
            parameters=params,
            side_conditions=tuple(sides),
            obligations=tuple(obls),
            parent=str(raw.get("parent", "")),
            external_premises=premises,
        )

    def assignment_at(self, n: int) -> dict[str, Fraction]:
        out = dict(self.parameters)
        out[self.stage.var] = self.stage.value(n)
        return out

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage.to_mapping(),
            "parameters": {k: _rat_pair(v) for k, v in sorted(self.parameters.items())},
            "side_conditions": [side.to_mapping() for side in self.side_conditions],
            "obligations": [obl.to_mapping() for obl in self.obligations],
            "parent": self.parent,
            "external_premises": list(self.external_premises),
        }


@dataclass(frozen=True)
class ResidualReport:
    """One gain-part residual, decided on the feasible stage interval."""

    obligation: str
    part_index: int
    residual: Fraction
    slope: Fraction
    holds: bool
    binding: bool
    tipping: tuple[str, str, Fraction] | None

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "obligation": self.obligation,
            "part_index": self.part_index,
            "residual": _rat_pair(self.residual),
            "slope": _rat_pair(self.slope),
            "holds": self.holds,
            "binding": self.binding,
        }
        if self.tipping is not None:
            name, sense, value = self.tipping
            payload["tipping"] = {
                "parameter": name,
                "sense": sense,
                "value": _rat_pair(value),
            }
        return payload


@dataclass(frozen=True)
class LedgerReport:
    """Exact decision of a :class:`ConvergenceLedger`."""

    holds: bool
    strength: ClaimStrength
    failing: tuple[str, ...]
    residuals: tuple[ResidualReport, ...]
    binding_threshold: tuple[str, str, Fraction] | None
    stage_invariant: bool
    sides_hold: bool

    def to_mapping(self) -> dict[str, Any]:
        threshold: dict[str, Any] | None = None
        if self.binding_threshold is not None:
            name, sense, value = self.binding_threshold
            threshold = {
                "parameter": name,
                "sense": sense,
                "value": _rat_pair(value),
            }
        return {
            "holds": self.holds,
            "strength": self.strength,
            "failing": list(self.failing),
            "residuals": [row.to_mapping() for row in self.residuals],
            "binding_threshold": threshold,
            "stage_invariant": self.stage_invariant,
            "sides_hold": self.sides_hold,
        }


@dataclass(frozen=True)
class LedgerCertificateReport:
    """Sealed certificate plus the kernel / Mathlib flags for this spec.

    ``mathlib_verified`` is always ``False`` on this path: the kernel
    emitter discharges integer cross-multiplications, not the Mathlib
    ``linarith`` lemmas.
    """

    certificate: dict[str, Any]
    obligation: Obligation
    report: LedgerReport
    theorem_prover_verified: bool
    mathlib_verified: bool
    lean: LeanCheckResult | None
    wall_seconds: float | None


def _compare(value: Fraction, sense: Sense, bound: Fraction) -> bool:
    if sense == "le":
        return value <= bound
    if sense == "lt":
        return value < bound
    if sense == "ge":
        return value >= bound
    return value > bound


def _stage_interval(
    ledger: ConvergenceLedger,
) -> tuple[Fraction | None, bool, Fraction | None, bool]:
    """Feasible interval for the stage variable after substituting parameters.

    Returns ``(lo, lo_closed, hi, hi_closed)``. ``None`` means unbounded.
    """
    lo: Fraction | None = None
    hi: Fraction | None = None
    lo_closed = True
    hi_closed = True
    params = dict(ledger.parameters)
    for side in ledger.side_conditions:
        reduced = side.form.substitute(params)
        leftover = reduced.names() - {ledger.stage.var}
        if leftover:
            continue
        slope = reduced.coeff(ledger.stage.var)
        if slope == 0:
            continue
        # slope * s ⋈ bound - const  =>  s ⋈ (bound - const) / slope, flipped if slope < 0.
        rhs = (side.bound - reduced.const) / slope
        sense = side.sense
        if slope < 0:
            sense = _SENSE_FLIP[sense]
        if sense in {"ge", "gt"}:
            closed = sense == "ge"
            if lo is None or rhs > lo or (rhs == lo and not closed):
                lo, lo_closed = rhs, closed
        else:
            closed = sense == "le"
            if hi is None or rhs < hi or (rhs == hi and not closed):
                hi, hi_closed = rhs, closed
    if ledger.stage.step > 0:
        start = ledger.stage.initial
        if lo is None or start > lo or (start == lo and not lo_closed):
            lo, lo_closed = start, True
    elif ledger.stage.step < 0:
        start = ledger.stage.initial
        if hi is None or start < hi or (start == hi and not hi_closed):
            hi, hi_closed = start, True
    return lo, lo_closed, hi, hi_closed


def _interval_empty(
    lo: Fraction | None,
    lo_closed: bool,
    hi: Fraction | None,
    hi_closed: bool,
) -> bool:
    if lo is None or hi is None:
        return False
    if lo < hi:
        return False
    if lo > hi:
        return True
    return not (lo_closed and hi_closed)


def _affine_negative_on_interval(
    form: AffineForm,
    var: str,
    lo: Fraction | None,
    lo_closed: bool,
    hi: Fraction | None,
    hi_closed: bool,
) -> bool:
    """``form < 0`` for every point of a (possibly unbounded) interval."""
    if _interval_empty(lo, lo_closed, hi, hi_closed):
        return False
    slope = form.coeff(var)
    if form.names() - {var}:
        raise ValueError("residual still depends on a non-stage variable")

    def at(point: Fraction) -> Fraction:
        return form.const + slope * point

    if slope == 0:
        return form.const < 0
    if slope > 0:
        if hi is None:
            return False
        value = at(hi)
        # Closed: need form(hi) < 0. Open: form(x) < form(hi) for x < hi,
        # so form(hi) <= 0 is enough (and form(hi) > 0 is a counterexample).
        return value < 0 if hi_closed else value <= 0
    if lo is None:
        return False
    value = at(lo)
    return value < 0 if lo_closed else value <= 0


def _tipping(
    residual: AffineForm,
    parameter: str,
    assignment: Mapping[str, Fraction],
) -> tuple[str, str, Fraction] | None:
    """Threshold on ``parameter`` making ``residual < 0``, if affine in it."""
    reduced = residual.substitute({k: v for k, v in assignment.items() if k != parameter})
    if reduced.names() - {parameter}:
        return None
    slope = reduced.coeff(parameter)
    if slope == 0:
        return None
    threshold = -reduced.const / slope
    sense = "lt" if slope > 0 else "gt"
    return parameter, sense, threshold


def _most_restrictive(
    tips: Sequence[tuple[str, str, Fraction]],
) -> tuple[str, str, Fraction] | None:
    uppers = [tip for tip in tips if tip[1] == "lt"]
    if not uppers:
        lowers = [tip for tip in tips if tip[1] == "gt"]
        if not lowers:
            return None
        return max(lowers, key=lambda tip: tip[2])
    return min(uppers, key=lambda tip: tip[2])


def sides_hold(ledger: ConvergenceLedger, *, n: int = 0) -> bool:
    assignment = ledger.assignment_at(n)
    return all(side.holds(assignment) for side in ledger.side_conditions)


def stage_invariant(ledger: ConvergenceLedger) -> bool:
    """Every side condition is preserved by ``s -> s + step`` on the feasible set."""
    step = ledger.stage.step
    var = ledger.stage.var
    params = dict(ledger.parameters)
    for side in ledger.side_conditions:
        reduced = side.form.substitute(params)
        delta = reduced.coeff(var) * step
        if delta == 0:
            continue
        if side.sense in {"ge", "gt"} and delta < 0:
            return False
        if side.sense in {"le", "lt"} and delta > 0:
            return False
    return True


def stage_invariant_obligation(ledger: ConvergenceLedger) -> Obligation:
    """Finite payload: each side condition is preserved by the stage map."""
    holds = stage_invariant(ledger)
    pairs = [_rat_pair(ledger.stage.step), _rat_pair(ledger.stage.initial)]
    payload: dict[str, Any] = {
        "type": "convergence_ledger_stage_invariant",
        "kind": "convergence_ledger_stage_invariant",
        "name": ledger.name,
        "holds": holds,
        "stage": ledger.stage.to_mapping(),
        "side_conditions": [side.to_mapping() for side in ledger.side_conditions],
    }
    max_abs = max((abs(int(item)) for pair in pairs for item in pair), default=1)
    return Obligation("convergence_ledger_stage_invariant", payload, holds, max_abs)


def _part_residual(obl: MarginObligation, part: AffineForm, stage: StageMap) -> AffineForm:
    """``lhs(s+step) - quantity(s) - part(s)``, still affine."""
    return obl.lhs.shift(stage.var, stage.step) - obl.quantity - part


def check_ledger(ledger: ConvergenceLedger) -> LedgerReport:
    """Decide every margin exactly over ``Q``. No floats."""
    invariant = stage_invariant(ledger)
    sides = sides_hold(ledger)
    lo, lo_closed, hi, hi_closed = _stage_interval(ledger)
    residuals: list[ResidualReport] = []
    failing: list[str] = []
    tips: list[tuple[str, str, Fraction]] = []
    assignment0 = ledger.assignment_at(0)

    if _interval_empty(lo, lo_closed, hi, hi_closed) or not sides:
        failing.append("infeasible_side_conditions")

    for obl in ledger.obligations:
        part_holds: list[bool] = []
        part_values: list[Fraction] = []
        part_rows: list[ResidualReport] = []
        for index, part in enumerate(obl.gain.parts):
            residual_form = _part_residual(obl, part, ledger.stage).substitute(
                ledger.parameters
            )
            value = residual_form.eval({ledger.stage.var: ledger.stage.initial})
            slope = residual_form.coeff(ledger.stage.var)
            holds = _affine_negative_on_interval(
                residual_form, ledger.stage.var, lo, lo_closed, hi, hi_closed
            )
            tip = None
            for param in ledger.parameters:
                candidate = _tipping(
                    _part_residual(obl, part, ledger.stage),
                    param,
                    assignment0,
                )
                if candidate is not None:
                    tip = candidate
                    tips.append(candidate)
                    break
            part_holds.append(holds)
            part_values.append(value)
            part_rows.append(
                ResidualReport(
                    obligation=obl.name,
                    part_index=index,
                    residual=value,
                    slope=slope,
                    holds=holds,
                    binding=False,
                    tipping=tip,
                )
            )
        if part_values:
            binding_index = part_values.index(max(part_values))
        else:
            binding_index = 0
        for index, row in enumerate(part_rows):
            marked = ResidualReport(
                obligation=row.obligation,
                part_index=row.part_index,
                residual=row.residual,
                slope=row.slope,
                holds=row.holds,
                binding=index == binding_index,
                tipping=row.tipping,
            )
            residuals.append(marked)
        if not all(part_holds):
            failing.append(obl.name)

    if not invariant:
        failing.append("stage_map_not_invariant")

    holds = not failing
    if not holds:
        strength: ClaimStrength = "BLOCKED"
    elif ledger.external_premises:
        strength = "CONDITIONAL"
    else:
        strength = "PROVED"
    return LedgerReport(
        holds=holds,
        strength=strength,
        failing=tuple(failing),
        residuals=tuple(residuals),
        binding_threshold=_most_restrictive(tips),
        stage_invariant=invariant,
        sides_hold=sides,
    )


def claim_strength(ledger: ConvergenceLedger, report: LedgerReport | None = None) -> ClaimStrength:
    decided = report if report is not None else check_ledger(ledger)
    return decided.strength


def derived_parent_flags(
    ledger: ConvergenceLedger,
    report: LedgerReport | None = None,
) -> dict[str, bool]:
    """Parent flags earned only by a discharged ledger with empty premises."""
    decided = report if report is not None else check_ledger(ledger)
    earned = decided.holds and not ledger.external_premises
    parent = ledger.parent.lower()
    return {
        "navier_stokes_proof_claim": earned
        and ("navier" in parent or "euler" in parent),
        "yang_mills_mass_gap_claim": earned
        and ("yang" in parent or "mills" in parent),
    }


def payload_earns_parent_claim(payload: Mapping[str, Any], key: str) -> bool:
    """``True`` iff ``payload`` is a discharged empty-premise ledger for ``key``.

    A nested ``ledger_payload`` (used when the sealed body is an
    inequality-engine certificate) is accepted on the same terms.
    """
    candidate: Mapping[str, Any] = payload
    if payload.get("type") != PAYLOAD_LEDGER:
        nested = payload.get("ledger_payload")
        if not isinstance(nested, Mapping):
            return False
        candidate = nested
    if candidate.get("type") != PAYLOAD_LEDGER:
        return False
    if not bool(candidate.get("holds")):
        return False
    premises = candidate.get("external_premises", ())
    if isinstance(premises, str):
        nonempty = bool(premises)
    elif isinstance(premises, Sequence):
        nonempty = any(str(item) for item in premises)
    else:
        return False
    if nonempty:
        return False
    parent = str(candidate.get("parent", "")).lower()
    if key == "navier_stokes_proof_claim":
        return "navier" in parent or "euler" in parent
    if key == "yang_mills_mass_gap_claim":
        return "yang" in parent or "mills" in parent
    return False


def honesty_payload(
    ledger: ConvergenceLedger | None = None,
    report: LedgerReport | None = None,
) -> dict[str, bool]:
    """Sealed honesty. Parent flags are derived, never asserted."""
    flags = {
        "unproven_claim": False,
        "collapse_limit_in_lean": False,
        "function_class_in_lean": False,
        "mathlib_path_used": False,
        "navier_stokes_proof_claim": False,
        "yang_mills_mass_gap_claim": False,
        "continuum_pde_claim": False,
        "continuum_claim": False,
    }
    if ledger is not None:
        flags.update(derived_parent_flags(ledger, report))
    return flags


def ledger_obligation(ledger: ConvergenceLedger) -> Obligation:
    """Finite residual payload ready to seal."""
    report = check_ledger(ledger)
    residuals = [row.to_mapping() for row in report.residuals]
    rat_pairs: list[list[str]] = []
    for row in residuals:
        residual = row.get("residual")
        if isinstance(residual, list):
            rat_pairs.append([str(residual[0]), str(residual[1])])
    payload: dict[str, Any] = {
        "type": PAYLOAD_LEDGER,
        "kind": PAYLOAD_LEDGER,
        "name": ledger.name,
        "parent": ledger.parent,
        "external_premises": list(ledger.external_premises),
        "holds": report.holds,
        "strength": report.strength,
        "failing": list(report.failing),
        "family": ledger.name,
        "stage": ledger.stage.to_mapping(),
        "parameters": {k: _rat_pair(v) for k, v in sorted(ledger.parameters.items())},
        "residuals": residuals,
        "binding_threshold": (
            {
                "parameter": report.binding_threshold[0],
                "sense": report.binding_threshold[1],
                "value": _rat_pair(report.binding_threshold[2]),
            }
            if report.binding_threshold is not None
            else None
        ),
        "stage_invariant": report.stage_invariant,
        "sides_hold": report.sides_hold,
        "ledger": ledger.to_mapping(),
    }
    max_abs = 1
    for pair in rat_pairs:
        max_abs = max(max_abs, abs(int(pair[0])), abs(int(pair[1])))
    return Obligation(PAYLOAD_LEDGER, payload, report.holds, max_abs)


def ledger_to_inequality_system(ledger: ConvergenceLedger) -> Any:
    """Encode discharge as a 1-D linear feasibility problem.

    ``x = 0`` is a witness iff every residual is strictly negative; the
    empty box ``1 x <= -1``, ``-1 x <= -1`` is used otherwise. The
    serialized ledger travels in ``data`` so replay can re-run
    :func:`check_ledger` independently of the linear backend.
    """
    from omnibias.core.proof.inequality import InequalitySystem

    report = check_ledger(ledger)
    if report.holds:
        a_mat = [["1"], ["-1"]]
        b_vec = ["0", "0"]
    else:
        a_mat = [["1"], ["-1"]]
        b_vec = ["-1", "-1"]
    return InequalitySystem(
        sort="linear",
        existential=True,
        name=ledger.name,
        data={
            "A": a_mat,
            "b": b_vec,
            "type": PAYLOAD_LEDGER,
            "ledger": ledger.to_mapping(),
            "report": report.to_mapping(),
        },
    )


def replay_ledger_certificate(certificate: Mapping[str, Any]) -> bool | None:
    """Re-run :func:`check_ledger` on the sealed ledger. Core-pure."""
    payload = certificate.get("payload")
    if not isinstance(payload, Mapping) or payload.get("type") != PAYLOAD_LEDGER:
        return None
    ledger_raw = payload.get("ledger")
    if not isinstance(ledger_raw, Mapping):
        return False
    try:
        ledger = ConvergenceLedger.from_mapping(ledger_raw)
    except ValueError:
        return False
    report = check_ledger(ledger)
    return report.holds == bool(payload.get("holds")) and report.strength == str(
        payload.get("strength", "")
    )


def _seal_report(
    ledger: ConvergenceLedger,
    *,
    run_lean: bool,
) -> LedgerCertificateReport:
    report = check_ledger(ledger)
    obligation = ledger_obligation(ledger)
    if obligation.max_abs_int > _INT_CAP:
        raise ValueError(
            f"obligation integers exceed cap {_INT_CAP}: {obligation.max_abs_int}"
        )
    cert = make_certificate(
        claim="convergence-ledger margins are rational identities",
        payload=obligation.payload,
        honesty=honesty_payload(ledger, report),
    )
    lean: LeanCheckResult | None = None
    verified = False
    wall: float | None = None
    if run_lean:
        t0 = time.perf_counter()
        lean = check_certificate(cert)
        wall = time.perf_counter() - t0
        verified = bool(lean.verified)
    return LedgerCertificateReport(
        certificate=cert,
        obligation=obligation,
        report=report,
        theorem_prover_verified=verified,
        mathlib_verified=False,
        lean=lean,
        wall_seconds=wall,
    )


def seal_ledger_certificate(
    ledger: ConvergenceLedger,
    *,
    run_lean: bool = True,
) -> LedgerCertificateReport:
    """Seal the v1 certificate and optionally drive the Lean kernel.

    ``theorem_prover_verified`` is set only on a genuine kernel pass.
    ``mathlib_verified`` stays ``False`` on this path.
    """
    return _seal_report(ledger, run_lean=run_lean)


def digest_is_valid(cert: Mapping[str, Any]) -> bool:
    return verify_certificate_digest(cert)


def _affine(const: Fraction | int | str, **coeffs: Fraction | int | str) -> AffineForm:
    return AffineForm(_as_frac(const), _coeffs_tuple(coeffs))


def navier_stokes_exponent_ledger() -> ConvergenceLedger:
    """Faithful transcription of OpenAI's ``NavierStokes/ExponentLedger``.

    ``sigma_0 = 1/5``, ``step = 1/10``, ``kappa = 1/100000``. The five
    margins of ``all_stage_arithmetic`` plus the ``.17`` bar-residual
    gain whose binding threshold is ``kappa < 1/200``. External
    premises are exactly what that file disclaims.
    """
    sigma = "sigma"
    wave = _affine(Fraction(1, 2), sigma=1)
    mean = _affine(1, sigma=1)
    mean_update = _affine(1, sigma=1, kappa=Fraction(-2))
    particular = MinForm(
        (
            _affine(Fraction(1, 2), kappa=Fraction(-3)),
            _affine(Fraction(1, 2), kappa=Fraction(-1)),
            _affine(Fraction(1, 2), sigma=1, kappa=Fraction(-1)),
            _affine(Fraction(2, 5)),
        )
    )
    signed = MinForm(
        (
            _affine(Fraction(1, 2), kappa=Fraction(-4)),
            _affine(Fraction(2, 5), kappa=Fraction(-1)),
            _affine(Fraction(1, 2), kappa=Fraction(-2)),
            _affine(Fraction(1, 2), sigma=1, kappa=Fraction(-3)),
        )
    )
    signed_bar = MinForm(
        (
            _affine(Fraction(9, 50), kappa=Fraction(-2)),
            _affine(Fraction(1, 2), kappa=Fraction(-3)),
            _affine(0, sigma=1, kappa=Fraction(-3)),
            _affine(1, kappa=Fraction(-1)),
            _affine(1, kappa=Fraction(-2)),
        )
    )
    completed_mean = MinForm(
        (
            _affine(Fraction(17, 100)),
            _affine(1, kappa=Fraction(-4)),
        )
    )
    return ConvergenceLedger(
        name="navier_stokes_exponent",
        stage=StageMap(var=sigma, step=Fraction(1, 10), initial=Fraction(1, 5)),
        parameters={"kappa": Fraction(1, 100000)},
        side_conditions=(
            SideCondition(_affine(0, sigma=1), "ge", Fraction(1, 5), name="sigma_floor"),
            SideCondition(_affine(0, kappa=1), "ge", Fraction(0), name="kappa_nonneg"),
            SideCondition(
                _affine(0, kappa=1), "le", Fraction(1, 100000), name="kappa_admissible"
            ),
        ),
        obligations=(
            MarginObligation("particular", wave, particular),
            MarginObligation("signed", wave, signed),
            MarginObligation(
                "mean_update_wave",
                mean_update,
                MinForm.singleton(AffineForm.constant(0)),
                next_quantity=wave,
            ),
            MarginObligation("completed_mean", mean, completed_mean),
            MarginObligation(
                "completed_defect",
                mean,
                MinForm.singleton(_affine(Fraction(9, 10), kappa=Fraction(-4))),
            ),
            MarginObligation(
                "signed_bar",
                AffineForm.constant(0),
                signed_bar,
                next_quantity=AffineForm.constant(Fraction(17, 100)),
            ),
        ),
        parent=NS_PARENT,
        external_premises=NS_EXTERNAL_PREMISES,
    )


def strong_coupling_polymer_ledger() -> ConvergenceLedger:
    """Kotecký–Preiss majorants locked in ``Check/Polymer.lean``.

    ``polymerBacktrack(4) = 15 < 20 = polymerFirstStep(4)`` and
    ``15 < 24 = polymerCrude(4)``. Degenerate stage (unused). External
    premises name the rungs that separate this arithmetic from the
    mass gap.
    """
    backtrack = AffineForm.constant(15)
    first_step = AffineForm.constant(20)
    crude = AffineForm.constant(24)
    return ConvergenceLedger(
        name="strong_coupling_polymer",
        stage=StageMap(var="n", step=Fraction(0), initial=Fraction(0)),
        parameters={"d": Fraction(4)},
        side_conditions=(
            SideCondition(_affine(0, d=1), "ge", Fraction(4), name="dimension_four"),
        ),
        obligations=(
            MarginObligation(
                "backtrack_lt_first_step",
                AffineForm.constant(0),
                MinForm.singleton(first_step),
                next_quantity=backtrack,
            ),
            MarginObligation(
                "backtrack_lt_crude",
                AffineForm.constant(0),
                MinForm.singleton(crude),
                next_quantity=backtrack,
            ),
        ),
        parent=YM_PARENT,
        external_premises=YM_EXTERNAL_PREMISES,
    )


NS_SCALE_EXTERNAL_PREMISES: tuple[str, ...] = (
    "analytic classes of the slow base and the high-frequency packets",
    "construction of each pulse family and the cutoff summation",
    "PDE residual estimates that the scale ledger treats as given",
)


def navier_stokes_scale_ledger() -> ConvergenceLedger:
    """Geometry / energy / ``kappa_s`` side conditions of the forced construction.

    ``h = 1/200``, ``kappa_s = 1/100000``. Margins are ``h < 1/100``
    (geometry) and ``h < 1/6`` (energy). Binding threshold is the
    stricter geometry bound. Degenerate stage. Premises stay nonempty.
    """
    h = AffineForm.variable("h")
    return ConvergenceLedger(
        name="navier_stokes_scale",
        stage=StageMap(var="n", step=Fraction(0), initial=Fraction(0)),
        parameters={"h": Fraction(1, 200), "kappa_s": Fraction(1, 100000)},
        side_conditions=(
            SideCondition(h, "ge", Fraction(0), name="h_nonneg"),
            SideCondition(h, "lt", Fraction(1, 100), name="h_geometry"),
            SideCondition(h, "lt", Fraction(1, 6), name="h_energy"),
            SideCondition(
                AffineForm.variable("kappa_s"), "ge", Fraction(0), name="kappa_s_nonneg"
            ),
            SideCondition(
                AffineForm.variable("kappa_s"),
                "le",
                Fraction(1, 100000),
                name="kappa_s_admissible",
            ),
        ),
        obligations=(
            MarginObligation(
                "geometry",
                AffineForm.constant(0),
                MinForm.singleton(AffineForm.constant(Fraction(1, 100))),
                next_quantity=h,
            ),
            MarginObligation(
                "energy",
                AffineForm.constant(0),
                MinForm.singleton(AffineForm.constant(Fraction(1, 6))),
                next_quantity=h,
            ),
        ),
        parent="Navier-Stokes forced blowup (Clay C/D)",
        external_premises=NS_SCALE_EXTERNAL_PREMISES,
    )


def failing_margin_ledger() -> ConvergenceLedger:
    """Deliberately broken particular-gain ledger (negative control)."""
    wave = _affine(Fraction(1, 2), sigma=1)
    return ConvergenceLedger(
        name="failing_particular",
        stage=StageMap(var="sigma", step=Fraction(1, 10), initial=Fraction(1, 5)),
        parameters={"kappa": Fraction(1, 100000)},
        side_conditions=(
            SideCondition(_affine(0, sigma=1), "ge", Fraction(1, 5), name="sigma_floor"),
        ),
        obligations=(
            MarginObligation(
                "particular",
                wave,
                MinForm.singleton(AffineForm.constant(Fraction(1, 100))),
            ),
        ),
        parent=NS_PARENT,
        external_premises=NS_EXTERNAL_PREMISES,
    )


def unbounded_slope_ledger(*, holds: bool) -> ConvergenceLedger:
    """Residual affine in an unbounded stage variable, decided by slope sign."""
    if holds:
        next_qty = _affine(0, s=Fraction(-1))
        gain = AffineForm.constant(1)
    else:
        next_qty = _affine(0, s=1)
        gain = AffineForm.constant(1)
    return ConvergenceLedger(
        name="unbounded_slope_holds" if holds else "unbounded_slope_fails",
        stage=StageMap(var="s", step=Fraction(1), initial=Fraction(0)),
        parameters={},
        side_conditions=(
            SideCondition(AffineForm.variable("s"), "ge", Fraction(0), name="s_nonneg"),
        ),
        obligations=(
            MarginObligation(
                "outgoing",
                AffineForm.constant(0),
                MinForm.singleton(gain),
                next_quantity=next_qty,
            ),
        ),
        parent="",
        external_premises=(),
    )


def empty_premise_discharged_ledger() -> ConvergenceLedger:
    """Toy ledger that earns a parent flag (empty premises, all margins hold)."""
    return ConvergenceLedger(
        name="empty_premise_toy",
        stage=StageMap(var="s", step=Fraction(1), initial=Fraction(0)),
        parameters={},
        side_conditions=(
            SideCondition(AffineForm.variable("s"), "ge", Fraction(0), name="s_nonneg"),
        ),
        obligations=(
            MarginObligation(
                "constant_gain",
                AffineForm.variable("s"),
                MinForm.singleton(AffineForm.constant(2)),
            ),
        ),
        parent=NS_PARENT,
        external_premises=(),
    )


def curated_convergence_ledgers() -> tuple[ConvergenceLedger, ...]:
    return (
        navier_stokes_exponent_ledger(),
        strong_coupling_polymer_ledger(),
        navier_stokes_scale_ledger(),
    )


def _register() -> None:
    register_catalog(
        CatalogEntry(
            kind=PAYLOAD_LEDGER,
            obligation=(
                "finite rational stage-budget margins of an iterative construction"
            ),
            parent="staged iterative constructions (NS exponent ledger / YM polymer)",
            parent_status="already_true",
            package="omnibias.core.proof.obligations.convergence_ledger",
            mode="exact_replay",
            complete=True,
            existential=False,
        ),
        curated_convergence_ledgers,
    )


_register()


__all__ = [
    "AffineForm",
    "ClaimStrength",
    "ConvergenceLedger",
    "LedgerCertificateReport",
    "LedgerReport",
    "MarginObligation",
    "MinForm",
    "NS_EXTERNAL_PREMISES",
    "NS_PARENT",
    "NS_SCALE_EXTERNAL_PREMISES",
    "PARENT_CLAIM_KEYS",
    "PAYLOAD_LEDGER",
    "ResidualReport",
    "SideCondition",
    "StageMap",
    "YM_EXTERNAL_PREMISES",
    "YM_PARENT",
    "check_ledger",
    "claim_strength",
    "curated_convergence_ledgers",
    "derived_parent_flags",
    "digest_is_valid",
    "empty_premise_discharged_ledger",
    "failing_margin_ledger",
    "honesty_payload",
    "ledger_obligation",
    "ledger_to_inequality_system",
    "navier_stokes_exponent_ledger",
    "navier_stokes_scale_ledger",
    "payload_earns_parent_claim",
    "replay_ledger_certificate",
    "seal_ledger_certificate",
    "sides_hold",
    "stage_invariant",
    "stage_invariant_obligation",
    "strong_coupling_polymer_ledger",
    "unbounded_slope_ledger",
]
