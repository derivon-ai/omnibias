# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named-collapse schema: erase a distinction, keep an invariant.

The founding three senses of collapse stay exactly the three documented
in ``docs/theory.md``:

* **bias** -- ``delta -> 0``; surviving object is a derivative
* **temperature** -- ``beta -> inf``; surviving object is a 0/1 indicator
* **enclosure** -- ``width -> 0`` of a *sound enclosure*; surviving
  object is a point plus a proof, or ``Inconclusive``

This module does **not** reimplement those limits. It catalogues them
and refuses a new name that is a rebrand (same moving parameter *and*
same surviving object). A later named collapse earns a registry slot
only when it is distinct on at least one of those two axes.

A float residual is never a proof. ``theorem_prover_verified`` and
``mathlib_verified`` stay false here; they are earned only by a genuine
``lake build``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.interval import Interval

CollapseStatus = Literal["collapsed", "excluded", "inconclusive"]
CollapseRegister = Literal[
    "differentiable",
    "discrete",
    "verified",
    "formal",
    "holonomic",
    "measure",
]


def default_honesty(*, spec_name: str) -> dict[str, bool]:
    """Honesty flags that a named collapse may never forge."""

    name = str(spec_name)
    return {
        "founding_bias_collapse": name == "bias",
        "temperature_collapse": name == "temperature",
        "enclosure_collapse": name == "enclosure",
        "float_residual_is_proof": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "continuum_parent_inferred": False,
    }


@dataclass(frozen=True)
class CollapseSpec:
    """A named collapse: what moves, what the limit is, what survives.

    Parameters
    ----------
    name:
        Stable registry key (``bias``, ``temperature``, ``enclosure``, …).
    parameter:
        The quantity that moves (``delta``, ``beta``, ``width``, …).
    limit:
        The limit value or description (``0``, ``inf``, …).
    surviving_object:
        What remains after the distinction is erased.
    failure:
        First-class failure (cancellation, gap, ``Inconclusive``, …).
    home:
        Dotted module that implements or documents the limit.
    register:
        Which omnibias register the surviving object lives in.
    founding:
        ``True`` only for the three documented senses. New names must
        leave this ``False``.
    """

    name: str
    parameter: str
    limit: str
    surviving_object: str
    failure: str
    home: str
    register: CollapseRegister
    founding: bool = False

    def __post_init__(self) -> None:
        fields = (
            ("name", self.name),
            ("parameter", self.parameter),
            ("limit", self.limit),
            ("surviving_object", self.surviving_object),
            ("failure", self.failure),
            ("home", self.home),
        )
        for label, value in fields:
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"CollapseSpec.{label} must be non-empty")
            if stripped != value:
                object.__setattr__(self, label, stripped)
        if self.founding and self.name not in FOUNDING_NAMES:
            raise ValueError(
                f"founding=True is reserved for {sorted(FOUNDING_NAMES)}; "
                f"got {self.name!r}"
            )


@dataclass(frozen=True)
class DistinctnessReport:
    """Whether two specs mint different objects."""

    left: str
    right: str
    distinct: bool
    reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "distinct": self.distinct,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CollapseOutcome:
    """Shared result of applying a named collapse to a sound enclosure.

    ``collapsed`` -- the enclosure contracted onto the surviving object.
    ``excluded`` -- the candidate (usually ``0``) lies outside the enclosure.
    ``inconclusive`` -- the enclosure still contains the candidate and has
    positive width; this is *not* falsity.
    """

    status: CollapseStatus
    spec_name: str
    surviving: str | int | None
    residual: Interval | None
    detail: str
    honesty: Mapping[str, bool]

    @property
    def collapsed(self) -> bool:
        return self.status == "collapsed"

    @property
    def excluded(self) -> bool:
        return self.status == "excluded"

    @property
    def inconclusive(self) -> bool:
        return self.status == "inconclusive"

    def to_payload(self) -> dict[str, object]:
        residual: dict[str, float] | None
        if self.residual is None:
            residual = None
        else:
            residual = {"lo": self.residual.lo, "hi": self.residual.hi}
        return {
            "status": self.status,
            "spec_name": self.spec_name,
            "surviving": self.surviving,
            "residual": residual,
            "detail": self.detail,
            "honesty": dict(self.honesty),
        }


@dataclass(frozen=True)
class RejectedCollapse:
    """A named idea that failed the registry gate and was removed."""

    spec: CollapseSpec
    reason: str


FOUNDING_NAMES: frozenset[str] = frozenset({"bias", "temperature", "enclosure"})

FOUNDING_COLLAPSES: tuple[CollapseSpec, ...] = (
    CollapseSpec(
        name="bias",
        parameter="delta",
        limit="0",
        surviving_object="derivative",
        failure="1/delta^(K-1) cancellation if the limit is taken in float",
        home="omnibias.core.polynomials",
        register="differentiable",
        founding=True,
    ),
    CollapseSpec(
        name="temperature",
        parameter="beta",
        limit="inf",
        surviving_object="indicator",
        failure="optimality gap",
        home="omnibias.partition.arrangement",
        register="discrete",
        founding=True,
    ),
    CollapseSpec(
        name="enclosure",
        parameter="width",
        limit="0",
        surviving_object="point_plus_proof",
        failure="Inconclusive",
        home="omnibias.core.verified.enclosure_collapse",
        register="verified",
        founding=True,
    ),
)


def are_distinct(left: CollapseSpec, right: CollapseSpec) -> DistinctnessReport:
    """Two specs are a rebrand iff parameter *and* surviving object match."""

    reasons: list[str] = []
    if left.parameter != right.parameter:
        reasons.append("parameter")
    if left.surviving_object != right.surviving_object:
        reasons.append("surviving_object")
    if left.limit != right.limit:
        reasons.append("limit")
    if left.failure != right.failure:
        reasons.append("failure")
    if left.register != right.register:
        reasons.append("register")
    return DistinctnessReport(
        left=left.name,
        right=right.name,
        distinct=left.parameter != right.parameter
        or left.surviving_object != right.surviving_object,
        reasons=tuple(reasons),
    )


def require_sound_enclosure(residual: object) -> Interval:
    """Accept only an :class:`Interval`. A float is not a certificate."""

    if isinstance(residual, Interval):
        return residual
    raise TypeError(
        "a float residual is not a certificate; pass a sound Interval enclosure"
    )


class CollapseRegistry:
    """In-memory catalogue of named collapses, isolated for tests."""

    def __init__(self, *, seed_founding: bool = True) -> None:
        self._active: dict[str, CollapseSpec] = {}
        self._rejected: dict[str, RejectedCollapse] = {}
        if seed_founding:
            for spec in FOUNDING_COLLAPSES:
                self._active[spec.name] = spec

    def get(self, name: str) -> CollapseSpec:
        try:
            return self._active[name]
        except KeyError as exc:
            raise KeyError(f"unknown collapse {name!r}") from exc

    def list_active(self) -> tuple[CollapseSpec, ...]:
        return tuple(self._active[name] for name in sorted(self._active))

    def list_rejected(self) -> tuple[RejectedCollapse, ...]:
        return tuple(self._rejected[name] for name in sorted(self._rejected))

    def founding(self) -> tuple[CollapseSpec, ...]:
        return tuple(spec for spec in self.list_active() if spec.founding)

    def register(self, spec: CollapseSpec) -> CollapseSpec:
        """Add a distinct non-founding spec, or refuse a rebrand."""

        if spec.founding:
            raise ValueError(
                "cannot register a new founding collapse; "
                "the three senses are closed"
            )
        if spec.name in FOUNDING_NAMES:
            raise ValueError(
                f"{spec.name!r} is a founding sense and is already catalogued"
            )
        if spec.name in self._active:
            raise ValueError(f"collapse {spec.name!r} is already registered")
        if spec.name in self._rejected:
            raise ValueError(
                f"collapse {spec.name!r} was rejected: "
                f"{self._rejected[spec.name].reason}"
            )
        for existing in self._active.values():
            report = are_distinct(spec, existing)
            if not report.distinct:
                raise ValueError(
                    f"{spec.name!r} is a rebrand of {existing.name!r}: "
                    f"same parameter {spec.parameter!r} and surviving "
                    f"object {spec.surviving_object!r}"
                )
        self._active[spec.name] = spec
        return spec

    def reject(self, name: str, reason: str) -> RejectedCollapse:
        """Remove a non-founding spec after it fails an empirical gate."""

        stripped = reason.strip()
        if not stripped:
            raise ValueError("reject reason must be non-empty")
        if name in FOUNDING_NAMES:
            raise ValueError(f"cannot reject founding collapse {name!r}")
        if name in self._rejected:
            return self._rejected[name]
        spec = self._active.pop(name, None)
        if spec is None:
            raise KeyError(f"unknown collapse {name!r}")
        record = RejectedCollapse(spec=spec, reason=stripped)
        self._rejected[name] = record
        return record


_REGISTRY = CollapseRegistry()


def get_collapse(name: str) -> CollapseSpec:
    return _REGISTRY.get(name)


def list_collapses() -> tuple[CollapseSpec, ...]:
    return _REGISTRY.list_active()


def list_rejected_collapses() -> tuple[RejectedCollapse, ...]:
    return _REGISTRY.list_rejected()


def register_collapse(spec: CollapseSpec) -> CollapseSpec:
    return _REGISTRY.register(spec)


def reject_collapse(name: str, reason: str) -> RejectedCollapse:
    return _REGISTRY.reject(name, reason)


def reset_collapse_registry() -> None:
    """Restore the process-global registry to the founding three.

    Tests that register or reject names must call this in teardown.
    """

    global _REGISTRY
    _REGISTRY = CollapseRegistry()


__all__ = [
    "CollapseOutcome",
    "CollapseRegister",
    "CollapseRegistry",
    "CollapseSpec",
    "CollapseStatus",
    "DistinctnessReport",
    "FOUNDING_COLLAPSES",
    "FOUNDING_NAMES",
    "RejectedCollapse",
    "are_distinct",
    "default_honesty",
    "get_collapse",
    "list_collapses",
    "list_rejected_collapses",
    "register_collapse",
    "reject_collapse",
    "require_sound_enclosure",
    "reset_collapse_registry",
]
