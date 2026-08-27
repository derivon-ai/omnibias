# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Unified inequality engine (theory 09-30).

A front door, not a new solver and not a complexity-class claim. Owning
packages register backends at import; this module never imports them.
The optimizer proposes; an exact ``Q`` / GF(2) / enclosure / tiny enum
proves. A float residual is never an :class:`ExactCheck`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from omnibias.core.proof.catalog import CatalogEntry, register_catalog
from omnibias.core.proof.certificate import make_certificate, schema_errors_v1
from omnibias.core.proof.discovery import ExactCheck

if TYPE_CHECKING:
    from omnibias.core.proof import (
        Certificate,
        Conjecture,
        FunctionProver,
        ProofAttempt,
        ProofMachine,
        Verdict,
    )

VerdictStatus = Literal["PROVED", "DISPROVED", "BLOCKED"]

INEQUALITY_KIND = "inequality_system"
InequalitySort = Literal["linear", "polynomial", "boolean", "csp"]
CheckRole = Literal[
    "witness",
    "empty",
    "positivity",
    "counterexample",
    "inconclusive",
]

_SORTS: frozenset[str] = frozenset({"linear", "polynomial", "boolean", "csp"})
_STACK_MODULES: tuple[str, ...] = (
    "omnibias.convex.inequality",
    "omnibias.sos.inequality",
    "omnibias.boolean.inequality",
    "omnibias.discrete.csp.inequality",
)

_PARENT_FLAGS: dict[str, bool] = {
    "jacobian_conjecture_proof_claim": False,
    "navier_stokes_proof_claim": False,
    "unproven_claim": False,
}


def default_inequality_honesty() -> dict[str, bool]:
    """Engine-level honesty. Backends may raise ``complete_solver`` on a cube."""

    return {
        "complete_solver": False,
        "unsat_proof": False,
        "p_equals_np_claim": False,
        "new_lp_algorithm_claim": False,
        "unsat_from_float_infeasible": False,
        "soft_residual_is_exact_check": False,
        **_PARENT_FLAGS,
    }


@dataclass(frozen=True)
class InequalitySystem:
    """JSON-able inequality payload. ``data`` is sort-specific."""

    sort: InequalitySort
    existential: bool
    data: Mapping[str, Any]
    name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sort": self.sort,
            "existential": self.existential,
            "data": dict(self.data),
            "name": self.name,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> InequalitySystem:
        sort = str(raw.get("sort", ""))
        if sort not in _SORTS:
            raise ValueError(f"unknown inequality sort {sort!r}")
        data_raw = raw.get("data", {})
        if not isinstance(data_raw, Mapping):
            raise ValueError("inequality data must be a mapping")
        return cls(
            sort=cast(InequalitySort, sort),
            existential=bool(raw.get("existential", True)),
            data=dict(data_raw),
            name=str(raw.get("name", "")),
        )


@dataclass(frozen=True)
class Proposal:
    """Float / soft proposal. ``skipped`` is allowed (Boolean exact path)."""

    method: str
    values: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    skipped: bool = False


@dataclass(frozen=True)
class RationalWitness:
    """Rationalized proposal. Values are fraction strings or discrete indices."""

    method: str
    values: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)


class InequalityBackend(Protocol):
    """Owning-package adapter. ``check`` must be exact / enclosure / tiny enum."""

    @property
    def sort(self) -> InequalitySort: ...

    def propose(self, system: InequalitySystem) -> Proposal: ...

    def rationalize(
        self, system: InequalitySystem, proposal: Proposal
    ) -> RationalWitness: ...

    def check(
        self, system: InequalitySystem, witness: RationalWitness
    ) -> ExactCheck: ...


_BACKENDS: dict[str, InequalityBackend] = {}


def register_inequality_backend(backend: InequalityBackend) -> None:
    """Register ``backend`` for its ``sort``. Same object is idempotent."""

    existing = _BACKENDS.get(backend.sort)
    if existing is not None and existing is not backend:
        if type(existing) is not type(backend):
            raise ValueError(f"inequality backend collision for {backend.sort!r}")
    _BACKENDS[backend.sort] = backend


def list_inequality_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def inequality_backend(sort: str) -> InequalityBackend | None:
    return _BACKENDS.get(sort)


def _reset_inequality_backends_for_tests() -> None:
    """Test helper. Not part of the public engine."""

    _BACKENDS.clear()


def load_inequality_stack() -> tuple[str, ...]:
    """Import owning adapters so they register. Core stays clean."""

    import importlib

    loaded: list[str] = []
    for name in _STACK_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        register = getattr(module, "_register", None)
        if callable(register):
            register()
        loaded.append(name)
    return tuple(loaded)


def locked_catalog() -> tuple[dict[str, Any], ...]:
    """Twelve locked instances: three per sort, expected statuses published."""

    high_a = []
    high_b = []
    for i in range(5):
        row_pos = ["0"] * 5
        row_neg = ["0"] * 5
        row_pos[i] = "1"
        row_neg[i] = "-1"
        high_a.append(row_pos)
        high_a.append(row_neg)
        high_b.extend(["-1", "-1"])
    return (
        {
            "name": "linear_box_sat",
            "sort": "linear",
            "existential": True,
            "expected": "PROVED",
            "data": {"A": [["1"], ["-1"]], "b": ["1", "0"]},
        },
        {
            "name": "linear_empty_enum",
            "sort": "linear",
            "existential": True,
            "expected": "DISPROVED",
            "data": {"A": [["1"], ["-1"]], "b": ["-1", "-1"]},
        },
        {
            "name": "linear_highdim_blocked",
            "sort": "linear",
            "existential": True,
            "expected": "BLOCKED",
            "data": {"A": high_a, "b": high_b},
        },
        {
            "name": "poly_constant_one",
            "sort": "polynomial",
            "existential": False,
            "expected": "PROVED",
            "data": {
                "poly_n_vars": 1,
                "poly_terms": [[[0], "1"]],
            },
        },
        {
            "name": "poly_constant_neg",
            "sort": "polynomial",
            "existential": False,
            "expected": "DISPROVED",
            "data": {
                "poly_n_vars": 1,
                "poly_terms": [[[0], "-1"]],
            },
        },
        {
            "name": "poly_x2_y2_blocked",
            "sort": "polynomial",
            "existential": False,
            "expected": "BLOCKED",
            "data": {
                "poly_n_vars": 2,
                "poly_terms": [[[2, 0], "1"], [[0, 2], "1"]],
            },
        },
        {
            "name": "boolean_and_sat",
            "sort": "boolean",
            "existential": True,
            "expected": "PROVED",
            "data": {"tables": [[0, 0, 0, 1]]},
        },
        {
            "name": "boolean_x_not_x_unsat",
            "sort": "boolean",
            "existential": True,
            "expected": "DISPROVED",
            "data": {"tables": [[1, 1]]},
        },
        {
            "name": "boolean_budget_zero",
            "sort": "boolean",
            "existential": True,
            "expected": "BLOCKED",
            "data": {"tables": [[0, 0, 0, 1]], "budget": 0},
        },
        {
            "name": "csp_two_var_sat",
            "sort": "csp",
            "existential": True,
            "expected": "PROVED",
            "data": {
                "variables": [
                    {"name": "x0", "domain": ["a", "b"]},
                    {"name": "x1", "domain": ["a", "b"]},
                ],
                "relations": [{"scope": [0, 1], "allowed": [[0, 0], [1, 1]]}],
                "oracle": False,
            },
        },
        {
            "name": "csp_empty_unsat",
            "sort": "csp",
            "existential": True,
            "expected": "DISPROVED",
            "data": {
                "variables": [
                    {"name": "x0", "domain": ["a", "b"]},
                    {"name": "x1", "domain": ["a", "b"]},
                ],
                "relations": [{"scope": [0, 1], "allowed": []}],
                "oracle": True,
            },
        },
        {
            "name": "csp_anneal_blocked",
            "sort": "csp",
            "existential": True,
            "expected": "BLOCKED",
            "data": {
                "variables": [
                    {"name": "x0", "domain": ["a", "b"]},
                    {"name": "x1", "domain": ["a", "b"]},
                ],
                "relations": [{"scope": [0, 1], "allowed": []}],
                "oracle": False,
            },
        },
    )


def _status_from_check(
    *,
    existential: bool,
    role: str,
) -> VerdictStatus:
    if role == "witness" and existential:
        return "PROVED"
    if role == "empty" and existential:
        return "DISPROVED"
    if role == "positivity" and not existential:
        return "PROVED"
    if role == "counterexample" and not existential:
        return "DISPROVED"
    if role == "empty" and not existential:
        return "PROVED"
    return "BLOCKED"


def _merge_honesty(extra: Mapping[str, Any] | None) -> dict[str, bool]:
    honesty = default_inequality_honesty()
    if extra is None:
        return honesty
    for key, value in extra.items():
        if isinstance(value, bool):
            honesty[str(key)] = value
    honesty["p_equals_np_claim"] = False
    honesty["new_lp_algorithm_claim"] = False
    honesty["unsat_from_float_infeasible"] = False
    honesty["soft_residual_is_exact_check"] = False
    honesty["unproven_claim"] = False
    honesty["jacobian_conjecture_proof_claim"] = False
    honesty["navier_stokes_proof_claim"] = False
    return honesty


def _seal_attempt(
    status: VerdictStatus,
    payload: Mapping[str, Any],
    honesty: Mapping[str, bool],
    detail: str,
) -> ProofAttempt:
    from omnibias.core.proof import ProofAttempt

    cert = make_certificate(
        claim=INEQUALITY_KIND,
        payload=dict(payload),
        honesty=dict(honesty),
    )
    return ProofAttempt(status=status, certificate=cert, detail=detail)


def run_inequality_pipeline(
    system: InequalitySystem,
) -> tuple[VerdictStatus, dict[str, Any], dict[str, bool]]:
    """Propose → rationalize → check. Returns ``(status, payload, honesty)``."""

    backend = _BACKENDS.get(system.sort)
    if backend is None:
        payload: dict[str, Any] = {
            "sort": system.sort,
            "existential": system.existential,
            "role": "inconclusive",
            "status": "BLOCKED",
            "detail": "backend_unavailable",
            "pipeline": [],
            "system": system.as_dict(),
        }
        return "BLOCKED", payload, default_inequality_honesty()
    proposal = backend.propose(system)
    witness = backend.rationalize(system, proposal)
    checked = backend.check(system, witness)
    extra = checked.payload
    role = str(extra.get("role", "inconclusive"))
    if extra.get("soft_residual_is_exact_check"):
        raise ValueError("soft residual cannot become an ExactCheck")
    pipeline: list[str] = []
    if not proposal.skipped:
        pipeline.append("propose")
    pipeline.extend(["rationalize", "check"])
    status = _status_from_check(existential=system.existential, role=role)
    payload = {
        "sort": system.sort,
        "existential": system.existential,
        "role": role,
        "status": status,
        "ok": checked.ok,
        "pipeline": pipeline,
        "propose_method": proposal.method,
        "rationalize_method": witness.method,
        "check_method": str(extra.get("method", "")),
        "witness": list(witness.values),
        "system": system.as_dict(),
        "detail": str(extra.get("detail", "")),
    }
    honesty_raw = extra.get("honesty")
    honesty = _merge_honesty(honesty_raw if isinstance(honesty_raw, Mapping) else None)
    if role == "empty":
        honesty["unsat_proof"] = True
    return status, payload, honesty


def prove_inequality(conjecture: Conjecture) -> ProofAttempt:
    """Dispatch ``inequality_system`` conjectures."""

    raw = dict(conjecture.data)
    if "sort" not in raw and "system" in raw and isinstance(raw["system"], Mapping):
        raw = dict(raw["system"])
    try:
        system = InequalitySystem.from_mapping(raw)
    except ValueError as exc:
        return _seal_attempt(
            "BLOCKED",
            {"role": "inconclusive", "detail": str(exc), "pipeline": []},
            default_inequality_honesty(),
            str(exc),
        )
    if not system.name:
        system = InequalitySystem(
            sort=system.sort,
            existential=system.existential,
            data=system.data,
            name=conjecture.name,
        )
    status, payload, honesty = run_inequality_pipeline(system)
    return _seal_attempt(status, payload, honesty, str(payload.get("detail", "")))


def replay_inequality(certificate: Certificate) -> bool | None:
    """Re-run ``check`` on the sealed witness. Does not re-propose."""

    payload = certificate.get("payload")
    if not isinstance(payload, Mapping):
        return False
    system_raw = payload.get("system")
    if not isinstance(system_raw, Mapping):
        return False
    try:
        system = InequalitySystem.from_mapping(system_raw)
    except ValueError:
        return False
    backend = _BACKENDS.get(system.sort)
    if backend is None:
        return str(payload.get("detail", "")) == "backend_unavailable"
    witness = RationalWitness(
        method=str(payload.get("rationalize_method", "")),
        values=tuple(str(item) for item in payload.get("witness", ())),
        payload={},
    )
    checked = backend.check(system, witness)
    extra = checked.payload
    role = str(extra.get("role", "inconclusive"))
    status = _status_from_check(existential=system.existential, role=role)
    return role == str(payload.get("role", "")) and status == str(
        payload.get("status", status)
    )


def inequality_schema_errors(certificate: Certificate) -> list[str]:
    errors = list(schema_errors_v1(certificate))
    payload = certificate.get("payload")
    if not isinstance(payload, Mapping):
        errors.append("payload missing")
        return errors
    if "role" not in payload:
        errors.append("payload.role missing")
    return errors


def inequality_prover() -> FunctionProver:
    from omnibias.core.proof import FunctionProver

    return FunctionProver(
        name="inequality_system",
        kinds=frozenset({INEQUALITY_KIND}),
        prove_fn=prove_inequality,
        schema_fn=inequality_schema_errors,
        replay_fn=replay_inequality,
    )


def build_inequality_machine() -> ProofMachine:
    """ProofMachine with the inequality kind. Loads owning backends."""

    from omnibias.core.proof import ProofMachine

    load_inequality_stack()
    return ProofMachine().register(inequality_prover())


def solve_inequality(system: InequalitySystem, *, replay: bool = True) -> Verdict:
    """Convenience: one system through :func:`build_inequality_machine`."""

    from omnibias.core.proof import Conjecture

    machine = build_inequality_machine()
    return machine.evaluate(
        Conjecture(
            name=system.name or system.sort,
            kind=INEQUALITY_KIND,
            data=system.as_dict(),
        ),
        replay=replay,
    )


def _register() -> None:
    register_catalog(
        CatalogEntry(
            kind=INEQUALITY_KIND,
            obligation="finite inequality system adjudicated by propose/rationalize/check",
            parent="inequality_solving",
            parent_status="already_true",
            package="omnibias.core.proof.inequality",
            mode="enclosure",
            complete=True,
            existential=True,
        ),
        locked_catalog,
    )


_register()


__all__ = [
    "CheckRole",
    "INEQUALITY_KIND",
    "InequalityBackend",
    "InequalitySort",
    "InequalitySystem",
    "Proposal",
    "RationalWitness",
    "build_inequality_machine",
    "default_inequality_honesty",
    "inequality_backend",
    "inequality_prover",
    "inequality_schema_errors",
    "list_inequality_backends",
    "load_inequality_stack",
    "locked_catalog",
    "prove_inequality",
    "register_inequality_backend",
    "replay_inequality",
    "run_inequality_pipeline",
    "solve_inequality",
]
