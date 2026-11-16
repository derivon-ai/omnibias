# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Conformance of a derivative engine to public fixtures, and the dispatch regret of its choices.

A derivative engine is any callable ``engine(quantity, case) -> array`` that returns, for one of the fixture networks, the
Laplacian (``"lap"``), the divergence of the vector field (``"divergence"``), ``log|det(I + J)|`` (``"logdet"``) or the solution
``y`` of ``(I + J) y = rhs`` (``"solve"``). It need not be an omnibias engine: a hand-written function, a third-party library
or a reference implementation conforms the same way. :func:`run_correctness` replays the fixtures against it and reports the
worst relative error per case and quantity; :func:`dispatch_regret` turns the measured time of every program a planner could have
chosen into the ratio of the chosen program to the best one.

The fixtures (``fixtures/cross_backend_operators_v1.json``) store exact float64 weights and points as hexadecimal floats and
references computed in 90-digit ``mpmath`` arithmetic by central differences of the network itself, so no derivative formula of
this package decides the expected values (``scripts/generate_cross_backend_fixtures.py`` regenerates them). The relative error is
``max |got - want| / max(1, |want|)`` entrywise. :func:`reference_engine` is a NumPy forward-mode implementation with no
autodiff dependency that conforms; the JAX and PyTorch front ends provide their engines as ``conformance_engine`` in
``omnibias.jax.operators`` and ``omnibias.torch.operators``.

Conformance is correctness at the fixture sizes, in float64. It says nothing about speed, large inputs or float32; the dispatch
protocol (``docs/stability.md``) is the speed side, and ``docs/precision.md`` the float32 one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Case",
    "CaseResult",
    "FIXTURE",
    "QUANTITIES",
    "dispatch_regret",
    "load_cases",
    "reference_engine",
    "run_correctness",
    "summarize",
]

FIXTURE = Path(__file__).with_name("fixtures") / "cross_backend_operators_v1.json"
QUANTITIES = ("lap", "divergence", "logdet", "solve")
Engine = Callable[[str, "Case"], Any]


@dataclass(frozen=True)
class Case:
    """One fixture network. ``weights[k]`` is ``(W, b)`` with ``W`` of shape ``(out, in)``; every layer but the last is followed
    by ``activation``; ``x`` has shape ``(B, D)`` and ``rhs`` (field cases) the same shape."""

    name: str
    activation: str
    dims: tuple[int, ...]
    weights: tuple[tuple[np.ndarray, np.ndarray], ...]
    x: np.ndarray
    rhs: np.ndarray | None
    reference: Mapping[str, np.ndarray] = field(repr=False)

    @property
    def quantities(self) -> tuple[str, ...]:
        return tuple(q for q in QUANTITIES if q in self.reference)


def _hex(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(float.fromhex(value))
    return np.asarray([_hex(v) for v in value]) if len(value) else np.zeros((0,))


def _dec(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(float(value))
    return np.asarray([_dec(v) for v in value])


def load_cases(path: str | Path | None = None) -> list[Case]:
    """The fixture cases (the shipped file unless ``path`` names another with the same layout)."""
    doc = json.loads(Path(path or FIXTURE).read_text())
    if doc.get("version") != 1:
        raise ValueError(f"fixture version {doc.get('version')!r} is not supported (expected 1)")
    cases = []
    for c in doc["cases"]:
        ref = {k: _dec(v) for k, v in c["reference"].items() if k in QUANTITIES}
        cases.append(
            Case(
                name=c["name"],
                activation=c["activation"],
                dims=tuple(int(d) for d in c["dims"]),
                weights=tuple((_hex(layer["W"]), _hex(layer["b"])) for layer in c["layers"]),
                x=_hex(c["interior"]),
                rhs=_hex(c["rhs"]) if c["dims"][-1] != 1 and len(c.get("rhs", ())) else None,
                reference=ref,
            )
        )
    return cases


@dataclass(frozen=True)
class CaseResult:
    case: str
    quantity: str
    rel_error: float
    tolerance: float
    passed: bool
    error: str | None = None


def _relative_error(got: Any, want: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64)
    if got.size != want.size:
        raise ValueError(f"result has {got.size} entries, expected {want.size}")
    got = got.reshape(want.shape)
    if not np.all(np.isfinite(got)):
        return math.inf
    return float(np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)))) if got.size else 0.0


def run_correctness(
    engine: Engine,
    *,
    tolerance: float = 1e-11,
    cases: Sequence[Case] | None = None,
    quantities: Sequence[str] | None = None,
) -> list[CaseResult]:
    """Replay the fixtures against ``engine``. An engine that raises on a case is a failed case with the message kept."""
    out = []
    for case in cases if cases is not None else load_cases():
        for quantity in case.quantities:
            if quantities is not None and quantity not in quantities:
                continue
            try:
                err = _relative_error(engine(quantity, case), case.reference[quantity])
                out.append(CaseResult(case.name, quantity, err, tolerance, err <= tolerance))
            except Exception as exc:  # noqa: BLE001 - the report keeps what the engine said
                out.append(
                    CaseResult(
                        case.name,
                        quantity,
                        math.inf,
                        tolerance,
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    return out


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    worst = max((r.rel_error for r in results), default=0.0)
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "failed": [(r.case, r.quantity, r.rel_error, r.error) for r in results if not r.passed],
        "worst_relative_error": worst,
    }


# ------------------------------------------------------------------------------------------------ the reference engine


def _tanh(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.tanh(z)
    d1 = 1.0 - s * s
    return s, d1, -2.0 * s * d1


def _silu(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = 1.0 / (1.0 + np.exp(-z))
    q = s * (1.0 - s)
    return z * s, s + z * q, q * (2.0 + z * (1.0 - 2.0 * s))


_ACTIVATIONS = {"tanh": _tanh, "silu": _silu}


def _jacobian_and_laplacians(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """``J`` of shape ``(B, out, D)`` and the per-output Laplacian ``(B, out)`` by second-order forward mode along each axis."""
    if case.activation not in _ACTIVATIONS:
        raise ValueError(f"the reference engine has no derivatives for {case.activation!r}")
    act = _ACTIVATIONS[case.activation]
    B, D = case.x.shape
    out_dim = case.dims[-1]
    jac = np.zeros((B, out_dim, D))
    lap = np.zeros((B, out_dim))
    for i in range(D):
        a, da, dda = case.x, np.zeros_like(case.x), np.zeros_like(case.x)
        da[:, i] = 1.0
        for k, (W, b) in enumerate(case.weights):
            z, dz, ddz = a @ W.T + b, da @ W.T, dda @ W.T
            if k < len(case.weights) - 1:
                s, s1, s2 = act(z)
                a, da, dda = s, s1 * dz, s2 * dz * dz + s1 * ddz
            else:
                a, da, dda = z, dz, ddz
        jac[:, :, i] = da
        lap += dda
    return jac, lap


def reference_engine(quantity: str, case: Case) -> np.ndarray:
    """Exact NumPy forward-mode evaluation; conforms to every fixture it can run (``tanh`` and ``silu`` networks)."""
    jac, lap = _jacobian_and_laplacians(case)
    if quantity == "lap":
        return lap[:, 0]
    eye = np.eye(jac.shape[-1])
    if quantity == "divergence":
        return np.asarray(np.trace(jac, axis1=-2, axis2=-1))
    if quantity == "logdet":
        return np.asarray(np.linalg.slogdet(eye + jac)[1])
    if quantity == "solve":
        if case.rhs is None:
            raise ValueError(f"{case.name} has no right-hand side")
        return np.linalg.solve(eye + jac, case.rhs[..., None])[..., 0]
    raise ValueError(f"unknown quantity {quantity!r}")


# ------------------------------------------------------------------------------------------------- dispatch regret


def dispatch_regret(
    timings: Mapping[str, Mapping[str, float]],
    chosen: Mapping[str, str],
    *,
    median_limit: float = 1.10,
    worst_limit: float = 1.5,
) -> dict[str, Any]:
    """Time of the chosen program over the time of the best program, per case.

    ``timings[case][program]`` is the measured steady-state seconds of every feasible program for ``case`` (programs that
    cannot run are omitted, not given infinity) and ``chosen[case]`` the program the planner picked. The summary reports the median
    and worst regret and whether both are within the limits (the dispatch gate of ``docs/stability.md`` uses 1.10 and 1.5).
    """
    regrets: dict[str, float] = {}
    for case, by_program in timings.items():
        if not by_program:
            raise ValueError(f"{case}: no timed program")
        if chosen[case] not in by_program:
            raise ValueError(f"{case}: the chosen program {chosen[case]!r} was not timed")
        regrets[case] = by_program[chosen[case]] / min(by_program.values())
    values = sorted(regrets.values())
    median = (
        values[len(values) // 2]
        if len(values) % 2
        else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])
    )
    worst = values[-1] if values else 1.0
    return {
        "regret": regrets,
        "median": median,
        "worst": worst,
        "within_limits": bool(median <= median_limit and worst <= worst_limit),
        "median_limit": median_limit,
        "worst_limit": worst_limit,
    }
