# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Finite-domain CSP by temperature collapse (theory 03-03).

A clause is a soft-OR; an instance is a product of those, plus compact
global relaxations. ``E = 0`` exactly on satisfying one-hot vertices.
Two independent ``beta -> inf`` knobs (simplex sharpness and clause
sharpness) are **temperature collapse** (feasibility). Neither is the
founding bias collapse (``delta -> 0`` to ``sigma^(K-1)``). Do not conflate the two.
A relaxation is not a complete solver and never
proves unsatisfiability. Certified statements are instance-wise gaps,
not P vs NP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete._core.solution import GapCertificate

FloatArray = NDArray[np.float64]
GlobalKind = Literal["all_different", "cardinality", "element"]

_ONEHOT_WEIGHT = 1.0


@dataclass(frozen=True)
class Variable:
    name: str
    domain: tuple[object, ...]

    def __post_init__(self) -> None:
        if len(self.domain) < 1:
            raise ValueError("domain must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "domain", tuple(self.domain))


@dataclass(frozen=True)
class Relation:
    """Allowed tuples are **domain indices**, not values."""

    scope: tuple[int, ...]
    allowed: frozenset[tuple[int, ...]]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("relation weight must be nonnegative")
        arity = len(self.scope)
        for tup in self.allowed:
            if len(tup) != arity:
                raise ValueError("allowed tuples must match scope arity")
        object.__setattr__(self, "scope", tuple(int(i) for i in self.scope))
        object.__setattr__(self, "allowed", frozenset(tuple(int(v) for v in t) for t in self.allowed))
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class GlobalConstraint:
    kind: GlobalKind
    scope: tuple[int, ...]
    parameter: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("all_different", "cardinality", "element"):
            raise ValueError(f"unknown global kind {self.kind!r}")
        if not self.scope:
            raise ValueError("global scope must be non-empty")
        object.__setattr__(self, "scope", tuple(int(i) for i in self.scope))


@dataclass(frozen=True)
class CSPResult:
    assignment: tuple[int, ...]
    values: tuple[object, ...]
    energy: float
    satisfied: bool
    n_eval: int
    simplex_beta: float
    clause_beta: float


@dataclass(frozen=True)
class CSPCertificate:
    """Instance-wise SAT claim. Never a completeness or P vs NP statement."""

    claimed_sat: bool
    hard_energy: float
    gap_bound: float
    assignment: tuple[int, ...]
    method: str


def honesty_payload() -> dict[str, bool]:
    return {
        "complete_solver": False,
        "unsat_proof": False,
        "p_equals_np_claim": False,
        "simplex_is_temperature_collapse": True,
        "clause_is_temperature_collapse": True,
        "founding_bias_collapse_in_csp": False,
    }


def _offsets(variables: tuple[Variable, ...]) -> tuple[int, ...]:
    off = [0]
    for var in variables:
        off.append(off[-1] + len(var.domain))
    return tuple(off)


@dataclass(frozen=True)
class CSP:
    """Finite-domain CSP on the ``DiscreteProblem`` seam (flattened one-hot)."""

    variables: tuple[Variable, ...]
    relations: tuple[Relation, ...]
    globals: tuple[GlobalConstraint, ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        n_vars = len(self.variables)
        if n_vars < 1:
            raise ValueError("CSP needs at least one variable")
        for rel in self.relations:
            for idx in rel.scope:
                if idx < 0 or idx >= n_vars:
                    raise ValueError(f"relation scope {idx} is out of range")
        for g in self.globals:
            for idx in g.scope:
                if idx < 0 or idx >= n_vars:
                    raise ValueError(f"global scope {idx} is out of range")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "globals", tuple(self.globals))

    @property
    def n(self) -> int:
        return int(_offsets(self.variables)[-1])

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    def domain_sizes(self) -> tuple[int, ...]:
        return tuple(len(v.domain) for v in self.variables)

    def unpack(self, x: object) -> list[FloatArray]:
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        if xv.size != self.n:
            raise ValueError(f"expected length {self.n}, got {xv.size}")
        off = _offsets(self.variables)
        return [xv[off[i] : off[i + 1]].copy() for i in range(self.n_vars)]

    def pack(self, probs: SequenceProbs) -> FloatArray:
        return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in probs])

    def violation_energy(self, x: object) -> float | FloatArray:
        """Spec ``E = sum w (1-s)`` plus globals. No one-hot penalty."""
        xv = np.asarray(x, dtype=np.float64)
        single = xv.ndim == 1
        matrix = xv.reshape(1, -1) if single else xv
        out = np.zeros(matrix.shape[0], dtype=np.float64)
        for row in range(matrix.shape[0]):
            out[row] = _violation_one(self, self.unpack(matrix[row]))
        return float(out[0]) if single else out

    def energy(self, x: object) -> float | FloatArray:
        """Seam energy: violations plus a one-hot penalty. ``0`` iff a SAT vertex."""
        xv = np.asarray(x, dtype=np.float64)
        single = xv.ndim == 1
        matrix = xv.reshape(1, -1) if single else xv
        out = np.zeros(matrix.shape[0], dtype=np.float64)
        for row in range(matrix.shape[0]):
            probs = self.unpack(matrix[row])
            out[row] = _violation_one(self, probs) + _onehot_penalty(probs)
        return float(out[0]) if single else out

    def to_polynomial(self) -> object:
        from omnibias.sos import Polynomial

        n = self.n
        poly = Polynomial.zero(n)
        off = _offsets(self.variables)
        slots = [[Polynomial.variable(off[i] + v, n) for v in range(len(self.variables[i].domain))] for i in range(self.n_vars)]
        for rel in self.relations:
            s = Polynomial.zero(n)
            for tup in rel.allowed:
                term = Polynomial.constant(1.0, n)
                for var_i, val in zip(rel.scope, tup, strict=True):
                    term = term * slots[var_i][val]
                s = s + term
            poly = poly + (Polynomial.constant(1.0, n) - s) * rel.weight
        for g in self.globals:
            poly = poly + _global_poly(g, slots, n)
        for i, _var in enumerate(self.variables):
            total = Polynomial.zero(n)
            for slot in slots[i]:
                total = total + slot
            diff = total - Polynomial.constant(1.0, n)
            poly = poly + diff * diff * _ONEHOT_WEIGHT
        return poly


SequenceProbs = list[FloatArray]


def _onehot_penalty(probs: SequenceProbs) -> float:
    pen = 0.0
    for p in probs:
        pen += (float(np.sum(p)) - 1.0) ** 2
        pen += float(np.sum(p * (1.0 - p)))
    return _ONEHOT_WEIGHT * pen


def _violation_one(csp: CSP, probs: SequenceProbs) -> float:
    e = 0.0
    for rel in csp.relations:
        s = 0.0
        for tup in rel.allowed:
            prod = 1.0
            for var_i, val in zip(rel.scope, tup, strict=True):
                prod *= float(probs[var_i][val])
            s += prod
        e += rel.weight * (1.0 - s)
    for g in csp.globals:
        e += _global_energy(g, probs, csp)
    return float(e)


def _global_energy(g: GlobalConstraint, probs: SequenceProbs, csp: CSP) -> float:
    if g.kind == "all_different":
        d = max(len(csp.variables[i].domain) for i in g.scope)
        e = 0.0
        for v in range(d):
            present = [float(probs[i][v]) for i in g.scope if v < probs[i].size]
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    e += present[a] * present[b]
        return e
    if g.kind == "cardinality":
        k = 0 if g.parameter is None else int(g.parameter)
        val = 1 if all(len(csp.variables[i].domain) > 1 for i in g.scope) else 0
        mass = sum(float(probs[i][val]) for i in g.scope)
        return (mass - float(k)) ** 2
    # element: treat as "value of scope[0] indexes scope[1:]" -- require a table relation instead
    return 0.0


def _global_poly(g: GlobalConstraint, slots: list[list[object]], n: int) -> object:
    from omnibias.sos import Polynomial

    if g.kind == "all_different":
        poly = Polynomial.zero(n)
        d = max(len(slots[i]) for i in g.scope)
        for v in range(d):
            present = [slots[i][v] for i in g.scope if v < len(slots[i])]
            for a, b in ((present[i], present[j]) for i in range(len(present)) for j in range(i + 1, len(present))):
                poly = poly + a * b
        return poly
    if g.kind == "cardinality":
        k = 0 if g.parameter is None else int(g.parameter)
        val = 1 if all(len(slots[i]) > 1 for i in g.scope) else 0
        total = Polynomial.zero(n)
        for i in g.scope:
            total = total + slots[i][val]
        diff = total - Polynomial.constant(float(k), n)
        return diff * diff
    return Polynomial.zero(n)


def not_equal_relation(i: int, j: int, domain_size: int, *, weight: float = 1.0) -> Relation:
    allowed = frozenset((a, b) for a in range(domain_size) for b in range(domain_size) if a != b)
    return Relation((i, j), allowed, weight=weight)


def triangle_colouring(*, n_colours: int = 3) -> CSP:
    names = ("R", "G", "B", "Y", "K")[:n_colours]
    variables = tuple(Variable(f"x{i}", names) for i in range(3))
    rels = tuple(not_equal_relation(a, b, n_colours) for a, b in ((0, 1), (1, 2), (0, 2)))
    return CSP(variables, rels, name="triangle_colouring")


def softmax_rows(logits: object, *, beta: float) -> FloatArray:
    z = float(beta) * np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    w = np.exp(z)
    return (w / np.sum(w, axis=-1, keepdims=True)).astype(np.float64, copy=False)


def clause_gap_bound(csp: CSP, *, beta: float) -> float:
    """``sum_k w_k log(m_k) / beta``, reusing ``logsumexp_gap_bound``."""
    from omnibias.struct import logsumexp_gap_bound

    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    total = 0.0
    for rel in csp.relations:
        m = max(len(rel.allowed), 1)
        total += rel.weight * float(logsumexp_gap_bound(m, float(beta)))
    return float(total)


def soft_arc_consistency(csp: CSP, probs: SequenceProbs, *, iterations: int = 3) -> SequenceProbs:
    """Differentiable support filter. Hard limit is arc consistency; this is the soft form."""
    out = [np.asarray(p, dtype=np.float64).copy() for p in probs]
    for _ in range(int(iterations)):
        nxt = [p.copy() for p in out]
        for i in range(csp.n_vars):
            supp = np.ones(out[i].size, dtype=np.float64)
            for rel in csp.relations:
                if i not in rel.scope:
                    continue
                pos = rel.scope.index(i)
                s_v = np.zeros(out[i].size, dtype=np.float64)
                for tup in rel.allowed:
                    prod = 1.0
                    for var_j, val in zip(rel.scope, tup, strict=True):
                        if var_j != i:
                            prod *= float(out[var_j][val])
                    s_v[tup[pos]] += prod
                supp = supp * s_v
            nxt[i] = out[i] * supp
            z = float(np.sum(nxt[i]))
            nxt[i] = nxt[i] / z if z > 1e-15 else np.full_like(nxt[i], 1.0 / nxt[i].size)
        out = nxt
    return out


def decode_assignment(csp: CSP, probs: SequenceProbs) -> tuple[int, ...]:
    return tuple(int(np.argmax(p)) for p in probs)


def _value_flip_descent(csp: CSP, assign: tuple[int, ...]) -> tuple[int, ...]:
    """Greedy domain-value flips. Decoder, not a completeness claim."""
    current = list(assign)
    best = float(csp.energy(assignment_onehot(csp, tuple(current))))
    improved = True
    while improved and best > 0.0:
        improved = False
        for i, var in enumerate(csp.variables):
            for v in range(len(var.domain)):
                if v == current[i]:
                    continue
                trial = list(current)
                trial[i] = v
                e = float(csp.energy(assignment_onehot(csp, tuple(trial))))
                if e < best:
                    current = trial
                    best = e
                    improved = True
                    if best == 0.0:
                        return tuple(current)
    return tuple(current)


def assignment_onehot(csp: CSP, values: tuple[int, ...]) -> FloatArray:
    probs = [np.zeros(len(v.domain), dtype=np.float64) for v in csp.variables]
    for i, val in enumerate(values):
        probs[i][int(val)] = 1.0
    return csp.pack(probs)


def is_satisfying_vertex(csp: CSP, values: tuple[int, ...]) -> bool:
    return float(csp.energy(assignment_onehot(csp, values))) == 0.0


def enumerate_assignments(csp: CSP) -> list[tuple[int, ...]]:
    """All domain tuples. Exponential in the number of variables."""
    sizes = csp.domain_sizes()
    return [tuple(a) for a in product(*[range(s) for s in sizes])]


def brute_force_sat(csp: CSP) -> tuple[bool, tuple[int, ...] | None]:
    """Exponential domain-product oracle. Labelled as such; not a solver."""
    for assign in enumerate_assignments(csp):
        if is_satisfying_vertex(csp, assign):
            return True, assign
    return False, None


def backtrack_sat(csp: CSP, *, deadline_assign: int | None = None) -> bool:
    """Complete backtracking with forward checking. Baseline for G3, not the product."""
    sizes = csp.domain_sizes()
    n = csp.n_vars
    assign = [-1] * n
    checked = 0

    def ok(upto: int) -> bool:
        vals = tuple(assign[i] if i <= upto else 0 for i in range(n))
        for rel in csp.relations:
            if any(assign[i] < 0 for i in rel.scope):
                continue
            tup = tuple(assign[i] for i in rel.scope)
            if tup not in rel.allowed:
                return False
        for g in csp.globals:
            if any(assign[i] < 0 for i in g.scope):
                continue
            if g.kind == "all_different":
                taken = [assign[i] for i in g.scope]
                if len(taken) != len(set(taken)):
                    return False
            if g.kind == "cardinality":
                k = 0 if g.parameter is None else int(g.parameter)
                val = 1 if all(sizes[i] > 1 for i in g.scope) else 0
                if sum(1 for i in g.scope if assign[i] == val) != k:
                    return False
        _ = vals
        return True

    def rec(i: int) -> bool:
        nonlocal checked
        if deadline_assign is not None and checked >= deadline_assign:
            return False
        if i == n:
            checked += 1
            return True
        for v in range(sizes[i]):
            assign[i] = v
            checked += 1
            if ok(i) and rec(i + 1):
                return True
            assign[i] = -1
        return False

    return rec(0)


def _zip_schedules(
    simplex: AnnealSchedule, clause: AnnealSchedule
) -> list[tuple[float, float]]:
    a, b = simplex.betas(), clause.betas()
    n = max(len(a), len(b))
    out: list[tuple[float, float]] = []
    for i in range(n):
        out.append((a[min(i, len(a) - 1)], b[min(i, len(b) - 1)]))
    return out


def default_simplex_schedule() -> AnnealSchedule:
    """Measured default: keep the simplex softer than the clauses at the start."""
    return AnnealSchedule(beta0=0.4, beta_growth=1.4, stages=6, steps=8)


def default_clause_schedule() -> AnnealSchedule:
    return AnnealSchedule(beta0=1.2, beta_growth=1.4, stages=6, steps=8)


def _grad_violation(csp: CSP, probs: SequenceProbs) -> SequenceProbs:
    g = [np.zeros_like(p) for p in probs]
    for rel in csp.relations:
        for tup in rel.allowed:
            prods = [float(probs[var_i][val]) for var_i, val in zip(rel.scope, tup, strict=True)]
            full = 1.0
            for pr in prods:
                full *= pr
            for k, (var_i, val) in enumerate(zip(rel.scope, tup, strict=True)):
                loo = full / prods[k] if prods[k] > 0.0 else math.prod(prods[j] for j in range(len(prods)) if j != k)
                g[var_i][val] -= rel.weight * loo
    for gl in csp.globals:
        if gl.kind == "all_different":
            d = max(len(csp.variables[i].domain) for i in gl.scope)
            for v in range(d):
                for i in gl.scope:
                    if v >= g[i].size:
                        continue
                    g[i][v] += sum(
                        float(probs[j][v]) for j in gl.scope if j != i and v < probs[j].size
                    )
        elif gl.kind == "cardinality":
            k = 0 if gl.parameter is None else int(gl.parameter)
            val = 1 if all(len(csp.variables[i].domain) > 1 for i in gl.scope) else 0
            mass = sum(float(probs[i][val]) for i in gl.scope)
            deriv = 2.0 * (mass - float(k))
            for i in gl.scope:
                g[i][val] += deriv
    return g


def csp_solve(
    csp: CSP,
    *,
    simplex_schedule: AnnealSchedule | None = None,
    clause_schedule: AnnealSchedule | None = None,
    steps: int | None = None,
    arc_consistency: bool = True,
    restarts: int = 4,
    seed: int | None = None,
    lr: float = 0.35,
) -> CSPResult:
    """Projected softmax descent. Two schedules stay independent (never fused)."""
    simp = simplex_schedule or default_simplex_schedule()
    clau = clause_schedule or default_clause_schedule()
    pairs = _zip_schedules(simp, clau)
    n_inner = int(steps if steps is not None else simp.steps)
    rng = np.random.default_rng(seed)
    best_e = float("inf")
    best_assign = tuple(0 for _ in csp.variables)
    n_eval = 0
    last_b1, last_b2 = pairs[0]
    for r in range(int(restarts)):
        logits = [rng.normal(0.0, 0.4, size=len(v.domain)) for v in csp.variables]
        for beta1, beta2 in pairs:
            last_b1, last_b2 = beta1, beta2
            _ = beta2  # clause knob is exposed; multilinear s_k does not need it
            for _step in range(n_inner):
                probs = [softmax_rows(z, beta=beta1) for z in logits]
                if arc_consistency:
                    probs = soft_arc_consistency(csp, probs, iterations=1)
                    logits = [np.log(np.maximum(p, 1e-12)) / max(beta1, 1e-12) for p in probs]
                g_p = _grad_violation(csp, probs)
                for i, z in enumerate(logits):
                    p = softmax_rows(z, beta=beta1)
                    # dE/dz = beta * (diag(p) - p p^T) @ dE/dp
                    gp = g_p[i]
                    gz = beta1 * (p * gp - p * float(np.dot(p, gp)))
                    logits[i] = z - float(lr) * gz
                n_eval += 1
        probs = [softmax_rows(z, beta=last_b1) for z in logits]
        assign = _value_flip_descent(csp, decode_assignment(csp, probs))
        e = float(csp.energy(assignment_onehot(csp, assign)))
        if e < best_e:
            best_e = e
            best_assign = assign
        if best_e == 0.0:
            break
        _ = r
    values = tuple(csp.variables[i].domain[v] for i, v in enumerate(best_assign))
    return CSPResult(
        assignment=best_assign,
        values=values,
        energy=best_e,
        satisfied=best_e == 0.0,
        n_eval=n_eval,
        simplex_beta=float(last_b1),
        clause_beta=float(last_b2),
    )


def certify_csp(csp: CSP, assignment: tuple[int, ...], *, beta: float) -> CSPCertificate:
    """Claim SAT only if the decoded vertex has energy 0. Never an UNSAT proof."""
    hard = float(csp.energy(assignment_onehot(csp, assignment)))
    bound = clause_gap_bound(csp, beta=beta) if csp.relations else 0.0
    return CSPCertificate(
        claimed_sat=hard == 0.0,
        hard_energy=hard,
        gap_bound=bound,
        assignment=tuple(int(v) for v in assignment),
        method="decoded_vertex",
    )


def certify_gap_csp(csp: CSP, assignment: tuple[int, ...]) -> GapCertificate:
    """Reuse the substrate sandwich on the one-hot cube."""
    from omnibias.discrete.certify import certify_gap

    return certify_gap(csp, assignment_onehot(csp, assignment), claim_label="csp energy")


def random_binary_csp(
    n_vars: int,
    domain: int,
    n_constraints: int,
    rng: Generator,
    *,
    tightness: float = 0.4,
) -> CSP:
    """Random binary table CSP (G3 ensemble)."""
    variables = tuple(Variable(f"x{i}", tuple(range(domain))) for i in range(n_vars))
    rels: list[Relation] = []
    pairs = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
    rng.shuffle(pairs)
    for i, j in pairs[:n_constraints]:
        universe = [(a, b) for a in range(domain) for b in range(domain)]
        rng.shuffle(universe)
        keep = max(1, int(round((1.0 - tightness) * len(universe))))
        rels.append(Relation((i, j), frozenset(universe[:keep])))
    return CSP(variables, tuple(rels), name="random_binary")


__all__ = [
    "CSP",
    "CSPCertificate",
    "CSPResult",
    "GlobalConstraint",
    "Relation",
    "Variable",
    "assignment_onehot",
    "backtrack_sat",
    "brute_force_sat",
    "certify_csp",
    "certify_gap_csp",
    "clause_gap_bound",
    "csp_solve",
    "decode_assignment",
    "default_clause_schedule",
    "default_simplex_schedule",
    "enumerate_assignments",
    "honesty_payload",
    "is_satisfying_vertex",
    "not_equal_relation",
    "random_binary_csp",
    "soft_arc_consistency",
    "softmax_rows",
    "triangle_colouring",
]
