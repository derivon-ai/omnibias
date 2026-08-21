# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite rational stencil obligations (theory 01-11).

The consistency conditions of a scale-free stencil are equalities of
rationals

    C_j = sum_{(i,p)} A_{i,p} c_i^{j-p} / (j-p)!  =  [j == q],

together with the reported leading moment ``C_N`` and (for a Birkhoff
support) the Polya comparisons plus a nonzero confluent-Vandermonde
determinant. Every quantity is a fraction of integers. Lean checks that
finite algebra; it does not see a collapse, a Taylor remainder, or any
statement quantified over a function class.

``theorem_prover_verified`` is earned only by a genuine kernel
``lake build``. This module never writes that flag into a certificate.
``mathlib_verified`` is a distinct tier and is never set here.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Any

from omnibias.core.multipack import MultiPackSpec, PackSpec, incidence_matrix
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate

PAYLOAD_STENCIL = "rational_stencil_consistency"
PAYLOAD_POISEDNESS = "rational_poisedness"

#: Largest absolute numerator/denominator the Lean emitter will accept.
#: Weaker than Lean ``Int`` (unbounded); it only bounds CI term size.
_INT_CAP = 10**18


def _as_frac(x: Fraction | int | str) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def _as_orders(raw: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    cleaned: list[tuple[int, ...]] = []
    for i, ords in enumerate(raw):
        seq = tuple(int(p) for p in ords)
        if not seq:
            raise ValueError(f"orders[{i}] must be non-empty")
        if any(p < 0 for p in seq):
            raise ValueError(f"orders[{i}] must be >= 0")
        if len(set(seq)) != len(seq):
            raise ValueError(f"orders[{i}] must be unique")
        cleaned.append(tuple(sorted(seq)))
    return tuple(cleaned)


def _rat_pair(q: Fraction) -> list[str]:
    return [str(q.numerator), str(q.denominator)]


def _moment(c: Fraction, j: int, p: int) -> Fraction:
    if j < p:
        return Fraction(0)
    k = j - p
    return (c**k) / Fraction(factorial(k))


def _vandermonde(
    nodes: Sequence[Fraction], pairs: Sequence[tuple[int, int]], n_rows: int
) -> list[list[Fraction]]:
    mat: list[list[Fraction]] = []
    for j in range(n_rows):
        row = [_moment(nodes[i], j, p) for i, p in pairs]
        mat.append(row)
    return mat


def _det_over_q(mat: Sequence[Sequence[Fraction]]) -> Fraction:
    n = len(mat)
    if n == 0:
        return Fraction(0)
    if any(len(row) != n for row in mat):
        raise ValueError("determinant requires a square matrix")
    a = [list(row) for row in mat]
    det = Fraction(1)
    for col in range(n):
        pivot = None
        best = Fraction(0)
        for r in range(col, n):
            mag = abs(a[r][col])
            if mag > best:
                best = mag
                pivot = r
        if pivot is None or a[pivot][col] == 0:
            return Fraction(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        pv = a[col][col]
        det *= pv
        for r in range(col + 1, n):
            if a[r][col] == 0:
                continue
            factor = a[r][col] / pv
            for j in range(col, n):
                a[r][j] -= factor * a[col][j]
    return det


def _mean_to_fraction(mean: float) -> Fraction:
    """Exact binary-float rational (integer / half means stay small)."""
    return Fraction(*float(mean).as_integer_ratio())


@dataclass(frozen=True)
class RationalStencil:
    """Scale-free stencil ``A_{i,p}`` at nodes ``c_i`` (units of ``h``).

    ``leading_coeff`` is the reported ``C_N``. Omit it to use the computed
    moment (self-check of the algebra, not of a foreign generator).
    """

    nodes: tuple[Fraction, ...]
    orders: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[Fraction, ...], ...]
    target_order: int
    leading_coeff: Fraction | None = None
    name: str = ""

    def __post_init__(self) -> None:
        nodes = tuple(_as_frac(c) for c in self.nodes)
        if not nodes:
            raise ValueError("RationalStencil requires at least one node")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be distinct")
        orders = _as_orders(self.orders)
        if len(nodes) != len(orders):
            raise ValueError("nodes and orders must have the same length")
        weights = tuple(tuple(_as_frac(a) for a in row) for row in self.weights)
        if len(weights) != len(nodes):
            raise ValueError("weights must match nodes")
        for i, (ords, row) in enumerate(zip(orders, weights, strict=True)):
            if len(ords) != len(row):
                raise ValueError(f"weights[{i}] must match orders[{i}]")
        q = int(self.target_order)
        if q < 0:
            raise ValueError(f"target_order must be >= 0, got {q}")
        lead = None if self.leading_coeff is None else _as_frac(self.leading_coeff)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "target_order", q)
        object.__setattr__(self, "leading_coeff", lead)
        object.__setattr__(self, "name", str(self.name))

    @property
    def n_conditions(self) -> int:
        return sum(len(o) for o in self.orders)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        out: list[tuple[int, int]] = []
        for i, ords in enumerate(self.orders):
            for p in ords:
                out.append((i, p))
        return tuple(out)

    def moment_sum(self, j: int) -> Fraction:
        acc = Fraction(0)
        for (i, p), a in zip(self.pairs(), self.flat_weights(), strict=True):
            acc += a * _moment(self.nodes[i], j, p)
        return acc

    def flat_weights(self) -> tuple[Fraction, ...]:
        return tuple(a for row in self.weights for a in row)

    def computed_leading(self) -> Fraction:
        return self.moment_sum(self.n_conditions)

    def reported_leading(self) -> Fraction:
        return self.leading_coeff if self.leading_coeff is not None else self.computed_leading()

    def with_flat_weights(self, flat: Sequence[Fraction | int | str]) -> RationalStencil:
        seq = tuple(_as_frac(a) for a in flat)
        if len(seq) != self.n_conditions:
            raise ValueError("flat weight length mismatch")
        packed: list[tuple[Fraction, ...]] = []
        k = 0
        for ords in self.orders:
            n = len(ords)
            packed.append(seq[k : k + n])
            k += n
        return RationalStencil(
            self.nodes,
            self.orders,
            tuple(packed),
            self.target_order,
            leading_coeff=self.leading_coeff,
            name=self.name,
        )

    def corrupt_first_weight(self, *, factor: Fraction | int = 2) -> RationalStencil:
        """Deliberately break consistency (G2 negative control)."""
        flat = list(self.flat_weights())
        if not flat:
            raise ValueError("no weights to corrupt")
        flat[0] = flat[0] * _as_frac(factor)
        return self.with_flat_weights(flat)


@dataclass(frozen=True)
class RationalSupport:
    """Birkhoff support (nodes + per-node orders) for exact poisedness."""

    nodes: tuple[Fraction, ...]
    orders: tuple[tuple[int, ...], ...]
    name: str = ""

    def __post_init__(self) -> None:
        nodes = tuple(_as_frac(c) for c in self.nodes)
        if not nodes:
            raise ValueError("RationalSupport requires at least one node")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be distinct")
        orders = _as_orders(self.orders)
        if len(nodes) != len(orders):
            raise ValueError("nodes and orders must have the same length")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "name", str(self.name))

    @property
    def n_conditions(self) -> int:
        return sum(len(o) for o in self.orders)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        out: list[tuple[int, int]] = []
        for i, ords in enumerate(self.orders):
            for p in ords:
                out.append((i, p))
        return tuple(out)

    @classmethod
    def from_stencil(cls, stencil: RationalStencil) -> RationalSupport:
        return cls(stencil.nodes, stencil.orders, name=stencil.name)

    @classmethod
    def from_multipack(cls, spec: MultiPackSpec, *, name: str = "") -> RationalSupport:
        means = spec.distinct_means
        e = incidence_matrix(spec)
        nodes = tuple(_mean_to_fraction(m) for m in means)
        orders = tuple(tuple(j for j, flag in enumerate(row) if flag) for row in e)
        return cls(nodes, orders, name=name)


@dataclass(frozen=True)
class Obligation:
    """A finite rational payload ready to seal. ``holds`` is Python algebra."""

    kind: str
    payload: dict[str, Any]
    holds: bool
    max_abs_int: int


@dataclass(frozen=True)
class StencilCertificateReport:
    """Sealed certificate plus the kernel / Mathlib flags for this spec.

    ``mathlib_verified`` is always ``False``: this path never touches the
    Mathlib-backed project.
    """

    certificate: dict[str, Any]
    obligation: Obligation
    theorem_prover_verified: bool
    mathlib_verified: bool
    lean: LeanCheckResult | None
    wall_seconds: float | None


def honesty_payload() -> dict[str, bool]:
    """Sealed honesty for the smoke artifact. Not a Lean pass."""
    return {
        "unproven_claim": False,
        "collapse_limit_in_lean": False,
        "taylor_theorem_in_lean": False,
        "function_class_in_lean": False,
        "mathlib_path_used": False,
        "navier_stokes_proof_claim": False,
    }


def _max_abs_from_pairs(pairs: Sequence[Sequence[str]]) -> int:
    acc = 1
    for pair in pairs:
        for s in pair:
            acc = max(acc, abs(int(s)))
    return acc


def stencil_consistency_obligation(stencil: RationalStencil) -> Obligation:
    """Finite conjunction of ``C_0 .. C_{N-1}`` plus the reported ``C_N``."""
    n = stencil.n_conditions
    q = stencil.target_order
    conditions: list[dict[str, Any]] = []
    holds = True
    rat_pairs: list[list[str]] = []
    for j in range(n):
        lhs = stencil.moment_sum(j)
        rhs = Fraction(1) if j == q else Fraction(0)
        if lhs != rhs:
            holds = False
        conditions.append({"j": j, "lhs": _rat_pair(lhs), "rhs": _rat_pair(rhs)})
        rat_pairs.extend([_rat_pair(lhs), _rat_pair(rhs)])
    lead_lhs = stencil.computed_leading()
    lead_rhs = stencil.reported_leading()
    if lead_lhs != lead_rhs:
        holds = False
    rat_pairs.extend([_rat_pair(lead_lhs), _rat_pair(lead_rhs)])
    payload: dict[str, Any] = {
        "type": PAYLOAD_STENCIL,
        "kind": PAYLOAD_STENCIL,
        "nodes": [_rat_pair(c) for c in stencil.nodes],
        "orders": [list(o) for o in stencil.orders],
        "weights": [[_rat_pair(a) for a in row] for row in stencil.weights],
        "target_order": q,
        "conditions": conditions,
        "leading_coeff": _rat_pair(lead_rhs),
        "leading_lhs": _rat_pair(lead_lhs),
        "name": stencil.name,
    }
    return Obligation(PAYLOAD_STENCIL, payload, holds, _max_abs_from_pairs(rat_pairs))


def _polya_pairs(support: RationalSupport) -> list[tuple[int, int]]:
    max_p = 0
    for ords in support.orders:
        if ords:
            max_p = max(max_p, max(ords))
    n_cols = max_p + 1
    counts = [0] * n_cols
    for ords in support.orders:
        for p in ords:
            counts[p] += 1
    out: list[tuple[int, int]] = []
    for j in range(n_cols):
        have = sum(counts[: j + 1])
        out.append((have, j + 1))
    return out


def poisedness_obligation(
    spec: RationalSupport | RationalStencil | MultiPackSpec,
) -> Obligation:
    """Polya inequalities plus a nonzero confluent Vandermonde determinant."""
    if isinstance(spec, MultiPackSpec):
        support = RationalSupport.from_multipack(spec)
    elif isinstance(spec, RationalStencil):
        support = RationalSupport.from_stencil(spec)
    else:
        support = spec
    n = support.n_conditions
    if n == 0:
        raise ValueError("poisedness requires at least one incidence pair")
    mat = _vandermonde(support.nodes, support.pairs(), n)
    det = _det_over_q(mat)
    polya = _polya_pairs(support)
    polya_ok = all(have >= need for have, need in polya)
    holds = polya_ok and det != 0
    payload: dict[str, Any] = {
        "type": PAYLOAD_POISEDNESS,
        "kind": PAYLOAD_POISEDNESS,
        "nodes": [_rat_pair(c) for c in support.nodes],
        "orders": [list(o) for o in support.orders],
        "polya": [{"j": j, "have": have, "need": need} for j, (have, need) in enumerate(polya)],
        "det": _rat_pair(det),
        "name": support.name,
    }
    max_abs = max(
        abs(det.numerator),
        abs(det.denominator),
        *(max(abs(h), abs(n_)) for h, n_ in polya),
        1,
    )
    return Obligation(PAYLOAD_POISEDNESS, payload, holds, max_abs)


def _seal_report(
    obligation: Obligation,
    *,
    claim: str,
    run_lean: bool,
) -> StencilCertificateReport:
    if obligation.max_abs_int > _INT_CAP:
        raise ValueError(
            f"obligation integers exceed cap {_INT_CAP}: {obligation.max_abs_int}"
        )
    cert = make_certificate(
        claim=claim,
        payload=obligation.payload,
        honesty={"unproven_claim": False},
    )
    lean: LeanCheckResult | None = None
    verified = False
    wall: float | None = None
    if run_lean:
        t0 = time.perf_counter()
        lean = check_certificate(cert)
        wall = time.perf_counter() - t0
        verified = bool(lean.verified)
    return StencilCertificateReport(
        certificate=cert,
        obligation=obligation,
        theorem_prover_verified=verified,
        mathlib_verified=False,
        lean=lean,
        wall_seconds=wall,
    )


def seal_stencil_certificate(
    stencil: RationalStencil,
    *,
    run_lean: bool = True,
) -> StencilCertificateReport:
    """Seal the v1 certificate and optionally drive the Lean kernel.

    ``theorem_prover_verified`` is set only on a genuine kernel pass.
    ``mathlib_verified`` stays ``False``.
    """
    obligation = stencil_consistency_obligation(stencil)
    return _seal_report(
        obligation,
        claim="stencil consistency conditions are rational identities",
        run_lean=run_lean,
    )


def seal_poisedness_certificate(
    spec: RationalSupport | RationalStencil | MultiPackSpec,
    *,
    run_lean: bool = True,
) -> StencilCertificateReport:
    """Seal Polya + nonzero-det poisedness. Same honesty rules as the stencil seal."""
    obligation = poisedness_obligation(spec)
    return _seal_report(
        obligation,
        claim="Birkhoff support is poised over the rationals",
        run_lean=run_lean,
    )


def _value_only(
    name: str,
    nodes: Sequence[Fraction | int | str],
    weights: Sequence[Fraction | int | str],
    target_order: int,
    *,
    leading: Fraction | int | str | None = None,
) -> RationalStencil:
    n = tuple(_as_frac(c) for c in nodes)
    w = tuple((_as_frac(a),) for a in weights)
    if len(n) != len(w):
        raise ValueError("value-only stencil: nodes and weights must match")
    orders = tuple((0,) for _ in n)
    lead = None if leading is None else _as_frac(leading)
    return RationalStencil(n, orders, w, target_order, leading_coeff=lead, name=name)


def curated_rational_stencils() -> tuple[RationalStencil, ...]:
    """Uniform, irregular, and Birkhoff fixtures (more than 20)."""
    return (
        _value_only("fwd1_2", (0, 1), (-1, 1), 1),
        _value_only("bwd1_2", (-1, 0), (-1, 1), 1),
        _value_only("ctr1_2", (-1, 1), (Fraction(-1, 2), Fraction(1, 2)), 1),
        _value_only("fwd1_3", (0, 1, 2), (Fraction(-3, 2), 2, Fraction(-1, 2)), 1),
        _value_only("bwd1_3", (-2, -1, 0), (Fraction(1, 2), -2, Fraction(3, 2)), 1),
        _value_only("ctr2_3", (-1, 0, 1), (1, -2, 1), 2),
        _value_only("fwd2_3", (0, 1, 2), (1, -2, 1), 2),
        _value_only("bwd2_3", (-2, -1, 0), (1, -2, 1), 2),
        _value_only(
            "ctr1_4",
            (-2, -1, 1, 2),
            (Fraction(1, 12), Fraction(-2, 3), Fraction(2, 3), Fraction(-1, 12)),
            1,
        ),
        _value_only(
            "fwd1_4",
            (0, 1, 2, 3),
            (Fraction(-11, 6), 3, Fraction(-3, 2), Fraction(1, 3)),
            1,
        ),
        _value_only(
            "ctr2_5",
            (-2, -1, 0, 1, 2),
            (Fraction(-1, 12), Fraction(4, 3), Fraction(-5, 2), Fraction(4, 3), Fraction(-1, 12)),
            2,
        ),
        _value_only("ctr4_5", (-2, -1, 0, 1, 2), (1, -4, 6, -4, 1), 4),
        _value_only("fwd3_4", (0, 1, 2, 3), (-1, 3, -3, 1), 3),
        _value_only("interp2", (-1, 1), (Fraction(1, 2), Fraction(1, 2)), 0),
        _value_only("interp3", (-1, 0, 1), (0, 1, 0), 0),
        RationalStencil(
            (Fraction(-1), Fraction(0), Fraction(1)),
            ((0,), (0,), (1,)),
            ((Fraction(-2, 3),), (Fraction(2, 3),), (Fraction(1, 3),)),
            1,
            leading_coeff=Fraction(5, 18),
            name="birkhoff_spec",
        ),
        _value_only(
            "irr_half",
            (Fraction(-1, 2), 0, 1),
            (Fraction(-4, 3), 1, Fraction(1, 3)),
            1,
        ),
        _value_only(
            "irr_spread",
            (-2, -1, 1, 3),
            (Fraction(1, 15), Fraction(-5, 8), Fraction(7, 12), Fraction(-1, 40)),
            1,
        ),
        RationalStencil(
            (Fraction(-1), Fraction(1)),
            ((0,), (0, 1)),
            ((Fraction(-1, 2),), (Fraction(1, 2), Fraction(0))),
            1,
            name="birk_right",
        ),
        _value_only(
            "fwd1_5",
            (0, 1, 2, 3, 4),
            (Fraction(-25, 12), 4, -3, Fraction(4, 3), Fraction(-1, 4)),
            1,
        ),
        _value_only("ctr3_4", (-2, -1, 1, 2), (Fraction(-1, 2), 1, -1, Fraction(1, 2)), 3),
        RationalStencil(
            (Fraction(0), Fraction(1)),
            ((0, 1), (0,)),
            ((Fraction(0), Fraction(1)), (Fraction(0),)),
            1,
            name="hermite_q1",
        ),
    )


def birkhoff_spec_stencil() -> RationalStencil:
    """Worked example from spec 01-11 / 01-04."""
    for st in curated_rational_stencils():
        if st.name == "birkhoff_spec":
            return st
    raise RuntimeError("birkhoff_spec missing from curated set")


def birkhoff_spec_support() -> RationalSupport:
    return RationalSupport.from_stencil(birkhoff_spec_stencil())


def birkhoff_spec_multipack() -> MultiPackSpec:
    return MultiPackSpec.from_packs(
        (
            PackSpec(order=0, mean=-1.0),
            PackSpec(order=0, mean=0.0),
            PackSpec(order=1, mean=1.0),
        )
    )


def payload_max_abs_int(payload: Mapping[str, Any]) -> int:
    """Largest absolute integer appearing in a sealed stencil / poisedness payload."""
    acc = 1
    conditions = payload.get("conditions")
    if isinstance(conditions, Sequence) and not isinstance(conditions, str | bytes):
        for row in conditions:
            if not isinstance(row, Mapping):
                continue
            for key in ("lhs", "rhs"):
                pair = row.get(key)
                if (
                    isinstance(pair, Sequence)
                    and not isinstance(pair, str | bytes)
                    and len(pair) == 2
                ):
                    acc = max(acc, abs(int(pair[0])), abs(int(pair[1])))
    for key in ("leading_coeff", "leading_lhs", "det"):
        pair = payload.get(key)
        if (
            isinstance(pair, Sequence)
            and not isinstance(pair, str | bytes)
            and len(pair) == 2
        ):
            acc = max(acc, abs(int(pair[0])), abs(int(pair[1])))
    return acc


def digest_is_valid(cert: Mapping[str, Any]) -> bool:
    return verify_certificate_digest(cert)


__all__ = [
    "Obligation",
    "PAYLOAD_POISEDNESS",
    "PAYLOAD_STENCIL",
    "RationalStencil",
    "RationalSupport",
    "StencilCertificateReport",
    "birkhoff_spec_multipack",
    "birkhoff_spec_stencil",
    "birkhoff_spec_support",
    "curated_rational_stencils",
    "digest_is_valid",
    "honesty_payload",
    "payload_max_abs_int",
    "poisedness_obligation",
    "seal_poisedness_certificate",
    "seal_stencil_certificate",
    "stencil_consistency_obligation",
]
