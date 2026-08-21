# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Shared observation, class ranking, and frequency memory.

A tagged :class:`Observation` is the one payload every condition-sort binder
views. Ranking is a total order on *certified* hits. Soft RMSE never outranks
an exact identity. :class:`ClassMemory` / :class:`FrequencyGate` only propose
an order — they never write :class:`~omnibias.core.proof.discovery.ExactCheck`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any, Literal, Protocol

from omnibias.core.proof.discovery import (
    Candidate,
    ExactCheck,
    Statement,
)
from omnibias.core.proof.lift import as_fraction, integer_null_space

ClassTier = Literal["exact", "enclosure", "empirical"]

_TIER_RANK: dict[ClassTier, int] = {"exact": 0, "enclosure": 1, "empirical": 2}

_DEFAULT_SORTS: tuple[str, ...] = (
    "conservation",
    "dfinite",
    "edge_colouring",
    "extremal_template",
    "forbidden_minor",
    "fractional_order",
    "jet_monomial",
    "ore",
    "pde_operator",
    "piecewise_hybrid",
    "residual_sign",
    "sos_onset",
    "sos_template",
)

FEATURE_DIM = 16
_SECOND_DERIV = frozenset({"u_xx", "u_yy", "lap", "laplacian"})
PolyTerm = tuple[tuple[int, ...], str]
PolyTerms = tuple[PolyTerm, ...]


def _parse_poly_terms(raw: Any) -> PolyTerms:
    terms: list[PolyTerm] = []
    for item in raw or ():
        exp, coeff = item[0], item[1]
        terms.append((tuple(int(power) for power in exp), str(coeff)))
    return tuple(terms)


def _poly_degree_estimate(terms: PolyTerms) -> int:
    if not terms:
        return 0
    return max(sum(int(power) for power in exp) for exp, _coeff in terms)


def _sign_change_count(observation: Observation) -> int:
    values: list[Fraction] = []
    if observation.sample_x:
        try:
            values = [as_fraction(Fraction(item)) for item in observation.sample_x]
        except (ValueError, ZeroDivisionError, TypeError):
            values = []
    elif "yp" in observation.jet_names and observation.jet_rows:
        index = observation.jet_names.index("yp")
        try:
            values = [as_fraction(Fraction(row[index])) for row in observation.jet_rows]
        except (ValueError, ZeroDivisionError, TypeError, IndexError):
            values = []
    count = 0
    prev: int | None = None
    for value in values:
        if value == 0:
            continue
        sign = 1 if value > 0 else -1
        if prev is not None and sign != prev:
            count += 1
        prev = sign
    return count


@dataclass(frozen=True)
class Observation:
    """Tagged, JSON-able payload. Values are fraction strings or ints. No numpy."""

    tag: str = ""
    sequence: tuple[str, ...] = ()
    jet_names: tuple[str, ...] = ()
    jet_rows: tuple[tuple[str, ...], ...] = ()
    design: tuple[tuple[str, ...], ...] = ()
    target: tuple[str, ...] = ()
    term_names: tuple[str, ...] = ()
    graph_n: int = 0
    graph_edges: tuple[tuple[int, int], ...] = ()
    poly_n_vars: int = 0
    poly_terms: PolyTerms = ()
    sample_x: tuple[str, ...] = ()
    poly_constraints: tuple[PolyTerms, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

    def extra_map(self) -> dict[str, str]:
        return dict(self.extra)

    def features(self) -> tuple[int, ...]:
        """Length-:data:`FEATURE_DIM` int tuple for memory / gate keys."""

        names = self.term_names + self.jet_names
        has_second = 1 if any(name in _SECOND_DERIV for name in names) else 0
        has_cont = 1 if ("rho_t" in names and "j_x" in names) else 0
        return (
            1 if self.sequence else 0,
            1 if self.jet_rows or self.jet_names else 0,
            1 if self.design or self.target else 0,
            1 if self.graph_edges or self.graph_n else 0,
            1 if self.poly_terms or self.poly_n_vars else 0,
            len(self.sequence),
            len(self.jet_rows),
            len(self.design),
            int(self.graph_n),
            int(self.poly_n_vars),
            has_second,
            has_cont,
            _sign_change_count(self),
            _poly_degree_estimate(self.poly_terms),
            len(self.poly_constraints),
            1 if self.sample_x else 0,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "sequence": list(self.sequence),
            "jet_names": list(self.jet_names),
            "jet_rows": [list(row) for row in self.jet_rows],
            "design": [list(row) for row in self.design],
            "target": list(self.target),
            "term_names": list(self.term_names),
            "graph_n": self.graph_n,
            "graph_edges": [list(edge) for edge in self.graph_edges],
            "poly_n_vars": self.poly_n_vars,
            "poly_terms": [[list(exp), coeff] for exp, coeff in self.poly_terms],
            "sample_x": list(self.sample_x),
            "poly_constraints": [
                [[list(exp), coeff] for exp, coeff in poly]
                for poly in self.poly_constraints
            ],
            "extra": dict(self.extra),
            "features": list(self.features()),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Observation:
        extra_raw = data.get("extra", ())
        if isinstance(extra_raw, Mapping):
            extra = tuple((str(key), str(value)) for key, value in extra_raw.items())
        else:
            extra = tuple((str(key), str(value)) for key, value in extra_raw)
        constraints_raw = data.get("poly_constraints", ())
        return cls(
            tag=str(data.get("tag", "")),
            sequence=tuple(str(item) for item in data.get("sequence", ())),
            jet_names=tuple(str(item) for item in data.get("jet_names", ())),
            jet_rows=tuple(
                tuple(str(item) for item in row) for row in data.get("jet_rows", ())
            ),
            design=tuple(
                tuple(str(item) for item in row) for row in data.get("design", ())
            ),
            target=tuple(str(item) for item in data.get("target", ())),
            term_names=tuple(str(item) for item in data.get("term_names", ())),
            graph_n=int(data.get("graph_n", 0)),
            graph_edges=tuple(
                (int(edge[0]), int(edge[1])) for edge in data.get("graph_edges", ())
            ),
            poly_n_vars=int(data.get("poly_n_vars", 0)),
            poly_terms=_parse_poly_terms(data.get("poly_terms", ())),
            sample_x=tuple(str(item) for item in data.get("sample_x", ())),
            poly_constraints=tuple(_parse_poly_terms(poly) for poly in constraints_raw),
            extra=extra,
        )


def fractions_to_ints(
    rows: Sequence[Sequence[Fraction]],
) -> tuple[tuple[int, ...], ...]:
    """Clear a common denominator so an exact-``Q`` table becomes integers."""

    denoms = [value.denominator for row in rows for value in row]
    scale = 1
    for denom in denoms:
        scale = scale * denom // gcd(scale, denom)
    return tuple(tuple(int(value * scale) for value in row) for row in rows)


@dataclass(frozen=True)
class ClassHit:
    """One certified (or empirical) class hit. Ranking ignores RMSE when exact exists."""

    sort: str
    check: ExactCheck
    tier: ClassTier
    tokens: int
    degree: int
    unique_in_family: bool = False
    rmse: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sort": self.sort,
            "tier": self.tier,
            "tokens": self.tokens,
            "degree": self.degree,
            "unique_in_family": self.unique_in_family,
            "rmse": self.rmse,
            "ok": self.check.ok,
        }


@dataclass(frozen=True)
class ClassSelection:
    """Ranked class outcome. ``best`` is the total-order winner, not visit order."""

    observation: Observation
    hits: tuple[ClassHit, ...]
    best: ClassHit | None
    honesty: Mapping[str, bool]
    search_incomplete: bool
    grammar_complete: bool = False
    result: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "best": None if self.best is None else self.best.as_dict(),
            "hits": [hit.as_dict() for hit in self.hits],
            "honesty": dict(self.honesty),
            "search_incomplete": self.search_incomplete,
            "grammar_complete": self.grammar_complete,
            "observation": self.observation.as_dict(),
        }


def _walk_maps(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield payload
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        yield nested
    sub = payload.get("sub")
    if isinstance(sub, Mapping):
        yield sub
        inner = sub.get("payload")
        if isinstance(inner, Mapping):
            yield inner


def _tier_of(check: ExactCheck) -> ClassTier:
    raw_tier = check.payload.get("tier")
    if raw_tier == "exact":
        return "exact"
    if raw_tier == "enclosure":
        return "enclosure"
    if raw_tier == "empirical":
        return "empirical"
    if check.payload.get("mode") == "empirical":
        return "empirical"
    for mapping in _walk_maps(check.payload):
        if mapping.get("tier") == "enclosure" or mapping.get("status") == "proved":
            return "enclosure"
        if mapping.get("tier") == "empirical" or mapping.get("mode") == "empirical":
            return "empirical"
    if check.ok:
        return "exact"
    return "empirical"


def _tokens_of(check: ExactCheck) -> int:
    for mapping in _walk_maps(check.payload):
        hyp = mapping.get("hypothesis")
        if isinstance(hyp, Mapping):
            tokens = hyp.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
        mask = mapping.get("mask")
        if isinstance(mask, int) and mask > 0:
            return int(mask).bit_count()
        equation = mapping.get("equation")
        if isinstance(equation, Mapping):
            coeffs = equation.get("coefficients")
            if isinstance(coeffs, (list, tuple)):
                nonzero = [item for item in coeffs if str(item) not in {"0", "0/1"}]
                if nonzero:
                    return len(nonzero)
    tokens = check.payload.get("tokens")
    if isinstance(tokens, int):
        return tokens
    return 1


def _degree_of(check: ExactCheck) -> int:
    for mapping in _walk_maps(check.payload):
        for key in ("order", "index_degree", "half_degree"):
            raw = mapping.get(key)
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str) and raw.lstrip("-").isdigit():
                return int(raw)
        hyp = mapping.get("hypothesis")
        if isinstance(hyp, Mapping):
            coeffs = hyp.get("coefficients")
            if isinstance(coeffs, list) and coeffs:
                text = str(coeffs[0]).split("/", 1)[0]
                if text.lstrip("-").isdigit():
                    return int(text)
    return 0


def _unique_of(check: ExactCheck) -> bool:
    for mapping in _walk_maps(check.payload):
        characterization = mapping.get("characterization")
        if isinstance(characterization, Mapping) and bool(
            characterization.get("unique_in_family")
        ):
            return True
    return bool(check.payload.get("unique_in_family"))


def _rmse_of(check: ExactCheck) -> str | None:
    raw = check.payload.get("rmse")
    return str(raw) if raw is not None else None


def class_hit_from_check(
    sort: str,
    check: ExactCheck,
    *,
    unique_in_family: bool = False,
) -> ClassHit:
    """Build a :class:`ClassHit` from an exact-check payload."""

    return ClassHit(
        sort=sort,
        check=check,
        tier=_tier_of(check),
        tokens=_tokens_of(check),
        degree=_degree_of(check),
        unique_in_family=unique_in_family or _unique_of(check),
        rmse=_rmse_of(check),
    )


def rank_class_hits(hits: Sequence[ClassHit]) -> list[ClassHit]:
    """Total order on certified hits. Exact beats enclosure beats empirical.

    RMSE is ignored whenever any exact hit exists. Empirical hits are dropped
    from the returned list in that case so they can never be ``best``.
    """

    has_exact = any(hit.tier == "exact" for hit in hits)
    pool = [hit for hit in hits if hit.tier != "empirical"] if has_exact else list(hits)
    return sorted(
        pool,
        key=lambda hit: (
            _TIER_RANK[hit.tier],
            hit.tokens,
            hit.degree,
            0 if hit.unique_in_family else 1,
            hit.sort,
        ),
    )


class ClassGate(Protocol):
    """Proposer over sorts. Must not write :class:`ExactCheck`."""

    def propose(self, observation: Observation) -> Sequence[str]: ...


class ClassMemory:
    """In-memory ``features() → counts[sort]``. Propose only; never a checker."""

    def __init__(self) -> None:
        self._counts: dict[tuple[int, ...], dict[str, int]] = {}

    def record(self, observation: Observation, sort: str) -> None:
        bucket = self._counts.setdefault(observation.features(), {})
        bucket[sort] = bucket.get(sort, 0) + 1

    def counts_for(self, observation: Observation) -> dict[str, int]:
        return dict(self._counts.get(observation.features(), {}))

    def propose(
        self,
        observation: Observation,
        sorts: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        names = tuple(sorts) if sorts is not None else _DEFAULT_SORTS
        counts = self._counts.get(observation.features(), {})
        return tuple(sorted(names, key=lambda name: (-counts.get(name, 0), name)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": [
                {"features": list(key), "sorts": dict(value)}
                for key, value in self._counts.items()
            ]
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ClassMemory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls()
        for row in data.get("counts", ()):
            if not isinstance(row, Mapping):
                continue
            key = tuple(int(item) for item in row.get("features", ()))
            sorts = row.get("sorts", {})
            if isinstance(sorts, Mapping):
                memory._counts[key] = {str(name): int(count) for name, count in sorts.items()}
        return memory


@dataclass
class FrequencyGate:
    """Frequency proposer. Thin wrapper around :class:`ClassMemory.propose`."""

    memory: ClassMemory = field(default_factory=ClassMemory)

    def propose(self, observation: Observation) -> tuple[str, ...]:
        return self.memory.propose(observation)


@dataclass
class LinearSpanFamily:
    """Bitmask family over an exact-``Q`` design. Nullity-one is the accept gate."""

    design: tuple[tuple[int, ...], ...]
    term_names: tuple[str, ...]
    sort: str = "jet_monomial"
    target: tuple[int, ...] | None = None
    name: str = "linear_span"
    complete: bool = False
    empty_miss_detail: str = "no witness in enumerated grammar"
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="linear_span",
            obligation="a unique linear relation in the supplied exact-Q columns",
            parent="observation class loop",
            parent_status="already_true",
        )
    )

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        sort: str,
    ) -> LinearSpanFamily | None:
        if observation.jet_rows and observation.jet_names:
            rows = tuple(
                tuple(as_fraction(Fraction(item)) for item in row)
                for row in observation.jet_rows
            )
            names = observation.jet_names
            target_frac: tuple[Fraction, ...] | None = None
        elif observation.design and observation.term_names:
            rows = tuple(
                tuple(as_fraction(Fraction(item)) for item in row)
                for row in observation.design
            )
            names = observation.term_names
            target_frac = (
                tuple(as_fraction(Fraction(item)) for item in observation.target)
                if observation.target
                else None
            )
        else:
            return None
        if target_frac is not None:
            augmented = tuple(row + (value,) for row, value in zip(rows, target_frac, strict=True))
            ints = fractions_to_ints(augmented)
            design = tuple(row[:-1] for row in ints)
            target = tuple(row[-1] for row in ints)
        else:
            design = fractions_to_ints(rows)
            target = None
        return cls(design=design, term_names=names, sort=sort, target=target)

    def cardinality(self) -> int:
        width = len(self.term_names)
        return 0 if width == 0 else (1 << width) - 1

    def origin(self) -> int:
        names = self.term_names
        if "yp" in names:
            return 1 << names.index("yp")
        if "u_xx" in names:
            return 1 << names.index("u_xx")
        if "rho_t" in names and "j_x" in names:
            return (1 << names.index("rho_t")) | (1 << names.index("j_x"))
        width = len(names)
        return (1 << width) - 1 if width else 1

    def neighbors(self, candidate: Candidate) -> Sequence[int]:
        if not isinstance(candidate, int):
            return ()
        width = len(self.term_names)
        limit = self.cardinality()
        out: list[int] = []
        for bit in range(width):
            nxt = int(candidate) ^ (1 << bit)
            if 0 < nxt <= limit:
                out.append(nxt)
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, int):
            return 0
        value = int(candidate)
        bonus = 0
        names = self.term_names
        if "yp" in names and value & (1 << names.index("yp")):
            bonus += 5
        if "u_xx" in names and value & (1 << names.index("u_xx")):
            bonus += 3
        return bonus - value.bit_count()

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, int):
            return None
        mask = int(candidate)
        width = len(self.term_names)
        if width == 0 or not 0 < mask <= self.cardinality():
            return None
        cols = [index for index in range(width) if mask & (1 << index)]
        if not cols:
            return None
        sub = [[row[index] for index in cols] for row in self.design]
        if self.target is not None and any(value != 0 for value in self.target):
            matrix = [list(row) + [-value] for row, value in zip(sub, self.target, strict=True)]
        else:
            matrix = [list(row) for row in sub]
        null = integer_null_space(matrix)
        ok = len(null) == 1
        from omnibias.core.proof.condition import condition_honesty

        pretty = None
        coeffs: tuple[str, ...] = ()
        if ok:
            vector = null[0]
            pieces: list[str] = []
            coeff_list: list[str] = []
            design_coefs = vector[: len(cols)]
            for name, coef in zip((self.term_names[i] for i in cols), design_coefs, strict=True):
                coeff_list.append(str(coef))
                if coef == 0:
                    continue
                pieces.append(f"({coef})*{name}")
            if self.target is not None and len(vector) > len(cols):
                coeff_list.append(str(vector[-1]))
            pretty = " + ".join(pieces) + " = 0" if pieces else "0 = 0"
            coeffs = tuple(coeff_list)
        return ExactCheck(
            ok=ok,
            payload={
                "sort": self.sort,
                "mask": mask,
                "tier": "exact",
                "tokens": len(cols),
                "annihilator": pretty,
                "equation": (
                    None
                    if pretty is None
                    else {
                        "kind": "polynomial_identity",
                        "pretty": pretty,
                        "coefficients": list(coeffs),
                    }
                ),
                "honesty": condition_honesty(discovered=ok),
                "no_condition_exists_claim": False,
            },
        )

    def grown_product_check(self) -> ExactCheck | None:
        """One-step column products. Used by ``select_class(..., grow=True)`` on a miss."""

        from omnibias.core.proof.condition import condition_honesty

        width = len(self.term_names)
        if width == 0:
            return None
        for left in range(width):
            for right in range(left, width):
                product = tuple(row[left] * row[right] for row in self.design)
                product_name = f"{self.term_names[left]}*{self.term_names[right]}"
                for other in range(width):
                    if self.term_names[other] == product_name:
                        continue
                    matrix = [
                        [row[other], product[index]]
                        for index, row in enumerate(self.design)
                    ]
                    if self.target is not None and any(value != 0 for value in self.target):
                        matrix = [
                            list(row) + [-value]
                            for row, value in zip(matrix, self.target, strict=True)
                        ]
                    null = integer_null_space(matrix)
                    if len(null) != 1:
                        continue
                    vector = null[0]
                    if self.target is not None and any(value != 0 for value in self.target):
                        if len(vector) < 3 or vector[-1] == 0:
                            continue
                    pretty = (
                        f"({vector[0]})*{self.term_names[other]} + ({vector[1]})*{product_name} = 0"
                    )
                    return ExactCheck(
                        ok=True,
                        payload={
                            "sort": self.sort,
                            "tier": "exact",
                            "tokens": 2,
                            "grown": True,
                            "annihilator": pretty,
                            "equation": {
                                "kind": "polynomial_identity",
                                "pretty": pretty,
                                "coefficients": [str(vector[0]), str(vector[1])],
                            },
                            "honesty": condition_honesty(discovered=True),
                            "no_condition_exists_claim": False,
                        },
                    )
        return None


__all__ = [
    "FEATURE_DIM",
    "ClassGate",
    "ClassHit",
    "ClassMemory",
    "ClassSelection",
    "ClassTier",
    "FrequencyGate",
    "LinearSpanFamily",
    "Observation",
    "class_hit_from_check",
    "fractions_to_ints",
    "rank_class_hits",
]
