# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact condition-language families owned by omnibias-symbolic.

Candidates are :class:`~omnibias.core.proof.condition.ConditionHypothesis`
values. Checkers reuse the existing exact-``Q`` spans (tanh Riccati, planted
heat, integer derivatives). Soft residuals never become ``ExactCheck``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from omnibias.core.proof.condition import (
    ConditionHypothesis,
    ConditionToken,
    GrammarGrowthFamily,
    GrammarSpec,
    condition_honesty,
    emit_condition,
    register_condition_sort,
)
from omnibias.core.proof.discovery import Candidate, DiscoveredEquation, ExactCheck, Statement
from omnibias.core.proof.lift import integer_null_space
from omnibias.core.proof.observe import LinearSpanFamily, Observation
from omnibias.symbolic.families import ActivationIdentityFamily
from omnibias.symbolic.lift import planted_heat_rational, snap_sparse_equation, sparse_from_coeffs

_JET_TERMS = ("one", "y", "y2", "yp", "ypp")
_PDE_TERMS = ("u", "u_x", "u_xx")
_CONS_TERMS = ("rho_t", "j_x")
_SECOND_DERIV = frozenset({"u_xx", "u_yy", "lap", "laplacian"})


def _grammar(
    sort: str,
    tokens: tuple[str, ...],
    *,
    constructors: tuple[str, ...] = (),
    complete: bool = False,
    max_growth_depth: int = 1,
) -> GrammarSpec:
    return GrammarSpec(
        sorts=(sort,),  # type: ignore[arg-type]
        tokens_by_sort={sort: tokens},  # type: ignore[dict-item]
        constructors=constructors,
        max_growth_depth=max_growth_depth,
        complete=complete,
    )


def hypothesis_from_names(
    sort: str,
    names: Sequence[str],
    *,
    coefficients: tuple[str, ...] = (),
    constructors: tuple[str, ...] = (),
) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort=sort,  # type: ignore[arg-type]
        tokens=tuple(ConditionToken(sort, name) for name in names),  # type: ignore[arg-type]
        coefficients=coefficients,
        constructors=constructors,
    )


def _mask_of(hypothesis: ConditionHypothesis, terms: tuple[str, ...]) -> int | None:
    mask = 0
    for token in hypothesis.tokens:
        if token.name not in terms:
            return None
        mask |= 1 << terms.index(token.name)
    return mask if mask else None


def _flip_names(
    hypothesis: ConditionHypothesis,
    terms: tuple[str, ...],
) -> list[ConditionHypothesis]:
    names = set(hypothesis.token_names())
    out: list[ConditionHypothesis] = []
    for term in terms:
        nxt = set(names)
        if term in nxt:
            if len(nxt) == 1:
                continue
            nxt.remove(term)
        else:
            nxt.add(term)
        out.append(hypothesis_from_names(hypothesis.sort, sorted(nxt, key=terms.index)))
    return out


def _attach_honesty(check: ExactCheck, *, discovered: bool) -> ExactCheck:
    payload = dict(check.payload)
    honesty = dict(payload.get("honesty", {})) if isinstance(payload.get("honesty"), dict) else {}
    honesty.update(condition_honesty(discovered=discovered))
    payload["honesty"] = honesty
    payload["no_condition_exists_claim"] = False
    return ExactCheck(ok=check.ok, payload=payload)


@dataclass
class JetConditionFamily:
    """Tanh Riccati as a jet-monomial hypothesis (reuses ActivationIdentityFamily)."""

    name: str = "condition_jet_monomial"
    grammar: GrammarSpec = field(
        default_factory=lambda: _grammar("jet_monomial", _JET_TERMS, complete=False)
    )
    _inner: ActivationIdentityFamily = field(default_factory=ActivationIdentityFamily)

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            hypothesis_from_names("jet_monomial", ("yp",)),
            parent="Riccati identities",
            parent_status="already_true",
            obligation="a jet-monomial identity for tanh in this grammar",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return (1 << len(_JET_TERMS)) - 1

    def origin(self) -> ConditionHypothesis:
        return hypothesis_from_names("jet_monomial", ("yp",))

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis):
            return ()
        return _flip_names(candidate, _JET_TERMS)

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis):
            return 0
        names = candidate.token_names()
        return (5 if "yp" in names else 0) - len(names)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "jet_monomial":
            return None
        mask = _mask_of(candidate, _JET_TERMS)
        if mask is None:
            return None
        inner = self._inner.check(mask)
        if inner is None:
            return None
        payload = dict(inner.payload)
        payload["hypothesis"] = candidate.as_dict()
        merged = ExactCheck(ok=inner.ok, payload=payload)
        return _attach_honesty(merged, discovered=inner.ok)


@dataclass
class PdeConditionFamily:
    """Planted heat as a PDE-operator hypothesis."""

    diffusivity: Fraction = Fraction(1, 8)
    name: str = "condition_pde_operator"
    grammar: GrammarSpec = field(
        default_factory=lambda: _grammar(
            "pde_operator",
            _PDE_TERMS,
            constructors=("multiply_columns",),
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            hypothesis_from_names("pde_operator", _PDE_TERMS),
            parent="heat equation",
            parent_status="already_true",
            obligation="snapped heat-operator coefficients with identically zero residual",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return (1 << len(_PDE_TERMS)) - 1

    def origin(self) -> ConditionHypothesis:
        return hypothesis_from_names("pde_operator", _PDE_TERMS)

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis):
            return ()
        return _flip_names(candidate, _PDE_TERMS)

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis):
            return 0
        return (3 if "u_xx" in candidate.token_names() else 0) - len(candidate.tokens)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "pde_operator":
            return None
        names = candidate.token_names()
        if "u_xx" not in names:
            return ExactCheck(
                ok=False,
                payload={
                    "hypothesis": candidate.as_dict(),
                    "honesty": condition_honesty(discovered=False),
                },
            )
        design, target, cols = planted_heat_rational(diffusivity=self.diffusivity)
        index = {name: i for i, name in enumerate(cols)}
        sub = [[row[index[name]] for name in names if name in index] for row in design]
        coefs = [Fraction(0)] * len(names)
        if "u_xx" in names:
            coefs[list(names).index("u_xx")] = self.diffusivity
        soft = sparse_from_coeffs([float(c) for c in coefs], names)
        snapped = snap_sparse_equation(soft, sub, target, denom_bound=16)
        ok = snapped is not None
        return ExactCheck(
            ok=ok,
            payload={
                "hypothesis": candidate.as_dict(),
                "equation": None if snapped is None else snapped.as_dict(),
                "annihilator": None if snapped is None else snapped.pretty,
                "honesty": condition_honesty(discovered=ok),
            },
        )


def planted_continuity_grid(
    n: int = 3,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """``ρ=t``, ``j=-x`` so ``ρ_t + j_x = 0`` on an integer ``(t, x)`` grid."""

    design: list[list[Fraction]] = []
    target: list[Fraction] = []
    for _t in range(n):
        for _x in range(n):
            design.append([Fraction(1), Fraction(-1)])
            target.append(Fraction(0))
    return design, target


@dataclass
class ConservationConditionFamily:
    """Exact continuity residual on a planted integer grid. Not Noether / torch."""

    name: str = "condition_conservation"
    grammar: GrammarSpec = field(
        default_factory=lambda: _grammar("conservation", _CONS_TERMS, complete=False)
    )

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            hypothesis_from_names("conservation", _CONS_TERMS),
            parent="continuity equation",
            parent_status="already_true",
            obligation="ρ_t + j_x = 0 on the planted integer grid",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return 3

    def origin(self) -> ConditionHypothesis:
        return hypothesis_from_names("conservation", _CONS_TERMS)

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis):
            return ()
        return _flip_names(candidate, _CONS_TERMS)

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis):
            return 0
        return len(candidate.tokens)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "conservation":
            return None
        names = candidate.token_names()
        if set(names) != set(_CONS_TERMS):
            return ExactCheck(
                ok=False,
                payload={
                    "hypothesis": candidate.as_dict(),
                    "honesty": condition_honesty(discovered=False),
                },
            )
        design, target = planted_continuity_grid()
        soft = sparse_from_coeffs((1.0, 1.0), _CONS_TERMS)
        snapped = snap_sparse_equation(
            soft, design, target, denom_bound=8, kind="conservation"
        )
        ok = snapped is not None
        return ExactCheck(
            ok=ok,
            payload={
                "hypothesis": candidate.as_dict(),
                "equation": None if snapped is None else snapped.as_dict(),
                "annihilator": None if snapped is None else snapped.pretty,
                "honesty": condition_honesty(discovered=ok),
            },
        )


def _poly_derivative(coeffs: tuple[int, ...], order: int) -> tuple[int, ...]:
    current = list(coeffs)
    for _ in range(order):
        current = [i * current[i] for i in range(1, len(current))]
    return tuple(current)


def _eval_poly(coeffs: tuple[int, ...], x: int) -> int:
    acc = 0
    power = 1
    for coef in coeffs:
        acc += coef * power
        power *= x
    return acc


def _column(observation: Observation, name: str) -> tuple[Fraction, ...] | None:
    if name not in observation.jet_names or not observation.jet_rows:
        return None
    index = observation.jet_names.index(name)
    return tuple(Fraction(row[index]) for row in observation.jet_rows)


def _sample_xs(observation: Observation | None) -> tuple[Fraction, ...]:
    if observation is not None and observation.sample_x:
        return tuple(Fraction(item) for item in observation.sample_x)
    return (Fraction(1), Fraction(2), Fraction(3), Fraction(4))


def _interpolate_poly(
    xs: Sequence[Fraction],
    ys: Sequence[Fraction],
) -> tuple[Fraction, ...] | None:
    if len(xs) != len(ys) or not xs:
        return None
    n = len(xs)
    table = [list(ys)]
    for depth in range(1, n):
        row: list[Fraction] = []
        for index in range(n - depth):
            denom = xs[index + depth] - xs[index]
            if denom == 0:
                return None
            row.append((table[depth - 1][index + 1] - table[depth - 1][index]) / denom)
        table.append(row)
    coeffs = [Fraction(0)] * n
    newton = [table[i][0] for i in range(n)]
    basis = [Fraction(1)] + [Fraction(0)] * (n - 1)
    for degree, node in enumerate(newton):
        for power in range(n):
            coeffs[power] += node * basis[power]
        if degree + 1 >= n:
            break
        shifted = [Fraction(0)] * n
        for power, coef in enumerate(basis):
            if coef == 0:
                continue
            shifted[power] -= coef * xs[degree]
            if power + 1 < n:
                shifted[power + 1] += coef
        basis = shifted
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    for x, y in zip(xs, ys, strict=True):
        if _eval_frac_poly(tuple(coeffs), x) != y:
            return None
    return tuple(coeffs)


def _eval_frac_poly(coeffs: Sequence[Fraction], x: Fraction) -> Fraction:
    acc = Fraction(0)
    power = Fraction(1)
    for coef in coeffs:
        acc += coef * power
        power *= x
    return acc


def _poly_derivative_frac(coeffs: Sequence[Fraction], order: int) -> tuple[Fraction, ...]:
    current = list(coeffs)
    for _ in range(order):
        current = [Fraction(i) * current[i] for i in range(1, len(current))]
    return tuple(current)


@dataclass
class FractionalOrderFamily:
    """Integer derivative order of an observed 1-D table. Non-integer stays empirical."""

    observation: Observation | None = None
    name: str = "condition_fractional_order"
    orders: tuple[int, ...] = (0, 1, 2)
    grammar: GrammarSpec = field(
        default_factory=lambda: _grammar(
            "fractional_order", ("order:0", "order:1", "order:2"), complete=False
        )
    )

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            hypothesis_from_names("fractional_order", ("order:1",), coefficients=("1",)),
            parent="integer differential order",
            parent_status="already_true",
            obligation="an integer order n with D^n y identically 2x on the sample grid",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return len(self.orders)

    def origin(self) -> ConditionHypothesis:
        return hypothesis_from_names(
            "fractional_order", (f"order:{self.orders[0]}",), coefficients=(str(self.orders[0]),)
        )

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.coefficients:
            return ()
        order = int(candidate.coefficients[0])
        out: list[ConditionHypothesis] = []
        for nxt in self.orders:
            if nxt == order:
                continue
            out.append(
                hypothesis_from_names(
                    "fractional_order", (f"order:{nxt}",), coefficients=(str(nxt),)
                )
            )
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.coefficients:
            return 0
        return -abs(int(candidate.coefficients[0]) - 1)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "fractional_order":
            return None
        if not candidate.coefficients:
            return None
        order = int(candidate.coefficients[0])
        if order not in self.orders:
            return None
        xs = _sample_xs(self.observation)
        y_col = _column(self.observation, "y") if self.observation is not None else None
        if y_col is None:
            coeffs: tuple[Fraction, ...] = (Fraction(0), Fraction(0), Fraction(1))
            xs = (Fraction(1), Fraction(2), Fraction(3))
        else:
            fitted = _interpolate_poly(xs[: len(y_col)], y_col[: len(xs)])
            if fitted is None:
                return ExactCheck(
                    ok=False,
                    payload={
                        "hypothesis": candidate.as_dict(),
                        "order": order,
                        "honesty": condition_honesty(discovered=False),
                    },
                )
            coeffs = fitted
        deriv = _poly_derivative_frac(coeffs, order)
        ok = all(_eval_frac_poly(deriv, x) == 2 * x for x in xs[: max(len(xs), 1)])
        pretty = f"D^{order}(y) = 2x"
        equation = (
            DiscoveredEquation(
                kind="polynomial_identity",
                pretty=pretty,
                coefficients=(str(order),),
            )
            if ok
            else None
        )
        return ExactCheck(
            ok=ok,
            payload={
                "hypothesis": candidate.as_dict(),
                "order": order,
                "equation": None if equation is None else equation.as_dict(),
                "annihilator": pretty if ok else None,
                "honesty": condition_honesty(discovered=ok),
            },
        )


def _abs_default_samples() -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    xs = (Fraction(-2), Fraction(-1), Fraction(-1, 2), Fraction(1, 2), Fraction(1), Fraction(2))
    yp = tuple(Fraction(-1) if x < 0 else Fraction(1) for x in xs)
    return xs, yp


def _piecewise_samples(
    observation: Observation | None,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    if observation is None:
        return _abs_default_samples()
    xs = _sample_xs(observation) if observation.sample_x else None
    yp = _column(observation, "yp")
    extra = observation.extra_map()
    if extra.get("signed") == "1" and xs is None:
        return _abs_default_samples()
    if xs is not None and yp is not None:
        n = min(len(xs), len(yp))
        return xs[:n], yp[:n]
    if extra.get("signed") == "1":
        return _abs_default_samples()
    return _abs_default_samples()


@dataclass
class PiecewiseHybridFamily:
    """Two-region sign split of an observed ``yp`` column. Not a learned SoftTree."""

    observation: Observation | None = None
    name: str = "condition_piecewise_hybrid"
    grammar: GrammarSpec = field(
        default_factory=lambda: _grammar(
            "piecewise_hybrid",
            ("yp",),
            constructors=("split_partition",),
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            hypothesis_from_names("piecewise_hybrid", ("yp",)),
            parent="piecewise identities",
            parent_status="already_true",
            obligation="y' = sign(x) on the two-region split of the observed samples",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return 2

    def origin(self) -> ConditionHypothesis:
        return hypothesis_from_names("piecewise_hybrid", ("yp",))

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis):
            return ()
        from omnibias.core.proof.condition import apply_constructor

        grown = apply_constructor(candidate, "split_partition", self.grammar)
        return () if grown is None else (grown,)

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis):
            return 0
        return sum(1 for name in candidate.token_names() if "@" in name)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "piecewise_hybrid":
            return None
        names = candidate.token_names()
        split = "yp@0" in names and "yp@1" in names
        xs, yp = _piecewise_samples(self.observation)
        left = tuple(yp[i] for i, x in enumerate(xs) if x < 0)
        right = tuple(yp[i] for i, x in enumerate(xs) if x > 0)
        if not split:
            constant = bool(yp) and all(value == yp[0] for value in yp)
            return ExactCheck(
                ok=constant,
                payload={
                    "hypothesis": candidate.as_dict(),
                    "split": False,
                    "honesty": condition_honesty(discovered=constant),
                },
            )
        left_ok = bool(left) and all(value == Fraction(-1) for value in left)
        right_ok = bool(right) and all(value == Fraction(1) for value in right)
        ok = left_ok and right_ok
        pretty = "yp@0 + 1 = 0 and yp@1 - 1 = 0"
        equation = DiscoveredEquation(
            kind="polynomial_identity",
            pretty=pretty,
            coefficients=("-1", "1"),
        )
        return ExactCheck(
            ok=ok,
            payload={
                "hypothesis": candidate.as_dict(),
                "split": True,
                "left": [str(v) for v in left],
                "right": [str(v) for v in right],
                "equation": equation.as_dict(),
                "annihilator": pretty,
                "honesty": condition_honesty(discovered=ok),
            },
        )


def _square_jet_value(name: str, x: int) -> int:
    if name == "y":
        return x * x
    if name == "yp":
        return 2 * x
    if name == "ypp":
        return 2
    if "*" in name:
        left, right = name.split("*", 1)
        return _square_jet_value(left, x) * _square_jet_value(right, x)
    raise KeyError(name)


def square_jet_identity_check(hypothesis: ConditionHypothesis) -> ExactCheck | None:
    """Exact nullity-one test for ``y=x^2`` columns named in ``hypothesis``."""

    names = hypothesis.token_names()
    if not names:
        return None
    xs = (1, 2, 3, 4)
    try:
        rows = [[_square_jet_value(name, x) for name in names] for x in xs]
    except KeyError:
        return None
    null = integer_null_space(rows)
    ok = len(null) == 1
    pretty = None
    coeffs: tuple[str, ...] = ()
    if ok:
        vector = null[0]
        pieces = []
        coeff_list = []
        for name, coef in zip(names, vector, strict=True):
            coeff_list.append(str(coef))
            if coef == 0:
                continue
            pieces.append(f"({coef})*{name}")
        pretty = " + ".join(pieces) + " = 0"
        coeffs = tuple(coeff_list)
    equation = (
        None
        if pretty is None
        else DiscoveredEquation(kind="polynomial_identity", pretty=pretty, coefficients=coeffs)
    )
    return ExactCheck(
        ok=ok,
        payload={
            "hypothesis": hypothesis.as_dict(),
            "equation": None if equation is None else equation.as_dict(),
            "annihilator": pretty,
            "honesty": condition_honesty(discovered=ok),
        },
    )


def jet_growth_family(*, complete: bool = False) -> GrammarGrowthFamily:
    """``{y, yp, ypp}`` misses; ``compose_jets(yp, yp)`` yields ``yp*yp - 4 y = 0``."""

    seed = hypothesis_from_names("jet_monomial", ("y", "yp", "ypp"))
    grammar = GrammarSpec(
        sorts=("jet_monomial",),
        tokens_by_sort={"jet_monomial": ("y", "yp", "ypp")},
        constructors=("compose_jets",),
        max_growth_depth=1,
        complete=complete,
    )
    family = GrammarGrowthFamily(
        grammar=grammar,
        seed=seed,
        checker=square_jet_identity_check,
        name="condition_grammar_growth",
        statement=Statement(
            name="condition_grammar_growth",
            obligation="a grown jet hypothesis is an exact identity for y=x^2",
            parent="jet identities",
            parent_status="already_true",
        ),
    )
    return family


def _eval_int_poly(coeffs: Sequence[int], value: Fraction) -> Fraction:
    acc = Fraction(0)
    for power, coef in enumerate(coeffs):
        acc += coef * value**power
    return acc


def observation_tanh() -> Observation:
    """Exact tanh jet table at rational ``t``. Binds ``jet_monomial`` only."""

    from omnibias.core.verified.coeffs import tanh_poly_coeffs_exact

    samples = (
        Fraction(-3, 4),
        Fraction(-1, 2),
        Fraction(-1, 4),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 4),
    )
    t1 = tanh_poly_coeffs_exact(1)
    t2 = tanh_poly_coeffs_exact(2)
    names = _JET_TERMS
    rows: list[tuple[str, ...]] = []
    for t in samples:
        values = {
            "one": Fraction(1),
            "y": t,
            "y2": t * t,
            "yp": _eval_int_poly(t1, t),
            "ypp": _eval_int_poly(t2, t),
        }
        rows.append(tuple(str(values[name]) for name in names))
    return Observation(
        tag="tanh",
        jet_names=names,
        jet_rows=tuple(rows),
        sample_x=tuple(str(t) for t in samples),
    )


def observation_heat(*, diffusivity: Fraction = Fraction(1, 8)) -> Observation:
    design, target, names = planted_heat_rational(diffusivity=diffusivity)
    return Observation(
        tag="heat",
        design=tuple(tuple(str(value) for value in row) for row in design),
        target=tuple(str(value) for value in target),
        term_names=tuple(names),
    )


def observation_heat_perturbed(*, diffusivity: Fraction = Fraction(1, 8)) -> Observation:
    planted = observation_heat(diffusivity=diffusivity)
    bumped = list(planted.target)
    if bumped:
        bumped[-1] = str(Fraction(bumped[-1]) + Fraction(1, 10))
    return Observation(
        tag="heat",
        design=planted.design,
        target=tuple(bumped),
        term_names=planted.term_names,
    )


def observation_continuity() -> Observation:
    design, target = planted_continuity_grid()
    return Observation(
        tag="continuity",
        design=tuple(tuple(str(value) for value in row) for row in design),
        target=tuple(str(value) for value in target),
        term_names=_CONS_TERMS,
    )


def observation_square() -> Observation:
    xs = (1, 2, 3, 4)
    names = ("y", "yp", "ypp")
    rows = tuple((str(x * x), str(2 * x), "2") for x in xs)
    return Observation(
        jet_names=names,
        jet_rows=rows,
        sample_x=tuple(str(x) for x in xs),
    )


def observation_abs() -> Observation:
    xs, yp = _abs_default_samples()
    y = tuple(x if x >= 0 else -x for x in xs)
    return Observation(
        jet_names=("y", "yp"),
        jet_rows=tuple((str(y[i]), str(yp[i])) for i in range(len(xs))),
        sample_x=tuple(str(x) for x in xs),
    )


def _yp_two_sided(observation: Observation) -> bool:
    yp = _column(observation, "yp")
    if yp is None:
        return observation.extra_map().get("signed") == "1" or observation.tag == "abs"
    return any(value < 0 for value in yp) and any(value > 0 for value in yp)


def bind_jet_monomial(observation: Observation | None = None) -> object | None:
    if observation is None:
        return JetConditionFamily()
    if observation.jet_rows:
        family = LinearSpanFamily.from_observation(observation, sort="jet_monomial")
        if family is not None:
            return family
        return JetConditionFamily()
    if observation.tag == "tanh":
        return JetConditionFamily()
    return None


def bind_pde_operator(observation: Observation | None = None) -> object | None:
    if observation is None:
        return PdeConditionFamily()
    names = observation.term_names
    if observation.design and any(name in _SECOND_DERIV for name in names):
        family = LinearSpanFamily.from_observation(observation, sort="pde_operator")
        if family is not None:
            return family
        return PdeConditionFamily()
    if observation.tag == "heat":
        return PdeConditionFamily()
    return None


def bind_conservation(observation: Observation | None = None) -> object | None:
    if observation is None:
        return ConservationConditionFamily()
    names = observation.term_names
    if "rho_t" in names and "j_x" in names:
        if observation.design:
            family = LinearSpanFamily.from_observation(observation, sort="conservation")
            if family is not None:
                return family
        return ConservationConditionFamily()
    if observation.tag == "continuity":
        return ConservationConditionFamily()
    return None


def bind_fractional_order(observation: Observation | None = None) -> object | None:
    if observation is None:
        return FractionalOrderFamily()
    names = observation.jet_names
    if {"y", "yp", "ypp"}.issubset(names) or observation.tag == "square":
        return FractionalOrderFamily(observation=observation)
    return None


def bind_piecewise_hybrid(observation: Observation | None = None) -> object | None:
    if observation is None:
        return PiecewiseHybridFamily()
    extra = observation.extra_map()
    if extra.get("signed") == "1" or observation.tag == "abs":
        return PiecewiseHybridFamily(observation=observation)
    if observation.sample_x and _yp_two_sided(observation):
        return PiecewiseHybridFamily(observation=observation)
    return None


def _residual_values(observation: Observation) -> tuple[Fraction, ...] | None:
    raw = observation.extra_map().get("residual")
    if not raw:
        return None
    parts = [part for part in raw.replace(";", ",").split(",") if part]
    try:
        return tuple(Fraction(part) for part in parts)
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class ResidualSignFamily:
    """Finite table sign of a packed residual. Not a Lyapunov theorem."""

    observation: Observation
    name: str = "condition_residual_sign"
    complete: bool = False
    empty_miss_detail: str = "no witness in enumerated grammar"

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            hypothesis_from_names("residual_sign", ("sign",)),
            parent="residual inequalities",
            parent_status="already_true",
            obligation="every packed residual sample has the same sign",
        )

    def cardinality(self) -> int:
        return 1

    def origin(self) -> ConditionHypothesis:
        return hypothesis_from_names("residual_sign", ("sign",))

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        return ()

    def score(self, candidate: Candidate) -> int:
        return 0

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "residual_sign":
            return None
        values = _residual_values(self.observation)
        if values is None:
            return None
        nonnegative = all(value >= 0 for value in values)
        nonpositive = all(value <= 0 for value in values)
        ok = bool(values) and (nonnegative or nonpositive)
        return ExactCheck(
            ok=ok,
            payload={
                "tier": "enclosure",
                "sign": ">=0" if nonnegative else ("<=0" if nonpositive else "mixed"),
                "n": len(values),
                "honesty": condition_honesty(discovered=ok),
            },
        )


def bind_residual_sign(observation: Observation | None = None) -> object | None:
    if observation is None or _residual_values(observation) is None:
        return None
    return ResidualSignFamily(observation=observation)


def _register_sorts() -> None:
    register_condition_sort("jet_monomial", bind_jet_monomial)
    register_condition_sort("pde_operator", bind_pde_operator)
    register_condition_sort("conservation", bind_conservation)
    register_condition_sort("fractional_order", bind_fractional_order)
    register_condition_sort("piecewise_hybrid", bind_piecewise_hybrid)
    register_condition_sort("residual_sign", bind_residual_sign)


_register_sorts()


__all__ = [
    "ConservationConditionFamily",
    "FractionalOrderFamily",
    "JetConditionFamily",
    "PdeConditionFamily",
    "PiecewiseHybridFamily",
    "ResidualSignFamily",
    "bind_conservation",
    "bind_fractional_order",
    "bind_jet_monomial",
    "bind_pde_operator",
    "bind_piecewise_hybrid",
    "bind_residual_sign",
    "hypothesis_from_names",
    "jet_growth_family",
    "observation_abs",
    "observation_continuity",
    "observation_heat",
    "observation_heat_perturbed",
    "observation_square",
    "observation_tanh",
    "planted_continuity_grid",
    "square_jet_identity_check",
]
