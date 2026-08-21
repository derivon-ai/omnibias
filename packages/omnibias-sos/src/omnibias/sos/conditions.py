# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""SOS-template condition family: half-degree as a hypothesis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from omnibias.core.proof.condition import (
    ConditionHypothesis,
    ConditionToken,
    GrammarSpec,
    condition_honesty,
    emit_condition,
    register_condition_sort,
)
from omnibias.core.proof.discovery import Candidate, ExactCheck
from omnibias.core.proof.observe import Observation
from omnibias.sos.families import SosDegreeFamily, planted_sum_of_squares
from omnibias.sos.positivstellensatz import certify_nonneg_on_set
from omnibias.sos.problem import Polynomial


def _sos_hypothesis(half: int) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort="sos_template",
        tokens=(ConditionToken("sos_template", "half_degree"),),
        coefficients=(str(half),),
    )


@dataclass
class SosTemplateFamily:
    """Wrap :class:`SosDegreeFamily` as an SOS-template hypothesis."""

    polynomial: Polynomial = field(default_factory=planted_sum_of_squares)
    max_half_degree: int = 2
    name: str = "condition_sos_template"
    grammar: GrammarSpec = field(
        default_factory=lambda: GrammarSpec(
            sorts=("sos_template",),
            tokens_by_sort={"sos_template": ("half_degree",)},
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self._inner = SosDegreeFamily(
            polynomial=self.polynomial,
            max_half_degree=self.max_half_degree,
        )
        self.statement = emit_condition(
            _sos_hypothesis(1),
            parent="global nonnegativity",
            parent_status="already_true",
            obligation="the planted polynomial is SOS at some half-degree in this grammar",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return self.max_half_degree

    def origin(self) -> ConditionHypothesis:
        return _sos_hypothesis(1)

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.coefficients:
            return ()
        half = int(candidate.coefficients[0])
        out: list[ConditionHypothesis] = []
        if half > 1:
            out.append(_sos_hypothesis(half - 1))
        if half < self.max_half_degree:
            out.append(_sos_hypothesis(half + 1))
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.coefficients:
            return 0
        return -int(candidate.coefficients[0])

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "sos_template":
            return None
        if not candidate.coefficients:
            return None
        half = int(candidate.coefficients[0])
        inner = self._inner.check(half)
        if inner is None:
            return None
        payload = dict(inner.payload)
        payload["hypothesis"] = candidate.as_dict()
        honesty = dict(payload.get("honesty", {})) if isinstance(payload.get("honesty"), dict) else {}
        honesty.update(condition_honesty(discovered=inner.ok))
        payload["honesty"] = honesty
        return ExactCheck(ok=inner.ok, payload=payload)


def observation_planted_sos() -> Observation:
    poly = planted_sum_of_squares()
    terms = tuple(
        (exponent, str(Fraction(coeff).limit_denominator()))
        for exponent, coeff in poly.coeffs.items()
    )
    return Observation(tag="sos", poly_n_vars=poly.n_vars, poly_terms=terms)


def _poly_from_terms(n_vars: int, terms: Sequence[tuple[tuple[int, ...], str]]) -> Polynomial:
    coeffs = {
        tuple(int(power) for power in exponent): float(Fraction(coeff))
        for exponent, coeff in terms
    }
    return Polynomial(n_vars, coeffs)


def bind_sos_template(observation: Observation | None = None) -> SosTemplateFamily | None:
    if observation is None:
        return SosTemplateFamily()
    if not observation.poly_terms or observation.poly_constraints:
        return None
    return SosTemplateFamily(
        polynomial=_poly_from_terms(observation.poly_n_vars, observation.poly_terms)
    )


def _onset_hypothesis(half: int) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort="sos_onset",
        tokens=(ConditionToken("sos_onset", "half_degree"),),
        coefficients=(str(half),),
    )


@dataclass
class SosOnsetFamily:
    """Putinar on-set nonnegativity. Enclosure, never a forged exact residual."""

    polynomial: Polynomial = field(default_factory=planted_sum_of_squares)
    constraints: tuple[Polynomial, ...] = ()
    max_half_degree: int = 2
    name: str = "condition_sos_onset"
    grammar: GrammarSpec = field(
        default_factory=lambda: GrammarSpec(
            sorts=("sos_onset",),
            tokens_by_sort={"sos_onset": ("half_degree",)},
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            _onset_hypothesis(1),
            parent="on-set nonnegativity",
            parent_status="already_true",
            obligation="the observed polynomial is nonnegative on the packed constraint set",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return self.max_half_degree

    def origin(self) -> ConditionHypothesis:
        return _onset_hypothesis(1)

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.coefficients:
            return ()
        half = int(candidate.coefficients[0])
        out: list[ConditionHypothesis] = []
        if half > 1:
            out.append(_onset_hypothesis(half - 1))
        if half < self.max_half_degree:
            out.append(_onset_hypothesis(half + 1))
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.coefficients:
            return 0
        return -int(candidate.coefficients[0])

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "sos_onset":
            return None
        if not candidate.coefficients:
            return None
        half = int(candidate.coefficients[0])
        cert = certify_nonneg_on_set(
            self.polynomial,
            self.constraints,
            half_degree=half,
        )
        ok = cert.status == "proved"
        return ExactCheck(
            ok=ok,
            payload={
                "tier": "enclosure",
                "status": cert.status,
                "half_degree": half,
                "hypothesis": candidate.as_dict(),
                "honesty": condition_honesty(discovered=ok),
            },
        )


def observation_planted_onset() -> Observation:
    poly = planted_sum_of_squares()
    terms = tuple(
        (exponent, str(Fraction(coeff).limit_denominator()))
        for exponent, coeff in poly.coeffs.items()
    )
    one = (((0, 0), "1"),)
    return Observation(
        poly_n_vars=poly.n_vars,
        poly_terms=terms,
        poly_constraints=(one,),
    )


def bind_sos_onset(observation: Observation | None = None) -> SosOnsetFamily | None:
    if observation is None:
        return SosOnsetFamily()
    if not observation.poly_terms or not observation.poly_constraints:
        return None
    constraints = tuple(
        _poly_from_terms(observation.poly_n_vars, terms)
        for terms in observation.poly_constraints
    )
    return SosOnsetFamily(
        polynomial=_poly_from_terms(observation.poly_n_vars, observation.poly_terms),
        constraints=constraints,
    )


def _register() -> None:
    register_condition_sort("sos_template", bind_sos_template)
    register_condition_sort("sos_onset", bind_sos_onset)


_register()


__all__ = [
    "SosOnsetFamily",
    "SosTemplateFamily",
    "bind_sos_onset",
    "bind_sos_template",
    "observation_planted_onset",
    "observation_planted_sos",
]
