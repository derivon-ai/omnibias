# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rational polynomial differential systems for elementary Taylor rules.

The independent variable is slot zero; state variables follow. A polynomial
is a tuple of ``(Fraction coefficient, exponent tuple)`` terms. Transcendental
normalization constants belong in state initializers, not these exact rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

Polynomial = tuple[tuple[Fraction, tuple[int, ...]], ...]


@dataclass(frozen=True)
class DifferentialSystemSpec:
    name: str
    states: tuple[str, ...]
    rhs: tuple[Polynomial, ...]
    output: Polynomial
    domain: str = "all real inputs"

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if not states or len(set(states)) != len(states) or len(self.rhs) != len(states):
            raise ValueError("differential system needs unique states and one RHS per state")

        def normalize(poly: Iterable[tuple[Any, Iterable[int]]]) -> Polynomial:
            terms: dict[tuple[int, ...], Fraction] = {}
            for coefficient, powers in poly:
                powers = tuple(powers)
                if len(powers) != len(states) + 1 or any(
                    not isinstance(p, int) or isinstance(p, bool) or p < 0 for p in powers
                ):
                    raise ValueError("polynomial powers must match input plus states and be nonnegative integers")
                terms[powers] = terms.get(powers, Fraction()) + Fraction(coefficient)
            return tuple((coefficient, powers) for powers, coefficient in sorted(terms.items()) if coefficient)

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "rhs", tuple(normalize(p) for p in self.rhs))
        object.__setattr__(self, "output", normalize(self.output))

    @classmethod
    def riccati(cls, name: str, coefficients: Iterable[int | float | Fraction]) -> DifferentialSystemSpec:
        """Adapt ActivationSpec.riccati_polynomial without altering its schema."""
        coefficients = tuple(coefficients)
        if not coefficients:
            raise ValueError("Riccati polynomial must be nonempty")
        return cls(name, (name,), (tuple((Fraction(c), (0, k)) for k, c in enumerate(coefficients)),),
                   ((Fraction(1), (0, 1)),))

    @classmethod
    def from_activation(cls, activation_spec: Any) -> DifferentialSystemSpec:
        polynomial = activation_spec.riccati_polynomial
        if polynomial is None:
            raise ValueError("activation has no Riccati polynomial")
        return cls.riccati(activation_spec.name, polynomial)

    def to_config(self) -> dict[str, Any]:
        def encode(poly: Polynomial) -> list[dict[str, Any]]:
            return [{"coefficient": [c.numerator, c.denominator], "powers": list(p)} for c, p in poly]
        return {"name": self.name, "states": list(self.states), "rhs": [encode(p) for p in self.rhs],
                "output": encode(self.output), "domain": self.domain}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DifferentialSystemSpec:
        def decode(poly: list[dict[str, Any]]) -> Polynomial:
            return tuple((Fraction(*term["coefficient"]), tuple(term["powers"])) for term in poly)
        return cls(config["name"], tuple(config["states"]), tuple(decode(p) for p in config["rhs"]),
                   decode(config["output"]), config.get("domain", "all real inputs"))


def _p(*terms: tuple[int | Fraction, tuple[int, ...]]) -> Polynomial:
    return tuple((Fraction(c), powers) for c, powers in terms)


SYSTEMS: dict[str, DifferentialSystemSpec] = {
    "tanh": DifferentialSystemSpec.riccati("tanh", (1, 0, -1)),
    "sigmoid": DifferentialSystemSpec.riccati("sigmoid", (0, 1, -1)),
    "exp": DifferentialSystemSpec.riccati("exp", (0, 1)),
    "reciprocal": DifferentialSystemSpec.riccati("reciprocal", (0, 0, -1)),
    "sin": DifferentialSystemSpec("sin", ("sin", "cos"),
        (_p((1, (0, 0, 1))), _p((-1, (0, 1, 0)))), _p((1, (0, 1, 0)))),
    "cos": DifferentialSystemSpec("cos", ("sin", "cos"),
        (_p((1, (0, 0, 1))), _p((-1, (0, 1, 0)))), _p((1, (0, 0, 1)))),
    "gaussian": DifferentialSystemSpec("gaussian", ("gaussian",),
        (_p((-1, (1, 1))),), _p((1, (0, 1)))),
    "gaussian_normalized": DifferentialSystemSpec("gaussian_normalized", ("gaussian_normalized",),
        (_p((-1, (1, 1))),), _p((1, (0, 1)))),
    "gaussian_rbf": DifferentialSystemSpec("gaussian_rbf", ("gaussian_rbf",),
        (_p((-2, (1, 1))),), _p((1, (0, 1)))),
    "erf": DifferentialSystemSpec("erf", ("erf", "erf_density"),
        (_p((1, (0, 0, 1))), _p((-2, (1, 0, 1)))), _p((1, (0, 1, 0)))),
    "erfc": DifferentialSystemSpec("erfc", ("erfc", "erf_density"),
        (_p((-1, (0, 0, 1))), _p((-2, (1, 0, 1)))), _p((1, (0, 1, 0)))),
    "gelu": DifferentialSystemSpec("gelu", ("normal_cdf", "gaussian_normalized"),
        (_p((1, (0, 0, 1))), _p((-1, (1, 0, 1)))), _p((1, (1, 1, 0)))),
    "softplus": DifferentialSystemSpec("softplus", ("softplus", "sigmoid"),
        (_p((1, (0, 0, 1))), _p((1, (0, 0, 1)), (-1, (0, 0, 2)))), _p((1, (0, 1, 0)))),
    "log": DifferentialSystemSpec("log", ("log", "reciprocal"),
        (_p((1, (0, 0, 1))), _p((-1, (0, 0, 2)))), _p((1, (0, 1, 0))), "positive input"),
    "sqrt": DifferentialSystemSpec("sqrt", ("sqrt", "reciprocal"),
        (_p((Fraction(1, 2), (0, 1, 1))), _p((-1, (0, 0, 2)))),
        _p((1, (0, 1, 0))), "strictly positive input for unrestricted jets"),
}


def get_system(name: str) -> DifferentialSystemSpec:
    try:
        return SYSTEMS[name]
    except KeyError as exc:
        raise NotImplementedError(f"no differential system registered for {name!r}") from exc


__all__ = ["DifferentialSystemSpec", "Polynomial", "get_system"]
