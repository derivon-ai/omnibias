# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Pack raw tables into a tag-optional :class:`~omnibias.core.proof.observe.Observation`.

Core stays fraction strings. This module may use numpy. A tag is never required
for a binder to attach.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.proof.observe import Observation, PolyTerms

Number = int | float | str | Fraction


def _frac_str(value: Number, *, denom: int = 10_000) -> str:
    if isinstance(value, str):
        return str(Fraction(value))
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return str(Fraction(value).limit_denominator(denom))


def _row(values: Sequence[Number], *, denom: int = 10_000) -> tuple[str, ...]:
    return tuple(_frac_str(value, denom=denom) for value in values)


def _poly_terms(
    terms: Sequence[tuple[Sequence[int], Number]],
    *,
    denom: int = 10_000,
) -> PolyTerms:
    return tuple(
        (tuple(int(power) for power in exp), _frac_str(coeff, denom=denom))
        for exp, coeff in terms
    )


def pack_sequence(samples: Sequence[Number], *, tag: str = "") -> Observation:
    return Observation(tag=tag, sequence=_row(samples))


def pack_jets(
    columns: Mapping[str, Sequence[Number]],
    *,
    sample_x: Sequence[Number] | None = None,
    tag: str = "",
) -> Observation:
    names = tuple(columns)
    if not names:
        return Observation(tag=tag, sample_x=_row(sample_x) if sample_x is not None else ())
    length = len(next(iter(columns.values())))
    rows = tuple(
        tuple(_frac_str(columns[name][index]) for name in names) for index in range(length)
    )
    return Observation(
        tag=tag,
        jet_names=names,
        jet_rows=rows,
        sample_x=_row(sample_x) if sample_x is not None else (),
    )


def pack_jet_bundle(bundle: Any, *, names: Sequence[str] | None = None, tag: str = "") -> Observation:
    jets = bundle.jets
    width = int(jets.shape[1])
    labels = tuple(names) if names is not None else tuple(
        "y" if index == 0 else ("yp" if index == 1 else ("ypp" if index == 2 else f"d{index}y"))
        for index in range(width)
    )
    columns = {labels[index]: [jets[row, index] for row in range(jets.shape[0])] for index in range(width)}
    return pack_jets(columns, sample_x=list(bundle.x), tag=tag)


def pack_design(
    design: Sequence[Sequence[Number]],
    target: Sequence[Number],
    term_names: Sequence[str],
    *,
    tag: str = "",
) -> Observation:
    return Observation(
        tag=tag,
        design=tuple(_row(row) for row in design),
        target=_row(target),
        term_names=tuple(str(name) for name in term_names),
    )


def pack_graph(
    n: int,
    edges: Sequence[tuple[int, int]],
    *,
    tag: str = "",
    extra: Sequence[tuple[str, str]] = (),
) -> Observation:
    return Observation(
        tag=tag,
        graph_n=int(n),
        graph_edges=tuple((int(edge[0]), int(edge[1])) for edge in edges),
        extra=tuple(extra),
    )


def pack_poly(
    n_vars: int,
    terms: Sequence[tuple[Sequence[int], Number]],
    *,
    constraints: Sequence[Sequence[tuple[Sequence[int], Number]]] = (),
    tag: str = "",
) -> Observation:
    return Observation(
        tag=tag,
        poly_n_vars=int(n_vars),
        poly_terms=_poly_terms(terms),
        poly_constraints=tuple(_poly_terms(poly) for poly in constraints),
    )


def pack_poly_geq(
    left: Sequence[tuple[Sequence[int], Number]],
    right: Sequence[tuple[Sequence[int], Number]],
    *,
    n_vars: int,
    tag: str = "",
) -> Observation:
    """Pack ``p - q`` so ``p >= q`` is a global SOS template."""

    coeffs: dict[tuple[int, ...], Fraction] = {}
    for exp, coeff in left:
        key = tuple(int(power) for power in exp)
        coeffs[key] = coeffs.get(key, Fraction(0)) + Fraction(str(_frac_str(coeff)))
    for exp, coeff in right:
        key = tuple(int(power) for power in exp)
        coeffs[key] = coeffs.get(key, Fraction(0)) - Fraction(str(_frac_str(coeff)))
    terms = tuple((exp, str(value)) for exp, value in coeffs.items() if value != 0)
    return Observation(tag=tag, poly_n_vars=int(n_vars), poly_terms=terms)


def pack_residual(
    values: Sequence[Number],
    *,
    observation: Observation | None = None,
    tag: str = "",
) -> Observation:
    residual = ",".join(_frac_str(value) for value in values)
    extra = (("residual", residual),)
    if observation is None:
        return Observation(tag=tag, extra=extra)
    merged = dict(observation.extra)
    merged["residual"] = residual
    return Observation(
        tag=observation.tag or tag,
        sequence=observation.sequence,
        jet_names=observation.jet_names,
        jet_rows=observation.jet_rows,
        design=observation.design,
        target=observation.target,
        term_names=observation.term_names,
        graph_n=observation.graph_n,
        graph_edges=observation.graph_edges,
        poly_n_vars=observation.poly_n_vars,
        poly_terms=observation.poly_terms,
        sample_x=observation.sample_x,
        poly_constraints=observation.poly_constraints,
        extra=tuple(merged.items()),
    )


def pack_neural_field(field: Any, x: Sequence[Number], *, max_order: int = 2, tag: str = "") -> Observation:
    import numpy as np
    from omnibias.symbolic.discovery import extract_neural_jets

    xs = np.asarray([float(Fraction(str(_frac_str(value)))) for value in x], dtype=float)
    bundle = extract_neural_jets(field, xs, max_order=max_order)
    return pack_jet_bundle(bundle, tag=tag)


def pack_observation(
    *,
    sequence: Sequence[Number] | None = None,
    columns: Mapping[str, Sequence[Number]] | None = None,
    sample_x: Sequence[Number] | None = None,
    design: Sequence[Sequence[Number]] | None = None,
    target: Sequence[Number] | None = None,
    term_names: Sequence[str] | None = None,
    graph_n: int = 0,
    graph_edges: Sequence[tuple[int, int]] = (),
    poly_n_vars: int = 0,
    poly_terms: Sequence[tuple[Sequence[int], Number]] = (),
    poly_constraints: Sequence[Sequence[tuple[Sequence[int], Number]]] = (),
    residual: Sequence[Number] | None = None,
    tag: str = "",
    extra: Sequence[tuple[str, str]] = (),
) -> Observation:
    """Fill whichever views are present. ``tag`` is optional."""

    extra_map = dict(extra)
    if residual is not None:
        extra_map["residual"] = ",".join(_frac_str(value) for value in residual)
    jets = pack_jets(columns, sample_x=sample_x, tag=tag) if columns else Observation(tag=tag)
    return Observation(
        tag=tag,
        sequence=_row(sequence) if sequence is not None else (),
        jet_names=jets.jet_names,
        jet_rows=jets.jet_rows,
        design=tuple(_row(row) for row in design) if design is not None else (),
        target=_row(target) if target is not None else (),
        term_names=tuple(str(name) for name in term_names) if term_names is not None else (),
        graph_n=int(graph_n),
        graph_edges=tuple((int(edge[0]), int(edge[1])) for edge in graph_edges),
        poly_n_vars=int(poly_n_vars),
        poly_terms=_poly_terms(poly_terms) if poly_terms else (),
        sample_x=jets.sample_x if jets.sample_x else (_row(sample_x) if sample_x is not None else ()),
        poly_constraints=tuple(_poly_terms(poly) for poly in poly_constraints),
        extra=tuple(extra_map.items()),
    )


__all__ = [
    "pack_design",
    "pack_graph",
    "pack_jet_bundle",
    "pack_jets",
    "pack_neural_field",
    "pack_observation",
    "pack_poly",
    "pack_poly_geq",
    "pack_residual",
    "pack_sequence",
]
