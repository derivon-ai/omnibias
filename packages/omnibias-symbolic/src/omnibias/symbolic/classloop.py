# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bilevel proposers ↔ snap class loop and the stack entry point.

Inner proposers emit candidates. Snap / ``check`` stays the accept gate.
The driver never marks a grammar complete. Symbolic does not import pinn
at module import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.proof.condition import ConditionSort, select_class
from omnibias.core.proof.observe import ClassMemory, Observation

_STACK_MODULES = (
    "omnibias.symbolic.conditions",
    "omnibias.holonomic.conditions",
    "omnibias.sos.conditions",
    "omnibias.combinatorics.minors",
    "omnibias.combinatorics.conditions",
)


def load_discovery_stack() -> tuple[str, ...]:
    """Import owning packages so their binders register. Core stays clean."""

    import importlib

    loaded: list[str] = []
    for name in _STACK_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        loaded.append(name)
    from omnibias.core.proof.condition import list_condition_sorts

    return list_condition_sorts() if loaded else ()


def _jet_arrays(
    observation: Observation,
) -> tuple[Any, Any] | None:
    if not observation.jet_rows or not observation.jet_names:
        return None
    import numpy as np

    names = observation.jet_names
    order = []
    for label in ("y", "yp", "ypp"):
        if label in names:
            order.append(names.index(label))
    if len(order) < 2:
        order = list(range(min(3, len(names))))
    if len(order) < 2:
        return None
    jets = np.asarray(
        [[float(Fraction(row[index])) for index in order] for row in observation.jet_rows],
        dtype=float,
    )
    if observation.sample_x:
        xs = np.asarray([float(Fraction(item)) for item in observation.sample_x], dtype=float)
    else:
        xs = np.arange(jets.shape[0], dtype=float)
    return xs, jets


def _run_stlsq(observation: Observation, *, denom_bound: int) -> dict[str, Any] | None:
    if not (observation.design and observation.target and observation.term_names):
        return None
    import numpy as np
    from omnibias.symbolic.discovery import fit_sparse_equation
    from omnibias.symbolic.propose import propose_jet_condition

    design = [[float(Fraction(value)) for value in row] for row in observation.design]
    target = [float(Fraction(value)) for value in observation.target]
    sort: ConditionSort = "pde_operator"
    if "u_xx" in observation.term_names:
        sort = "pde_operator"
    elif observation.jet_names:
        sort = "jet_monomial"
    equation = fit_sparse_equation(
        np.asarray(design, dtype=float),
        np.asarray(target, dtype=float),
        list(observation.term_names),
    )
    return propose_jet_condition(
        equation,
        design,
        target,
        sort=sort,
        denom_bound=denom_bound,
        kind="pde_span" if sort == "pde_operator" else "polynomial_identity",
    )


def _run_neural_jet(observation: Observation, *, denom_bound: int) -> dict[str, Any] | None:
    arrays = _jet_arrays(observation)
    if arrays is None:
        return None
    import numpy as np
    from omnibias.symbolic.discovery import (
        JetBundle,
        NeuralJetDiscoverer,
        build_jet_relation_library,
    )
    from omnibias.symbolic.propose import propose_jet_condition

    xs, jets = arrays
    bundle = JetBundle(x=xs, jets=jets)
    discoverer = NeuralJetDiscoverer(
        max_library_degree=2,
        alphas=(1e-8, 1e-6),
        thresholds=(1e-5, 1e-4),
        include_x=True,
    )
    result = discoverer.discover(bundle, bundle, bundle)
    design, _names = build_jet_relation_library(
        bundle,
        lhs_order=result.lhs_order,
        max_degree=2,
        include_x=True,
    )
    target = np.asarray(jets[:, result.lhs_order], dtype=float)
    return propose_jet_condition(
        result.equation,
        design.tolist(),
        target.tolist(),
        sort="jet_monomial",
        denom_bound=denom_bound,
        kind="polynomial_identity",
    )


def _run_field(
    observation: Observation,
    *,
    denom_bound: int,
    hidden: int,
) -> dict[str, Any] | None:
    y_col = None
    if "y" in observation.jet_names and observation.jet_rows:
        index = observation.jet_names.index("y")
        y_col = [float(Fraction(row[index])) for row in observation.jet_rows]
    if y_col is None or not observation.sample_x:
        return None
    import numpy as np
    from omnibias.symbolic.discovery import extract_neural_jets, fit_neural_field_1d
    from omnibias.symbolic.ingest import pack_jet_bundle

    xs = np.asarray([float(Fraction(item)) for item in observation.sample_x], dtype=float)
    ys = np.asarray(y_col, dtype=float)
    field = fit_neural_field_1d(xs, ys, hidden=hidden, ridge=1e-4, seed=0)
    bundle = extract_neural_jets(field, xs, max_order=2)
    packed = pack_jet_bundle(bundle)
    return _run_neural_jet(packed, denom_bound=denom_bound)


def _run_pinn(
    observation: Observation,
    *,
    denom_bound: int,
    pinn_train: bool,
    steps: int,
) -> dict[str, Any] | None:
    from omnibias.symbolic.propose import propose_from_residual

    residual_raw = observation.extra_map().get("residual")
    if pinn_train:
        try:
            from omnibias.pinn.jax.discovery.observation import fit_residual_and_pack
        except ImportError:
            fit_residual_and_pack = None
        if fit_residual_and_pack is not None:
            observation = fit_residual_and_pack(observation, steps=steps)
            residual_raw = observation.extra_map().get("residual")
    if residual_raw:
        values = [float(Fraction(part)) for part in residual_raw.replace(";", ",").split(",") if part]
        if observation.design and observation.target and observation.term_names:
            coeffs = [0.0] * len(observation.term_names)
            if "u_xx" in observation.term_names:
                coeffs[observation.term_names.index("u_xx")] = 1.0
            return propose_from_residual(
                values,
                design=[[float(Fraction(item)) for item in row] for row in observation.design],
                coeffs=coeffs,
                names=observation.term_names,
                target=[float(Fraction(item)) for item in observation.target],
                denom_bound=denom_bound,
            )
        return propose_from_residual(values)
    return None


def run_bilevel_class_loop(
    observation: Observation,
    *,
    memory: ClassMemory | None = None,
    gate: Any | None = None,
    denom_bound: int = 32,
    sorts: Sequence[ConditionSort] | None = None,
    budget: int | None = None,
    inner_budget: int | None = None,
    proposers: Sequence[str] = ("stlsq", "neural_jet"),
    fit_field: bool = False,
    field_hidden: int = 8,
    pinn_train: bool = False,
    pinn_steps: int = 2,
    grow: bool = True,
) -> dict[str, Any]:
    """Inner proposers then outer :func:`select_class`. Snap is the accept gate."""

    inner: dict[str, Any] | None = None
    names = tuple(proposers)
    if fit_field and "field" not in names:
        names = names + ("field",)
    if pinn_train and "pinn" not in names:
        names = names + ("pinn",)
    for name in names:
        payload: dict[str, Any] | None = None
        if name == "stlsq":
            payload = _run_stlsq(observation, denom_bound=denom_bound)
        elif name == "neural_jet":
            payload = _run_neural_jet(observation, denom_bound=denom_bound)
        elif name == "field":
            payload = _run_field(observation, denom_bound=denom_bound, hidden=field_hidden)
        elif name == "pinn":
            payload = _run_pinn(
                observation,
                denom_bound=denom_bound,
                pinn_train=pinn_train,
                steps=pinn_steps,
            )
        if payload is None:
            continue
        inner = payload
        if payload.get("mode") == "exact_search":
            break
    selection = select_class(
        observation,
        collect=True,
        memory=memory,
        gate=gate,
        grammar_complete=False,
        sorts=sorts,
        budget=budget,
        inner_budget=inner_budget,
        grow=grow,
    )
    return {
        "inner": inner,
        "selection": selection,
        "honesty": dict(selection.honesty),
        "best": None if selection.best is None else selection.best.sort,
        "mode": None if inner is None else inner.get("mode"),
        "grammar_complete": False,
        "no_condition_exists_claim": False,
    }


def discover_observation(
    raw: Observation | Mapping[str, Any],
    *,
    memory: ClassMemory | None = None,
    memory_path: str | None = None,
    grow: bool = True,
    proposers: Sequence[str] = ("stlsq", "neural_jet"),
    fit_field: bool = False,
    pinn_train: bool = False,
    sorts: Sequence[ConditionSort] | None = None,
    budget: int | None = None,
    inner_budget: int | None = None,
) -> dict[str, Any]:
    """Pack if needed, load the stack, run the bilevel class loop."""

    load_discovery_stack()
    if isinstance(raw, Observation):
        observation = raw
    else:
        observation = Observation.from_dict(raw)
    if memory is None and memory_path:
        try:
            memory = ClassMemory.load(memory_path)
        except FileNotFoundError:
            memory = ClassMemory()
    payload = run_bilevel_class_loop(
        observation,
        memory=memory,
        proposers=proposers,
        fit_field=fit_field,
        pinn_train=pinn_train,
        sorts=sorts,
        budget=budget,
        inner_budget=inner_budget,
        grow=grow,
    )
    if memory is not None and memory_path:
        memory.save(memory_path)
    return payload


__all__ = [
    "discover_observation",
    "load_discovery_stack",
    "run_bilevel_class_loop",
]
