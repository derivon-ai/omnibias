# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Describe a third-party derivative engine to the planner.

A planner can only rank what it can price and refuse what it can recognise. :func:`engine_algorithm` turns the description of
an external engine (a name, the operator it serves, what it cannot do, and a work model) into the :class:`Algorithm` the planner
registers next to its own programs; the back ends (``omnibias.jax.operators.register_engine``,
``omnibias.torch.operators.register_engine``) add the executor that calls the engine. The engine then competes under the same
rules as a built-in: it is refused with a reason when it does not apply, ranked by the cost model, timed by ``mode="find"``,
named in ``explain()`` and checked by :mod:`omnibias.core.conformance`.

The engine contract is deliberately narrow. For ``"laplacian"`` the engine is called as ``engine(f, x, power=p)`` with the
pointwise model ``f`` (one point in, one scalar out) and a batch of points ``x``; it returns the ``p``-th Laplace power at
each point. For ``"divergence"`` ``f`` returns a vector of the input's length and the engine returns its divergence per
point. Metrics, coordinate subsets and drift directions are not forwarded, so an engine is always refused for a request that
uses them, with that reason.

The work model is not guessed. ``work`` is either a function ``descriptor -> Work`` or the name of a registered algorithm
whose work model the engine is priced like (for instance ``"forward_laplacian"`` for a forward-mode Laplacian engine): the
ranking then rests on the declared model, and ``explain()`` shows it. ``CostModel.efficiency`` calibrates the engine
per device exactly as it does the built-in programs.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core.planner import Algorithm, Descriptor, Work, algorithms_for

__all__ = ["SUPPORTED_OPERATORS", "engine_algorithm"]

#: Operators an external engine can serve through the narrow contract above.
SUPPORTED_OPERATORS = ("laplacian", "divergence")


def _borrowed_work(name: str, operator: str) -> Callable[[Descriptor], Work]:
    for algorithm in algorithms_for(operator):
        if algorithm.name == name:
            return algorithm.work
    known = sorted(a.name for a in algorithms_for(operator))
    raise ValueError(
        f"no registered algorithm {name!r} for {operator!r} to borrow a work model from; known: {known}"
    )


def engine_algorithm(
    name: str,
    operator: str,
    *,
    work: Callable[[Descriptor], Work] | str,
    backend: str,
    requires: Callable[[Descriptor], str | None] | None = None,
    max_power: int = 1,
    exact: bool = True,
    generic: bool = False,
) -> Algorithm:
    """The planner entry for an external engine; see the module docstring for the contract."""
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"engines can serve {SUPPORTED_OPERATORS}, not {operator!r}")
    if not name or not name.isidentifier():
        raise ValueError(f"engine name must be an identifier, got {name!r}")
    if max_power < 1:
        raise ValueError("max_power must be >= 1")
    work_fn = _borrowed_work(work, operator) if isinstance(work, str) else work
    if not callable(work_fn):
        raise TypeError("work must be a callable or the name of a registered algorithm")

    def applies(descriptor: Descriptor) -> str | None:
        request = descriptor.request
        if request.kind == "laplacian" and request.power > max_power:
            return f"engine {name!r} serves Laplace powers up to {max_power}, not {request.power}"
        if request.metric != "euclidean" or request.selected is not None or request.directions:
            return (
                f"engine {name!r} does not take a metric, a coordinate subset or drift directions"
            )
        return None if requires is None else requires(descriptor)

    return Algorithm(
        name=name,
        operators=frozenset({operator}),
        requires=applies,
        work=work_fn,
        backend=backend,
        exact=exact,
        generic=generic,
    )
