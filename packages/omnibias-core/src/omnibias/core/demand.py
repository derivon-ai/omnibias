# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Which derivative quantities a PDE residual reads from a field.

A residual is a function of the field value and a few of its derivatives at a point. Which structured program can
serve it depends on exactly which of those it reads: a residual that reads only the squared gradient norm
``|S^T grad u|^2`` can be served by a first-layer projection, one that reads the full gradient vector cannot.
Declaring the demand makes that a fact the planner can use instead of a property of a callback's signature.

Beyond the value, gradient and Laplacian the vocabulary names the quantities the stacked stream engine provides:
``hess_form`` (``tr(A hess u)`` for any symmetric ``A``, indefinite ones included), ``mixed`` (second partials
``d^alpha u``), ``third`` and ``fourth`` (partials of total order three and four, e.g. ``u_xxx`` of KdV or ``u_xxyy``),
and ``lap_grad`` (the gradient of the Laplacian).  The data a name is defined by (the matrix ``A``, the multi-indices
``alpha``) travels in a :class:`QuantitySpec`.  A drift ``directional`` term may use per-point directions; that is an
argument of the engine, not a separate name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

__all__ = [
    "DERIVATIVE_ORDER",
    "Demand",
    "QUANTITIES",
    "Quantities",
    "QuantitySpec",
    "Residual",
    "block_order",
    "legacy_full_gradient",
    "legacy_projected",
    "partial_requests",
]

#: Quantity name to the highest derivative order it needs.
DERIVATIVE_ORDER: dict[str, int] = {
    "value": 0,
    "grad": 1,
    "grad_sq": 1,
    "directional": 1,
    "lap": 2,
    "hess_form": 2,
    "mixed": 2,
    "third": 3,
    "lap_grad": 3,
    "bilap": 4,
    "fourth": 4,
}

QUANTITIES: tuple[str, ...] = tuple(DERIVATIVE_ORDER)


@dataclass(frozen=True)
class Demand:
    """A set of quantities read by a residual.

    ``value`` is always read. ``grad`` is the full gradient over all inputs; ``grad_sq`` is the metric norm
    ``|S^T grad u|^2`` on the selected inputs; ``lap`` is ``tr(S S^T hess u)``; ``bilap`` is ``lap`` applied twice;
    ``directional`` is ``v . grad u`` along fixed directions; ``hess_form`` is ``tr(A hess u)`` for a symmetric ``A``
    (any sign); ``mixed``, ``third`` and ``fourth`` are the partials ``d^alpha u`` of total order two, three and four
    named by the multi-indices of a :class:`QuantitySpec`; ``lap_grad`` is ``grad(lap u)``.
    """

    quantities: frozenset[str]

    def __post_init__(self) -> None:
        names = frozenset(self.quantities) | {"value"}
        unknown = names - set(QUANTITIES)
        if unknown:
            raise ValueError(f"unknown quantities {sorted(unknown)}; choose from {QUANTITIES}")
        object.__setattr__(self, "quantities", names)

    @classmethod
    def of(cls, *names: str) -> Demand:
        return cls(frozenset(names))

    @classmethod
    def from_iterable(cls, names: Iterable[str]) -> Demand:
        return cls(frozenset(names))

    def __contains__(self, name: object) -> bool:
        return name in self.quantities

    @property
    def order(self) -> int:
        """The highest derivative order any demanded quantity needs (0 to 4)."""
        return max(DERIVATIVE_ORDER[q] for q in self.quantities)

    @property
    def needs_full_gradient(self) -> bool:
        """The full gradient vector is read, so the input directions cannot be replaced by a projection."""
        return "grad" in self.quantities

    @property
    def projectable(self) -> bool:
        """Every demanded quantity is a function of projected streams (norms, Laplacians, directional terms)."""
        return self.quantities <= _PROJECTED_ENGINE

    def union(self, other: Demand) -> Demand:
        return Demand(self.quantities | other.quantities)

    def to_config(self) -> list[str]:
        return sorted(self.quantities, key=QUANTITIES.index)


_FULL_ENGINE = frozenset({"value", "grad", "grad_sq", "lap"})
_PROJECTED_ENGINE = frozenset({"value", "grad_sq", "lap", "bilap", "directional"})
_STREAMS_ENGINE = frozenset(DERIVATIVE_ORDER)


class Quantities(NamedTuple):
    """The derivative quantities of a field at one block of points; undeclared ones are ``None``."""

    u: Any
    x: Any
    grad: Any | None = None
    grad_sq: Any | None = None
    lap: Any | None = None
    bilap: Any | None = None
    directional: Any | None = None
    hess_form: Any | None = None
    mixed: Any | None = None
    third: Any | None = None
    lap_grad: Any | None = None
    fourth: Any | None = None


_HIDDEN = tuple(name for name in Quantities._fields if name not in ("u", "x"))


@dataclass(frozen=True)
class QuantitySpec:
    """The data the quantities ``hess_form``, ``mixed``, ``third`` and ``fourth`` are defined by.

    ``hess_form`` is a symmetric ``(m, m)`` matrix ``A`` (an indefinite one is fine: wave, Klein-Gordon, mixed second
    partials) over the ``m`` selected inputs.  ``mixed``, ``third`` and ``fourth`` are tuples of multi-indices: a
    length-``D`` tuple of per-axis orders, ``(3, 0)`` for ``u_xxx`` or ``(2, 2)`` for ``u_xxyy``, whose total order is two,
    three and four respectively.  The corresponding outputs list the partials in the order given.
    """

    hess_form: Any | None = None
    mixed: tuple[tuple[int, ...], ...] = ()
    third: tuple[tuple[int, ...], ...] = ()
    fourth: tuple[tuple[int, ...], ...] = ()

    _ORDERS = {"mixed": 2, "third": 3, "fourth": 4}

    def __post_init__(self) -> None:
        for name, order in self._ORDERS.items():
            alphas = tuple(tuple(int(k) for k in a) for a in getattr(self, name))
            for a in alphas:
                if min(a, default=0) < 0 or sum(a) != order:
                    raise ValueError(f"{name} multi-indices need total order {order}, got {a}")
            object.__setattr__(self, name, alphas)

    def check(self, demand: Demand, dim: int | None = None) -> None:
        """Every demanded quantity that needs data has it, and data is given only for what is demanded."""
        for name in ("hess_form", *self._ORDERS):
            given = getattr(self, name)
            has = given is not None if name == "hess_form" else bool(given)
            if name in demand and not has:
                raise ValueError(f"the residual reads {name!r}: the QuantitySpec must define it")
            if has and name not in demand:
                raise ValueError(
                    f"the QuantitySpec defines {name!r} but the residual does not read it"
                )
        if dim is not None:
            for name in self._ORDERS:
                for a in getattr(self, name):
                    if len(a) != dim:
                        raise ValueError(f"{name} multi-index {a} must have length {dim}")


class Residual:
    """A pointwise residual ``fn(Quantities) -> (B,) | (B, R)`` that reads exactly the quantities in ``demand``."""

    __slots__ = ("demand", "fn")

    def __init__(self, fn: Callable[[Quantities], Any], demand: Iterable[str] | Demand) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn
        self.demand = demand if isinstance(demand, Demand) else Demand.from_iterable(demand)

    @property
    def order(self) -> int:
        """The derivative order the residual needs (0 to 4)."""
        return self.demand.order

    def __call__(self, q: Quantities) -> Any:
        return self.fn(self.masked(q))

    def masked(self, q: Quantities) -> Quantities:
        """``q`` with every quantity outside the declared demand replaced by ``None``."""
        keep = self.demand.quantities
        fields: dict[str, Any] = {name: getattr(q, name) for name in Quantities._fields}
        for name in _HIDDEN:
            if name not in keep:
                fields[name] = None
        return Quantities(**fields)

    def with_rows(self, transform: Callable[[Any], Any]) -> Residual:
        """The residual whose output is ``transform(output)`` (row weights, masks) with the same demand."""
        fn = self.fn
        return Residual(lambda q: transform(fn(q)), self.demand)


def _check_order(residual: Residual, order: int | None, engine: str) -> int:
    served = {"full": _FULL_ENGINE, "projected": _PROJECTED_ENGINE, "streams": _STREAMS_ENGINE}.get(
        engine
    )
    if served is None:
        raise ValueError("engine must be 'full', 'projected' or 'streams'")
    extra = residual.demand.quantities - served
    if extra:
        only_streams = extra - _FULL_ENGINE - _PROJECTED_ENGINE
        hints = []
        if extra - only_streams:
            hints.append(
                "use the first-layer projected engine"
                if engine == "full"
                else "the full gradient is not formed; use the full-gradient engine, or read grad_sq"
            )
        if only_streams:
            hints.append("the stacked stream engine provides " + ", ".join(sorted(only_streams)))
        raise ValueError(f"the {engine} engine cannot provide {sorted(extra)}: {'; '.join(hints)}")
    needed = residual.order
    if order is None:
        order = needed
    if order < needed:
        raise ValueError(
            f"the residual reads derivatives of order {needed}; the block asks for order {order}"
        )
    if engine == "full" and order not in (0, 1, 2):
        raise ValueError("the full-gradient engine serves blocks of order 0, 1 or 2")
    if engine == "projected" and order not in (0, 1, 2, 4):
        raise ValueError("the projected engine serves blocks of order 0, 1, 2 or 4")
    if engine == "streams" and order not in (0, 1, 2, 3, 4):
        raise ValueError("the stream engine serves blocks of order 0 to 4")
    return order


def block_order(residual: Residual, order: int | None, engine: str) -> int:
    """The engine order of a block holding ``residual`` (inferred from its demand when ``order`` is ``None``)."""
    return _check_order(residual, order, engine)


def legacy_full_gradient(fn: Callable[[Any, Any, Any, Any], Any], *, order: int = 2) -> Residual:
    """A :class:`Residual` from an old ``fn(u, grad, lap, x)`` callback of the given block order."""
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    names = ["value", "grad", "lap"][: order + 1] if order else ["value"]
    return Residual(lambda q: fn(q.u, q.grad, q.lap, q.x), names)


def legacy_projected(
    fn: Callable[..., Any], *, order: int = 2, directional: bool = False
) -> Residual:
    """A :class:`Residual` from an old projected callback.

    ``order`` 0, 1, 2 wrap ``fn(u, grad_sq, lap, directional, x)``; ``order=4`` wraps
    ``fn(u, grad_sq, lap, bilap, directional, x)``. ``directional`` declares that the callback reads the drift term.
    """
    if order not in (0, 1, 2, 4):
        raise ValueError("order must be 0, 1, 2 or 4")
    names = ["value"]
    if order >= 1:
        names.append("grad_sq")
    if order >= 2:
        names.append("lap")
    if order == 4:
        names.append("bilap")
    if directional:
        names.append("directional")
    assert all(n in DERIVATIVE_ORDER for n in names)
    if order == 4:
        return Residual(lambda q: fn(q.u, q.grad_sq, q.lap, q.bilap, q.directional, q.x), names)
    if order == 0:
        return Residual(lambda q: fn(q.u, None, None, None, q.x), names)
    return Residual(lambda q: fn(q.u, q.grad_sq, q.lap, q.directional, q.x), names)


def partial_requests(
    demand: Demand,
    spec: QuantitySpec,
    dim: int,
    coordinates: Sequence[int] | None = None,
) -> tuple[list[tuple[int, ...]], dict[str, tuple[int, int]], tuple[float, ...]]:
    """The multi-indices of every partial a demand reads, the slice each quantity occupies, and the ``bilap`` weights.

    ``mixed``, ``third`` and ``fourth`` list the spec's multi-indices; ``lap_grad`` lists the ``D^2`` third partials
    ``d_j^2 d_i`` whose sums over ``j`` form the gradient of the Laplacian; ``bilap`` lists ``u_iiii`` and ``u_iijj``
    over ``coordinates`` (all inputs by default) with weights ``1`` and ``2``. One engine pass over the concatenated
    list carries each polarization direction once.
    """
    alphas: list[tuple[int, ...]] = []
    where: dict[str, tuple[int, int]] = {}
    weights: tuple[float, ...] = ()

    def add(name: str, items: Sequence[tuple[int, ...]]) -> None:
        where[name] = (len(alphas), len(alphas) + len(items))
        alphas.extend(items)

    for name in ("mixed", "third", "fourth"):
        if name in demand:
            add(name, getattr(spec, name))
    if "lap_grad" in demand:
        add(
            "lap_grad",
            [
                tuple(2 * (k == j) + (k == i) for k in range(dim))
                for i in range(dim)
                for j in range(dim)
            ],
        )
    if "bilap" in demand:
        idx = list(range(dim)) if coordinates is None else [int(c) for c in coordinates]
        items = [tuple(4 * (k == i) for k in range(dim)) for i in idx]
        w = [1.0] * len(idx)
        for p, i in enumerate(idx):
            for j in idx[p + 1 :]:
                items.append(tuple(2 * (k == i) + 2 * (k == j) for k in range(dim)))
                w.append(2.0)
        add("bilap", items)
        weights = tuple(w)
    return alphas, where, weights
