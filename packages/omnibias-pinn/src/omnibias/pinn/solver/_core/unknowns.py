# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Unknown PDE coefficients: the descriptor that makes inverse problems first-class.

A forward solve knows every coefficient; an *inverse* solve treats one or more of
them as unknowns to be recovered from observations. The obstacle is that a
:class:`~omnibias.pinn.solver._core.system.System` is frozen and its residuals are
plain closures over Python floats, so there is no path for a gradient to reach a
coefficient.

:class:`Unknown` is the fix, and it is deliberately a *descriptor* rather than a
tensor: ``_core`` stays backend-free, and one ``System`` therefore serves both
modes. A coefficient is read through :func:`coefficient`, which resolves an
``Unknown`` against the current **binding** -- so

* inside :func:`bind_unknowns` the residual sees whatever the caller bound (a
  ``torch`` parameter during an inverse solve, or the ground-truth float when
  generating synthetic observations), and
* outside any binding the coefficient is *unbound*, which the forward drivers
  refuse rather than silently solving at the initial guess.

The binding lives in a :class:`~contextvars.ContextVar`, so it is scoped to the
``with`` block, restored on exit, and independent per thread / per asyncio task --
nested solves cannot leak values into each other.

Transforms
----------
Physical coefficients are usually constrained (a diffusivity is positive, a
volume fraction lies in ``[0, 1]``), and an optimiser that is free to step
negative will happily produce a nonsense residual that still descends. Each
``Unknown`` therefore carries a transform mapping an *unconstrained* raw
parameter to the constrained value:

===============  ============================================  ===============
``transform``    value from raw                                domain
===============  ============================================  ===============
``"none"``       ``raw``                                       all reals
``"positive"``   ``softplus(raw)``                             ``> 0``
``"bounded"``    ``lower + (upper - lower) * sigmoid(raw)``     ``(lower, upper)``
===============  ============================================  ===============

The optimiser then works on an unconstrained variable and the constraint holds by
construction -- no projection step, no clipping, and the gradient stays exact.
:meth:`Unknown.to_raw` is the exact inverse, used once to initialise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from omnibias.pinn.solver._core.arrays import array_namespace

#: Transform names accepted by :class:`Unknown`.
TRANSFORMS = frozenset({"none", "positive", "bounded"})

#: Read-only empty default, so the ContextVar cannot hand out a shared mutable.
_NOTHING: Mapping[str, Any] = MappingProxyType({})

#: The active ``name -> value`` binding. Empty means "no inverse solve running".
_BINDING: ContextVar[Mapping[str, Any]] = ContextVar(
    "omnibias_pinn_solver_unknowns", default=_NOTHING
)

#: Largest raw value handled by the exact ``softplus`` inverse before it is
#: indistinguishable from the identity in float64.
_SOFTPLUS_LINEAR = 30.0


@dataclass(frozen=True)
class Unknown:
    """A PDE coefficient to be recovered instead of supplied.

    Parameters
    ----------
    name
        Identifier, unique within a system. It keys the binding and labels the
        recovered value in the solution.
    initial
        Starting guess, in *physical* units (not raw units). It must satisfy the
        transform's constraint, which is checked here rather than discovered as a
        ``nan`` several hundred iterations into a solve.
    transform
        One of :data:`TRANSFORMS`; see the module docstring.
    lower, upper
        Bounds for ``transform="bounded"``; rejected otherwise.
    """

    name: str
    initial: float = 1.0
    transform: str = "none"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            # The torch driver exposes the raw parameters through a ParameterDict,
            # whose keys become dotted paths in ``named_parameters()``; a name with
            # a dot in it would silently break the functional optimiser path.
            raise ValueError(
                f"Unknown name must be a Python identifier, got {self.name!r}"
            )
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"unknown transform {self.transform!r}; "
                f"expected one of {sorted(TRANSFORMS)}"
            )
        if not math.isfinite(self.initial):
            raise ValueError(f"{self.name!r}: initial must be finite")
        if self.transform == "bounded":
            if self.lower is None or self.upper is None:
                raise ValueError(
                    f"{self.name!r}: transform='bounded' requires lower and upper"
                )
            if not self.lower < self.upper:
                raise ValueError(
                    f"{self.name!r}: need lower < upper, got "
                    f"{self.lower} >= {self.upper}"
                )
            if not self.lower < self.initial < self.upper:
                raise ValueError(
                    f"{self.name!r}: initial {self.initial} is not strictly inside "
                    f"({self.lower}, {self.upper})"
                )
        else:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    f"{self.name!r}: lower/upper only apply to transform='bounded'"
                )
            if self.transform == "positive" and self.initial <= 0.0:
                raise ValueError(
                    f"{self.name!r}: transform='positive' requires initial > 0, "
                    f"got {self.initial}"
                )

    # -- transform ----------------------------------------------------

    def from_raw(self, raw: Any) -> Any:
        """Map an unconstrained raw parameter to the constrained value.

        Works on a float or on a backend tensor; the tensor path resolves its
        namespace at call time so ``_core`` keeps importing no backend.
        """
        if self.transform == "none":
            return raw
        xp = array_namespace(raw)
        if self.transform == "positive":
            # softplus, via the log1p(exp(-|x|)) + max(x, 0) rearrangement that
            # never exponentiates a large positive number.
            return xp.log1p(xp.exp(-abs(raw))) + xp.maximum(
                raw, xp.zeros_like(raw)
            )
        lo, hi = float(self.lower or 0.0), float(self.upper or 0.0)
        return lo + (hi - lo) / (1.0 + xp.exp(-raw))

    def to_raw(self, value: float) -> float:
        """Exact inverse of :meth:`from_raw` on a physical float."""
        v = float(value)
        if self.transform == "none":
            return v
        if self.transform == "positive":
            if v <= 0.0:
                raise ValueError(f"{self.name!r}: positive transform needs value > 0")
            # log(expm1(v)) overflows for large v where softplus is the identity.
            return v if v > _SOFTPLUS_LINEAR else math.log(math.expm1(v))
        lo, hi = float(self.lower or 0.0), float(self.upper or 0.0)
        if not lo < v < hi:
            raise ValueError(
                f"{self.name!r}: value {v} is not strictly inside ({lo}, {hi})"
            )
        t = (v - lo) / (hi - lo)
        return math.log(t / (1.0 - t))

    def initial_raw(self) -> float:
        """The raw parameter value that reproduces :attr:`initial`."""
        return self.to_raw(self.initial)

    # -- binding ------------------------------------------------------

    def is_bound(self) -> bool:
        return self.name in _BINDING.get()

    def resolve(self) -> Any:
        """The currently bound value.

        Raises
        ------
        LookupError
            If no value is bound. Returning :attr:`initial` instead would let a
            forward driver quietly solve the wrong problem, which is the one
            failure mode this whole module exists to prevent.
        """
        binding = _BINDING.get()
        try:
            return binding[self.name]
        except KeyError:
            raise LookupError(
                f"unknown coefficient {self.name!r} is not bound; run an inverse "
                "driver (solve_inverse) or pin it with bind_unknowns({...})"
            ) from None


#: A PDE coefficient: a known constant, or an :class:`Unknown` to recover.
Coefficient = float | Unknown


def coefficient(value: Coefficient) -> Any:
    """Resolve a coefficient inside a residual closure.

    A float passes straight through, so a forward system pays nothing for this
    indirection; an :class:`Unknown` becomes the bound tensor, which is what puts
    the coefficient on the autograd graph.
    """
    return value.resolve() if isinstance(value, Unknown) else value


def collect_unknowns(*values: Any) -> tuple[Unknown, ...]:
    """The distinct :class:`Unknown` descriptors among ``values``, in order.

    Nested tuples / lists (``diffusivities``, a vector ``velocity``) are walked.
    Two *different* descriptors sharing a name is a bug -- the binding is keyed by
    name, so one of them would silently take the other's value.
    """
    found: dict[str, Unknown] = {}
    for value in _flatten(values):
        if not isinstance(value, Unknown):
            continue
        seen = found.get(value.name)
        if seen is None:
            found[value.name] = value
        elif seen != value:
            raise ValueError(
                f"two different unknowns share the name {value.name!r}: "
                f"{seen!r} and {value!r}"
            )
    return tuple(found.values())


def _flatten(values: Iterable[Any]) -> Iterator[Any]:
    for value in values:
        if isinstance(value, tuple | list):
            yield from _flatten(value)
        else:
            yield value


@contextmanager
def bind_unknowns(values: Mapping[str, Any]) -> Iterator[None]:
    """Bind unknown coefficients for the duration of the block.

    The values may be floats (pin a system at known coefficients to generate
    synthetic observations) or backend tensors (what ``solve_inverse`` binds, so
    the residual gradient reaches the coefficient). Bindings nest: an inner block
    overrides only the names it mentions.
    """
    token = _BINDING.set({**_BINDING.get(), **values})
    try:
        yield
    finally:
        _BINDING.reset(token)


def bound_names() -> frozenset[str]:
    """Names bound in the current context (diagnostics and driver guards)."""
    return frozenset(_BINDING.get())


__all__ = [
    "Coefficient",
    "TRANSFORMS",
    "Unknown",
    "bind_unknowns",
    "bound_names",
    "coefficient",
    "collect_unknowns",
]
