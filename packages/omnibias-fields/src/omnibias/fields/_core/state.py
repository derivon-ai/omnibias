# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic ``FieldState`` schema.

A :class:`FieldState` is the *value object* produced by
``field.evaluate(coords)`` (alias ``field(coords)``). It bundles together:

- The runtime input tensor ``coords`` (shape ``(B, D)``).
- A back-reference to the typed field that produced it (used by ops to
  read the parameters / activation spec).
- The :class:`ComponentSpec` and :class:`CoordinateSpec`.
- A backend-specific ``ops`` module reference (so the views can delegate
  attribute access into the right ``ops.*`` kernel).
- A :class:`SigmaCache` shared across all derivative-order calls inside
  the residual.

Crucial design point: every public op consumes a ``FieldState`` rather
than ``(u, coords)`` tensors. This is what keeps the closed-form path
*actually* closed-form -- all chain-rule helpers know which field they
are differentiating and reuse cached pre-activations.

The class is implemented in pure Python (no torch / jax) and lives in
:mod:`omnibias.fields._core` so the torch and jax backends share the same
attribute-DSL surface by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.view import ComponentView, VectorView, did_you_mean

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    from omnibias.fields._core.field_base import FieldBase

T = TypeVar("T")


class FieldState(Generic[T]):
    """Frozen evaluation snapshot of a typed PINN field.

    Parameters
    ----------
    coords
        Input tensor of shape ``(B, D)``. Owned by the caller of
        ``field(coords)``; the state stores a reference (no copy). It must be
        treated as **immutable** for the lifetime of the state: the
        :class:`SigmaCache` memoises ``sigma^(n)(z)`` against the pre-activation
        ``z`` derived from these coords, so mutating ``coords`` (or ``z``) in
        place silently desynchronises the cached derivatives from freshly
        recomputed values. In-place mutation of a torch ``z`` is caught by the
        cache (raises on the next read); rebuild the state from new coords
        rather than mutating in place.
    field
        The :class:`FieldBase` instance that produced this state. Ops
        read ``field.spec`` (activation), ``field.params``, etc.
    components
        :class:`ComponentSpec` describing the output channels. Frozen.
    coordinate_spec
        :class:`CoordinateSpec` describing the input axes. Frozen.
    ops
        Backend-specific ``ops`` module (``omnibias.fields.torch._ops_dispatch``
        or its jax twin). Looked up by attribute access in the views.
    sigma_cache
        :class:`SigmaCache` keyed by derivative order. Filled lazily by
        whichever ops touch it first.
    extra
        Free-form per-state cache: ops can stash intermediate tensors
        (like a precomputed gradient) keyed by string. Mutable but
        intended for op-internal use.

        **Readout-independence contract.** A :class:`FieldState` is a
        coherent snapshot of one evaluation. Entries written by the field
        or its ops may depend only on ``coords`` and *frozen feature*
        parameters (hidden weights, temporal heads, geometry factors) --
        never on the *readout* parameters the frozen-feature linear
        solver sweeps (``c`` / ``b`` on a one-layer field, ``V`` / ``b_t``
        on a spectral / Chebyshev field, the final affine layer of a jet
        MLP). Fields that honour this contract declare it via the class
        attribute named by
        :data:`~omnibias.fields._core.field_base.READOUT_INDEPENDENT_ATTR`.

        Caller-installed entries are the caller's responsibility. The
        ``lim_along`` extension, for example, stashes user-supplied
        closures in ``extra["lim_along"]``; those closures are typically
        readout-dependent and unverifiable by the library, so installing
        them disqualifies the field from declaring readout-independence
        for any subsequent frozen-feature solve against that state.

    Notes
    -----
    The class behaves frozen-ish: the dataclass attributes are set in
    ``__init__`` and not changed afterwards. The :class:`SigmaCache`
    contents and the ``extra`` dict are intentionally mutable so ops can
    fill them lazily during a single residual evaluation.
    """

    __slots__ = (
        "coords",
        "field",
        "components",
        "coordinate_spec",
        "ops",
        "sigma_cache",
        "extra",
    )

    def __init__(
        self,
        *,
        coords: T,
        field: FieldBase,
        components: ComponentSpec,
        coordinate_spec: CoordinateSpec,
        ops: ModuleType,
        sigma_cache: SigmaCache[T],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(components, ComponentSpec):
            raise TypeError(
                f"components must be ComponentSpec, got {type(components).__name__}"
            )
        if not isinstance(coordinate_spec, CoordinateSpec):
            raise TypeError(
                f"coordinate_spec must be CoordinateSpec, got {type(coordinate_spec).__name__}"
            )
        if not isinstance(sigma_cache, SigmaCache):
            raise TypeError(
                f"sigma_cache must be SigmaCache, got {type(sigma_cache).__name__}"
            )
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "coordinate_spec", coordinate_spec)
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "sigma_cache", sigma_cache)
        object.__setattr__(self, "extra", {} if extra is None else dict(extra))

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__slots__:
            raise AttributeError(
                f"FieldState is frozen; cannot reassign {name!r}. Use .extra "
                "for per-state op caches."
            )
        raise AttributeError(f"FieldState has no attribute {name!r}")

    # -- attribute DSL ------------------------------------------------

    def __getattr__(self, name: str) -> ComponentView | VectorView:
        # __getattr__ is called only when ordinary lookup misses; the
        # __slots__ attributes (coords, field, ...) are handled by the
        # default descriptor protocol.
        comps = self.components
        if comps.is_component(name):
            return ComponentView(self, name)
        if comps.is_group(name):
            return VectorView(self, comps.group_members(name))
        # Provide a helpful message on typos.
        raise AttributeError(did_you_mean(name, comps))

    def __getitem__(self, name: str) -> ComponentView | VectorView:
        return self.__getattr__(name)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.components.has(name)

    @property
    def values(self) -> dict[str, T]:
        """One-shot dict snapshot of every scalar component value."""
        return {n: self.ops.value(self, n) for n in self.components.names}

    @property
    def axes(self) -> tuple[str, ...]:
        return self.coordinate_spec.axes

    @property
    def n_components(self) -> int:
        return self.components.n_components

    def __repr__(self) -> str:
        cached = self.sigma_cache.orders()
        try:
            shape: Sequence[int] | str = tuple(self.coords.shape)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover -- non-tensor coords
            shape = "?"
        return (
            f"FieldState(field={type(self.field).__name__}, "
            f"components={self.components.names!r}, "
            f"coords_shape={shape}, "
            f"cached_sigma_orders={cached})"
        )


__all__ = ["FieldState"]
