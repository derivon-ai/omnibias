# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lazy per-order cache for ``sigma^(n)(z)`` values.

A typical PINN residual touches several derivative orders of the same
pre-activation ``z = W x + beta``. The naive code path re-evaluates the
fast-path ``spec.fastpath(z, n)`` every time which is wasteful: each call
walks the same recurrence (Eulerian / Hermite / Stirling polynomials).

The :class:`SigmaCache` is a small memoiser: each
``cache.get_or_compute(order, build_fn)`` evaluates ``build_fn(order)`` on
first access and reuses the result thereafter. The cache is per-pass --
construct one inside :class:`FieldState` per ``field(coords)`` call, so we
avoid any cross-batch cache coherency issue.

The cache holds *backend tensors* (``torch.Tensor`` or ``jax.Array``); the
class itself has no dependency on either backend. It is just an
``order: int -> tensor`` dict with a typed accessor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


def _tensor_version(z: object) -> int | None:
    """Return a torch tensor's in-place mutation counter, else ``None``.

    Torch increments ``Tensor._version`` on every in-place op (autograd uses
    it to detect aliasing). We reuse it as a cheap O(1) tripwire. jax arrays
    and numpy arrays are immutable / have no such counter, so the guard is a
    no-op for them (``None``).
    """
    version = getattr(z, "_version", None)
    return version if isinstance(version, int) else None


class SigmaCache(Generic[T]):
    """Order-keyed cache for the activation derivative tower.

    Parameters
    ----------
    z
        The pre-activation tensor ``z = W x + beta``. Stored as the
        ``z`` attribute and reused by every :meth:`get_or_compute` call.

    Notes
    -----
    The cache is intentionally mutable so that ops can fill it lazily.
    The enclosing :class:`FieldState` is itself frozen, so the *handle*
    to the cache cannot be reassigned, but its contents can grow during
    a single ``field(coords)`` evaluation. This matches the pattern of a
    ``functools.cached_property`` but with explicit type-safe accessors.

    **Immutability contract.** Every cached value is a pure function of ``z``
    captured at construction. ``z`` (and the ``coords`` it is derived from)
    must therefore be treated as **immutable** for the lifetime of the cache:
    mutating it in place silently invalidates every cached ``sigma^(n)(z)``.
    For torch tensors this is enforced cheaply via ``Tensor._version`` -- a
    read after an in-place mutation of ``z`` raises :class:`RuntimeError`
    rather than returning stale derivatives. Rebuild the state from the new
    coords instead of mutating in place.
    """

    __slots__ = ("z", "_store", "_z_version")

    def __init__(self, z: T) -> None:
        self.z: T = z
        self._store: dict[int, T] = {}
        self._z_version: int | None = _tensor_version(z)

    def _assert_z_fresh(self) -> None:
        """Raise if ``z`` was mutated in place since construction (torch only)."""
        if self._z_version is None:
            return
        current = _tensor_version(self.z)
        if current != self._z_version:
            raise RuntimeError(
                "SigmaCache.z was mutated in place after construction "
                f"(version {self._z_version} -> {current}); cached "
                "sigma^(n)(z) values are now stale. coords / z must be treated "
                "as immutable for the lifetime of a FieldState -- rebuild the "
                "state from the new coords instead of mutating in place."
            )

    def has(self, order: int) -> bool:
        return order in self._store

    def get_or_compute(self, order: int, build: Callable[[int], T]) -> T:
        """Return ``sigma^(order)(z)``, computing it on first access."""
        if order < 0:
            raise ValueError(f"derivative order must be >= 0, got {order}")
        self._assert_z_fresh()
        cached = self._store.get(order)
        if cached is None:
            cached = build(order)
            self._store[order] = cached
        return cached

    def put(self, order: int, value: T) -> None:
        """Manual insert; useful when an op already knows the value."""
        if order < 0:
            raise ValueError(f"derivative order must be >= 0, got {order}")
        self._assert_z_fresh()
        self._store[order] = value

    def get(self, order: int) -> T:
        """Get a cached value, raising :class:`KeyError` if absent."""
        self._assert_z_fresh()
        if order not in self._store:
            raise KeyError(
                f"order {order} not in sigma cache; "
                f"available orders: {sorted(self._store)}"
            )
        return self._store[order]

    def orders(self) -> tuple[int, ...]:
        """Tuple of derivative orders currently cached."""
        return tuple(sorted(self._store))

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, order: object) -> bool:
        return isinstance(order, int) and order in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"SigmaCache(orders={self.orders()})"


__all__ = ["SigmaCache"]
