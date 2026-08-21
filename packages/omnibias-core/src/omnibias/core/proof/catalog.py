# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Cross-package registry of discoverable statements.

Core never imports holonomic / combinatorics / symbolic / pinn. Owning packages
call :func:`register_catalog` at import time. :func:`list_catalog` therefore
shows only packages that have already been imported. :func:`discover` runs a
registered factory or tells the caller which package to import.

Modes:

* ``exact_search`` -- :func:`~omnibias.core.proof.discovery.run_discovery`
* ``exact_replay`` -- an existing ``verify_*`` (never a planted fake search)
* ``enclosure`` / ``empirical`` -- ProofMachine or residual payload; never an
  :class:`~omnibias.core.proof.discovery.ExactCheck` forged from a float residual
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from omnibias.core.proof.discovery import ParentStatus

CatalogMode = Literal["exact_search", "exact_replay", "enclosure", "empirical"]
CatalogFactory = Callable[..., Any]


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata for one discoverable kind. Factories live beside the entry."""

    kind: str
    obligation: str
    parent: str
    parent_status: ParentStatus
    package: str
    mode: CatalogMode
    complete: bool
    existential: bool = True


_ENTRIES: dict[str, CatalogEntry] = {}
_FACTORIES: dict[str, CatalogFactory] = {}


def register_catalog(
    entry: CatalogEntry,
    factory: CatalogFactory | None = None,
) -> None:
    """Register ``entry`` (and optionally its factory). Same entry is idempotent."""

    existing = _ENTRIES.get(entry.kind)
    if existing is not None and existing != entry:
        raise ValueError(
            f"catalog kind collision: {entry.kind!r} already registered as {existing}"
        )
    _ENTRIES[entry.kind] = entry
    if factory is not None:
        _FACTORIES[entry.kind] = factory


def list_catalog() -> tuple[CatalogEntry, ...]:
    """Entries from packages imported so far, sorted by kind."""

    return tuple(sorted(_ENTRIES.values(), key=lambda item: item.kind))


def catalog_entry(kind: str) -> CatalogEntry | None:
    return _ENTRIES.get(kind)


def discover(kind: str, **kwargs: Any) -> Any:
    """Run the registered factory for ``kind``.

    Raises
    ------
    KeyError
        No factory is registered. Import the owning package first.
    """

    factory = _FACTORIES.get(kind)
    if factory is None:
        entry = _ENTRIES.get(kind)
        package = entry.package if entry is not None else "the owning package"
        raise KeyError(f"no factory for {kind!r}; import {package}")
    return factory(**kwargs)


def _reset_catalog_for_tests() -> None:
    """Test helper. Not part of the public engine."""

    _ENTRIES.clear()
    _FACTORIES.clear()


__all__ = [
    "CatalogEntry",
    "CatalogFactory",
    "CatalogMode",
    "catalog_entry",
    "discover",
    "list_catalog",
    "register_catalog",
]
