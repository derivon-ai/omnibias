# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The deprecation schedule of promoted names, as data.

A name leaves ``omnibias.<backend>.experimental`` for a stable module in three steps, recorded in one :class:`Alias`:

* ``since``: the first release in which the stable name exists. Until ``warn_from`` the old name is a silent alias for it:
  both resolve to the *same object*, so nothing that works keeps working differently.
* ``warn_from``: the first release in which importing or reading the old name raises a :class:`DeprecationWarning` that
  names the stable replacement and the removal release.
* ``remove_in``: the first release without the alias. The alias is kept for at least two minor releases after ``since``.

``warn_if_due`` compares the schedule with the installed release (the pre-release suffix is ignored, so ``0.6.0rc1`` already
follows the ``0.6.0`` schedule). The back ends register their aliases when their stable module is imported and route the old
module's attribute access through :func:`resolve`; this module has no framework dependency. The schedule is exposed through
``docs/stability.md`` and tested for consistency (``remove_in`` strictly after ``warn_from`` strictly after ``since``).
"""

from __future__ import annotations

import importlib
import re
import warnings
from dataclasses import dataclass
from typing import Any

__all__ = ["ALIASES", "Alias", "lookup", "register", "resolve", "version_tuple", "warn_if_due"]

_VERSION = re.compile(r"\d+(?:\.\d+)*")


def version_tuple(version: str) -> tuple[int, ...]:
    """``"0.5.0rc1"`` -> ``(0, 5, 0)``; a version with no leading number is rejected."""
    match = _VERSION.match(version.strip())
    if match is None:
        raise ValueError(f"not a version: {version!r}")
    return tuple(int(part) for part in match.group().split("."))


@dataclass(frozen=True)
class Alias:
    """``old`` and ``new`` are dotted ``module.attribute`` paths."""

    old: str
    new: str
    since: str
    warn_from: str
    remove_in: str

    def __post_init__(self) -> None:
        for path in (self.old, self.new):
            if "." not in path:
                raise ValueError(f"{path!r} must be a dotted module.attribute path")
        since, warn, remove = (
            version_tuple(v) for v in (self.since, self.warn_from, self.remove_in)
        )
        if not since <= warn < remove:
            raise ValueError(
                f"{self.old}: need since <= warn_from < remove_in, got {self.since}, {self.warn_from}, {self.remove_in}"
            )

    def due(self, current: str) -> bool:
        """Whether the old name warns in release ``current``."""
        return version_tuple(current) >= version_tuple(self.warn_from)

    def expired(self, current: str) -> bool:
        """Whether the old name should already be gone in release ``current``."""
        return version_tuple(current) >= version_tuple(self.remove_in)

    def message(self) -> str:
        return (
            f"{self.old} is deprecated since {self.warn_from} and will be removed in {self.remove_in}; "
            f"use {self.new}"
        )


ALIASES: dict[str, Alias] = {}


def register(alias: Alias) -> Alias:
    """Add ``alias``; registering the same schedule again is a no-op, a different one for the same name is an error."""
    known = ALIASES.get(alias.old)
    if known is not None and known != alias:
        raise ValueError(f"{alias.old} is already registered with a different schedule")
    ALIASES[alias.old] = alias
    return alias


def lookup(old: str) -> Alias | None:
    return ALIASES.get(old)


def warn_if_due(old: str, current: str, *, stacklevel: int = 3) -> bool:
    """Emit the :class:`DeprecationWarning` for ``old`` when ``current`` has reached its ``warn_from``."""
    alias = ALIASES.get(old)
    if alias is None or not alias.due(current):
        return False
    warnings.warn(alias.message(), DeprecationWarning, stacklevel=stacklevel)
    return True


def resolve(old: str, current: str) -> Any:
    """The object behind the old name (the stable one), warning when the schedule says so."""
    alias = ALIASES.get(old)
    if alias is None:
        raise AttributeError(old)
    warn_if_due(old, current, stacklevel=4)
    module, _, attr = alias.new.rpartition(".")
    return getattr(importlib.import_module(module), attr)
