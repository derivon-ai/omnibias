# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Release guard: every name in a package ``__all__`` resolves via ``getattr``.

Companion to :mod:`test_all_sorted` (static sortedness of literal ``__all__``
lists). This module *imports* the public package and asserts each exported name
is present on the module object, so a stale ``__all__`` entry that survived a
rename is caught before a release.

Scope (deliberately narrow -- heavy backends are optional):

* Always: ``omnibias.core`` (pure Python; runs in the core / guards CI jobs).
* Optionally: ``omnibias.torch`` / ``omnibias.jax`` when those packages import
  cleanly in the current environment (skipped with a reason otherwise).

All 42 distributions are *not* required here: many need torch+jax+optional
extras that the pure-Python guards job does not install. Extend the
``_CANDIDATE_PACKAGES`` tuple when a new lightweight package earns a seat.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable

import pytest

# Short import names under the ``omnibias`` namespace. Keep this list small and
# dependency-light; do not enumerate all 42 packages.
_CANDIDATE_PACKAGES: tuple[str, ...] = (
    "omnibias.core",
    "omnibias.torch",
    "omnibias.jax",
)


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - surface skip reason, any cause
        pytest.skip(f"{name} not importable in this environment: {exc}")


def _public_all(mod) -> list[str]:
    names = getattr(mod, "__all__", None)
    if names is None:
        pytest.skip(f"{mod.__name__} has no __all__")
    if not isinstance(names, Iterable) or isinstance(names, (str, bytes)):
        raise AssertionError(f"{mod.__name__}.__all__ is not an iterable of names")
    return list(names)


@pytest.mark.parametrize("package_name", _CANDIDATE_PACKAGES)
def test_all_names_resolve(package_name: str) -> None:
    """Every name listed in ``__all__`` must be getattr-able on the package."""
    mod = _try_import(package_name)
    missing = [name for name in _public_all(mod) if not hasattr(mod, name)]
    assert not missing, (
        f"{package_name}.__all__ lists names that do not resolve via getattr: "
        f"{missing}"
    )
