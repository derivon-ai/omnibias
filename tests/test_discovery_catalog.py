# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Imported-package catalog: no kind collisions across proofmachines."""

from __future__ import annotations

from omnibias.core.proof import list_catalog


def test_imported_catalogs_have_unique_kinds() -> None:
    kinds: list[str] = []
    try:
        import omnibias.holonomic.proofmachine  # noqa: F401
    except ImportError:
        pass
    try:
        import omnibias.combinatorics.proofmachine  # noqa: F401
    except ImportError:
        pass
    try:
        import omnibias.symbolic.proofmachine  # noqa: F401
    except ImportError:
        pass
    try:
        import omnibias.sos.proofmachine  # noqa: F401
    except ImportError:
        pass
    try:
        import omnibias.geometry.gauge.proofmachine  # noqa: F401
    except ImportError:
        pass
    try:
        import omnibias.pinn.certified.machine  # noqa: F401
    except ImportError:
        pass
    listed = list_catalog()
    kinds = [entry.kind for entry in listed]
    assert len(kinds) == len(set(kinds))
    assert any(entry.mode == "exact_search" for entry in listed)
