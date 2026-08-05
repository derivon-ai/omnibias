# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-level smoke tests: version, sorted __all__, backend-free import."""

from __future__ import annotations

import subprocess
import sys


def test_version() -> None:
    """The version lives on the distribution, not on the ``solver`` submodule.

    ``omnibias.pinn.solver`` is a submodule of the ``omnibias-pinn`` distribution,
    so it must *not* carry its own stray ``__version__`` marker; the single source
    of truth is ``omnibias.pinn.__version__`` (derived from installed metadata).
    """
    import importlib.metadata as importlib_metadata

    import omnibias.pinn as pinn
    import omnibias.pinn.solver as pde

    assert pinn.__version__ == importlib_metadata.version("omnibias-pinn")
    assert not hasattr(pde, "__version__"), (
        "omnibias.pinn.solver must not define a standalone __version__; the "
        "distribution version is the single source of truth"
    )


def test_all_is_sorted_and_exported() -> None:
    import omnibias.pinn.solver as pde

    assert list(pde.__all__) == sorted(pde.__all__)
    for name in pde.__all__:
        assert hasattr(pde, name), f"missing public symbol {name!r}"


def test_core_all_is_sorted() -> None:
    import omnibias.pinn.solver._core as core

    assert list(core.__all__) == sorted(core.__all__)


def test_import_is_backend_free() -> None:
    """Importing omnibias.pinn.solver must not pull in torch or jax."""
    code = (
        "import sys, omnibias.pinn.solver;"
        "assert 'torch' not in sys.modules, 'torch imported';"
        "assert 'jax' not in sys.modules, 'jax imported';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
