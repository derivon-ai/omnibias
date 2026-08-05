# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Every package that promises a bit-identical JAX twin must state the precondition.

The promise is real but conditional: the shared coefficient module makes the two
backends agree bit for bit **in double precision**, and JAX defaults to
``float32``. Under the default the twins stay internally consistent and agree
only to ``float32`` tolerance -- which in the decoders (``discrete``, ``qubo``,
``struct``) and hardening steps (``partition``, ``tab``) is enough to flip a
rounded bit and move an answer by a whole unit, not by a rounding error.

omnibias deliberately does not enable ``jax_enable_x64`` on import: the flag is
process-global and irreversible once arrays exist, so a library that set it
would silently re-specify the numerics of unrelated JAX code in the same
process. The obligation that comes with that choice is to say so wherever the
claim is made, which is what this test enforces.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"

_CLAIM = re.compile(r"bit-identical|bit-parity|bit-for-bit", re.I)
_PRECONDITION = re.compile(r"x64|float64|double precision", re.I)


def _module_docstring(path: Path) -> str:
    try:
        return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
        return ""


def _top_package_dirs() -> list[Path]:
    """The importable top-level directory of each distribution, e.g. ``omnibias/qubo``."""
    tops: list[Path] = []
    for distribution in sorted(PACKAGES.glob("omnibias-*")):
        tops.extend(p for p in sorted(distribution.glob("src/omnibias/*")) if p.is_dir())
    return tops


def _claims_bit_identity(top: Path) -> bool:
    return any(_CLAIM.search(_module_docstring(p)) for p in top.rglob("*.py"))


def _has_jax_twin(top: Path) -> bool:
    return (top / "jax").is_dir() or any(p.is_dir() for p in top.glob("*/jax"))


def test_packages_with_a_jax_twin_state_the_x64_precondition() -> None:
    missing = [
        top.name
        for top in _top_package_dirs()
        if _has_jax_twin(top)
        and _claims_bit_identity(top)
        and not _PRECONDITION.search(_module_docstring(top / "__init__.py"))
    ]
    assert not missing, (
        "these packages promise a bit-identical JAX twin without saying that the "
        "promise needs 64-bit JAX; add the `.. important::` note to the top-level "
        f"__init__ docstring: {missing}"
    )


def test_the_precision_helper_reports_the_live_flag() -> None:
    import pytest

    pytest.importorskip("jax")
    import jax
    from omnibias.jax.precision import X64_HINT, require_x64, x64_enabled

    assert x64_enabled() is bool(jax.config.jax_enable_x64)
    assert "jax_enable_x64" in X64_HINT
    if x64_enabled():
        require_x64()  # must not raise
    else:
        with pytest.raises(RuntimeError, match="64-bit JAX"):
            require_x64()


def _sets_x64_at_import(path: Path) -> bool:
    """A real ``jax.config.update("jax_enable_x64", ...)`` call, not a doc mention."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not (isinstance(target, ast.Attribute) and target.attr == "update"):
            continue
        if any(
            isinstance(arg, ast.Constant) and arg.value == "jax_enable_x64"
            for arg in node.args
        ):
            return True
    return False


def test_the_library_does_not_flip_the_global_flag_on_import() -> None:
    """Importing omnibias must not re-specify a user's JAX numerics behind their back."""
    # `omnibias.pinn.solver.jax` is the one documented exception: it is a solver
    # entry point whose results are meaningless in float32, and its docstring says so.
    allowed = {"omnibias/pinn/solver/jax/__init__.py"}
    offenders = sorted(
        str(path.relative_to(PACKAGES))
        for top in _top_package_dirs()
        for path in top.rglob("__init__.py")
        if _sets_x64_at_import(path)
        and not any(path.as_posix().endswith(a) for a in allowed)
    )
    assert not offenders, (
        "these modules enable 64-bit JAX at import time, which mutates global state "
        f"for every other JAX user in the process: {offenders}"
    )
