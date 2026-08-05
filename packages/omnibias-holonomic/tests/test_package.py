# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-level invariants: version, sorted __all__, purity, and honesty labels."""

from __future__ import annotations

import ast
from pathlib import Path

import omnibias.holonomic as H

_CORE = Path(H.__file__).parent / "_core"


def test_version() -> None:
    assert H.__version__ == "0.1.0a1"


def test_all_is_sorted_and_exported() -> None:
    assert H.__all__ == sorted(H.__all__)
    for name in H.__all__:
        assert hasattr(H, name), name


def test_core_has_no_backend_imports() -> None:
    # The holonomic core is pure Python: never import a tensor backend.
    banned = {"torch", "jax", "tensorflow", "keras", "numpy"}
    for path in _CORE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, f"{path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, f"{path.name}: {node.module}"


def test_docstring_states_theorem_prover_honesty() -> None:
    from omnibias.holonomic._core import certify

    doc = (certify.__doc__ or "") + (H.__doc__ or "")
    assert "theorem_prover_verified" in doc
    assert "lake" in doc
    assert "never forged" in doc


def test_docstring_labels_guessing_as_heuristic() -> None:
    # Fetch the submodule via importlib: the package re-exports the ``zeilberger``
    # function, which shadows the module of the same name under attribute access.
    import importlib

    from omnibias.holonomic._core import guess

    zeilberger_mod = importlib.import_module("omnibias.holonomic._core.zeilberger")
    doc = (guess.__doc__ or "") + (zeilberger_mod.__doc__ or "")
    assert "heuristic" in doc.lower() or "guess" in doc.lower()
    # the unconditional path is distinguished from the guessed one.
    assert "gosper_definite_sum" in (zeilberger_mod.__doc__ or "")


def test_theorem_prover_verified_never_true_without_lean() -> None:
    from math import comb

    proof = H.prove_hypergeometric_identity(
        name="sum C(n,k) = 2^n",
        summand=lambda n, k: comb(n, k),
        closed_form=lambda n: 2**n,
        n_max=6,
    )
    if not proof.lean_available:
        assert not proof.theorem_prover_verified
