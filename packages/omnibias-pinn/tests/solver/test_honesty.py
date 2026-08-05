# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Honesty regression: unproven_claim is never True, labels are well-formed.

These checks are backend-free (they exercise only the pure-Python ``_core``
honesty helpers plus a static source scan), so they always run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import omnibias.pinn.solver as pde
import pytest
from omnibias.pinn.solver._core import honesty

_GUARDED = {"unproven_claim", "continuum_claim"}


def test_method_labels_are_the_known_set() -> None:
    assert honesty.METHOD_LABELS == {
        honesty.CLOSED_FORM,
        honesty.AUTODIFF,
        honesty.NUMERICAL,
        honesty.SPECTRAL,
        honesty.HIGH_ORDER,
    }
    # each label is a distinct, non-empty, lower-case token
    assert len(honesty.METHOD_LABELS) == 5
    for label in honesty.METHOD_LABELS:
        assert label and label == label.lower()


def test_honesty_labels_default_makes_no_unproven_claim() -> None:
    labels = pde.honesty_labels()
    assert labels["unproven_claim"] is False
    assert labels["continuum_claim"] is False
    assert labels["interval_verified"] is False
    assert "theorem_prover_verified" not in labels


def test_honesty_labels_rejects_reserved_formal_key() -> None:
    with pytest.raises(ValueError, match="theorem_prover_verified"):
        pde.honesty_labels(theorem_prover_verified=False)


def test_honesty_labels_preserves_extra_and_verified_flag() -> None:
    labels = pde.honesty_labels(interval_verified=True, method=honesty.CLOSED_FORM)
    assert labels["interval_verified"] is True
    assert labels["method"] == honesty.CLOSED_FORM
    assert labels["unproven_claim"] is False


def test_honesty_labels_rejects_forged_unproven_claim() -> None:
    with pytest.raises(ValueError, match="unproven_claim"):
        pde.honesty_labels(unproven_claim=True)


def test_honesty_labels_rejects_continuum_claim() -> None:
    with pytest.raises(ValueError, match="continuum"):
        pde.honesty_labels(continuum_claim=True)


def test_assert_no_unproven_claim_guard() -> None:
    honesty.assert_no_unproven_claim({"unproven_claim": False})
    honesty.assert_no_unproven_claim({})  # missing key defaults to False
    with pytest.raises(ValueError):
        honesty.assert_no_unproven_claim({"unproven_claim": True})


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def test_no_source_forges_a_true_unproven_or_continuum_claim() -> None:
    """Static (AST) guard: no shipped source sets unproven/continuum_claim to True.

    Parses each module rather than grepping text, so prose in docstrings / error
    messages that mentions ``unproven_claim=True`` is not a false positive; only real
    dict-literal values and call keywords are inspected.
    """
    src_root = Path(pde.__file__).parent
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=False):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value in _GUARDED
                        and _is_true(value)
                    ):
                        offenders.append(f"{path.name}:{node.lineno}: dict {key.value}=True")
            elif isinstance(node, ast.keyword):
                if node.arg in _GUARDED and _is_true(node.value):
                    offenders.append(f"{path.name}:{node.lineno}: kwarg {node.arg}=True")
    assert not offenders, "sources must never set unproven/continuum True:\n" + "\n".join(
        offenders
    )
