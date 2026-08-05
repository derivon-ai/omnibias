# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Honesty guards for omnibias.discrete.sparse: yes-if framing + the two-collapse note."""

from __future__ import annotations

import importlib

import omnibias.discrete.sparse as sp

# Names that would imply a poly-time *exact* best-subset solver (i.e. P = NP).
_FORBIDDEN = {
    "solve",
    "solve_sparse",
    "best_subset",
    "best_subset_exact",
    "exact_support",
    "optimal_support",
    "l0_solve",
    "solve_l0",
}


def test_no_exact_solver_symbol_is_exported() -> None:
    assert not (set(sp.__all__) & _FORBIDDEN)
    for name in _FORBIDDEN:
        assert not hasattr(sp, name), f"{name} implies a poly-time exact NP solver"


def test_package_docstring_is_honest_yes_if() -> None:
    doc = (sp.__doc__ or "").lower()
    assert "np-hard" in doc or "p = np" in doc  # names the hard limit
    assert "certified" in doc and "gap" in doc  # the object it *does* deliver
    assert "never exactness" in doc or "never an exactness" in doc  # explicit yes-if


def test_relaxation_docstrings_carry_the_two_collapse_note() -> None:
    for mod_name in (
        "omnibias.discrete.sparse.jax.relaxation",
        "omnibias.discrete.sparse.torch.relaxation",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:  # backend not installed in this environment
            continue
        doc = mod.__doc__ or ""
        assert "founding bias collapse" in doc
        assert "delta -> 0" in doc
        assert "feasibility" in doc
        # the new axis must be labelled a penalty-exponent homotopy, not the founding limit
        assert "penalty-exponent" in doc


def test_container_docstring_distinguishes_the_collapse_senses() -> None:
    # Required even with no backend installed (the always-apply terminology rule).
    doc = sp.__doc__ or ""
    assert "founding bias collapse" in doc
    assert "l_p -> l_0" in doc and "penalty-exponent" in doc
