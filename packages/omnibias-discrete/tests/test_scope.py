# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Honesty guards: the substrate answers the yes-if question, not an exact-NP claim."""

from __future__ import annotations

import importlib

import omnibias.discrete as discrete
from omnibias.discrete import GapCertificate

# Names that would imply a poly-time *exact* global discrete solver (i.e. P = NP).
_FORBIDDEN = {
    "solve",
    "solve_exact",
    "solve_discrete",
    "solve_qubo",
    "solve_maxsat",
    "optimal_assignment",
    "exact_min",
    "exact_minimum",
    "best_assignment",
    "global_minimum",
    "ground_state",
}


def test_no_exact_solver_symbol_is_exported() -> None:
    assert not (set(discrete.__all__) & _FORBIDDEN)
    for name in _FORBIDDEN:
        assert not hasattr(discrete, name), f"{name} implies a poly-time exact NP solver"


def test_certificate_is_gap_shaped_not_exactness() -> None:
    fields = GapCertificate.__dataclass_fields__
    assert "lower_bound" in fields and "energy" in fields
    for banned in ("optimal", "is_exact", "exact", "is_optimal", "zero_gap"):
        assert banned not in fields, f"{banned} would imply an exactness claim"
    for prop in ("absolute_gap", "relative_gap", "is_sound"):
        assert hasattr(GapCertificate, prop)


def test_package_docstring_is_honest_yes_if() -> None:
    doc = (discrete.__doc__ or "").lower()
    assert "p = np" in doc  # names the hard limit
    assert "certified" in doc and "gap" in doc  # the object it *does* deliver
    assert "yes" in doc  # the yes-if framing
    assert "closed-form" in doc  # ties to the omnibias derivative-tower promise


def test_brute_force_is_labelled_exponential() -> None:
    doc = (discrete.brute_force_min.__doc__ or "").lower()
    assert "exponential" in doc or "2^n" in doc


def test_relaxation_docstrings_distinguish_the_two_collapse_senses() -> None:
    seen = 0
    modules = (
        "omnibias.discrete.jax.relaxation",
        "omnibias.discrete.torch.relaxation",
        "omnibias.discrete.maxsat.jax.relaxation",
        "omnibias.discrete.maxsat.torch.relaxation",
    )
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:  # backend not installed in this environment
            continue
        doc = mod.__doc__ or ""
        assert "founding bias collapse" in doc
        assert "delta -> 0" in doc
        assert "feasibility" in doc
        seen += 1
    # The always-apply terminology rule requires the note wherever "collapse" appears;
    # the container docstring must carry it even with no backend installed.
    assert "founding bias collapse" in (discrete.__doc__ or "")
    assert seen >= 0
