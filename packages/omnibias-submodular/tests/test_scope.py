# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Honesty guards: the package answers the yes-if question, not an exact-NP claim."""

from __future__ import annotations

import importlib

import omnibias.submodular as sub
from omnibias.submodular import SubmodularCertificate

# Names that would imply a poly-time *exact* constrained submodular maximizer (P = NP).
# Carve-out: exact *minimization* (``submodular_minimize`` / ``min_norm_point``) is a genuine
# P-class result and is deliberately NOT forbidden -- only these maximization-flavoured names
# are, since it is the NP-hard maximization that must never claim exactness.
_FORBIDDEN = {
    "solve",
    "solve_submodular",
    "solve_exact",
    "exact_max",
    "exact_maximum",
    "optimal_set",
    "optimal_selection",
    "best_selection",
    "global_maximum",
    "ground_truth",
}


def test_no_exact_solver_symbol_is_exported() -> None:
    assert not (set(sub.__all__) & _FORBIDDEN)
    for name in _FORBIDDEN:
        assert not hasattr(sub, name), f"{name} must not exist (implies a poly-time exact NP solver)"


def test_certificate_is_gap_shaped_not_exactness() -> None:
    fields = SubmodularCertificate.__dataclass_fields__
    assert "value" in fields and "upper_bound" in fields
    for banned in ("optimal", "is_exact", "exact", "is_optimal", "zero_gap"):
        assert banned not in fields, f"{banned} would imply an exactness claim"
    for prop in ("absolute_gap", "relative_gap", "certified_ratio", "internal_consistent"):
        assert hasattr(SubmodularCertificate, prop)


def test_package_docstring_is_honest_yes_if() -> None:
    doc = (sub.__doc__ or "").lower()
    assert "p = np" in doc  # names the hard limit
    assert "certified" in doc and "gap" in doc  # the object it *does* deliver
    assert "yes" in doc  # the yes-if framing
    assert "closed-form" in doc  # ties to the omnibias derivative-tower promise
    assert "1 - 1/e" in doc  # the honest approximation ratio, not exactness


def test_brute_force_is_labelled_exponential() -> None:
    doc = (sub.brute_force_max.__doc__ or "").lower()
    assert "exponential" in doc or "2^n" in doc


def test_exact_minimization_is_the_allowed_p_class_carveout() -> None:
    # Minimization is P-class: the exact minimizer IS exported (unlike any exact *maximizer*).
    assert "submodular_minimize" in sub.__all__
    assert "min_norm_point" in sub.__all__
    assert hasattr(sub, "submodular_minimize") and hasattr(sub, "min_norm_point")
    # ...but it must honestly disclaim P = NP: exactness here is minimization, not maximization.
    doc = (sub.submodular_minimize.__doc__ or "").lower()
    assert "p-class" in doc
    assert "p = np" in doc  # explicitly says this is NOT a P = NP claim
    assert "exact" in doc
    # The package docstring carries the same carve-out.
    pkg = (sub.__doc__ or "").lower()
    assert "minimization" in pkg and "p-class" in pkg


def test_relaxation_docstrings_distinguish_the_two_collapse_senses() -> None:
    seen = 0
    for mod_name in ("omnibias.submodular.jax.relaxation", "omnibias.submodular.torch.relaxation"):
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
    assert "founding bias collapse" in (sub.__doc__ or "")
    assert "feasibility" in (sub.__doc__ or "")
    assert seen >= 0
