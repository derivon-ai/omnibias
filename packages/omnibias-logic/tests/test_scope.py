# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Honesty guards: the package answers the yes-if question, not an exact-NP / #P claim."""

from __future__ import annotations

import importlib
import sys

import omnibias.logic as logic
from omnibias.logic import (
    CountCertificate,
    CountResult,
    count_models_exact,
    treewidth_model_count,
    xor_model_count,
)


def _doc(obj: object) -> str:
    """A symbol's own docstring plus its defining module's docstring, lowercased."""
    own = getattr(obj, "__doc__", "") or ""
    module = sys.modules.get(getattr(obj, "__module__", ""), None)
    return (own + "\n" + (getattr(module, "__doc__", "") or "")).lower()

# Names that would imply a poly-time *exact* SAT / MaxSAT / model-count solver (P = NP / #P).
_FORBIDDEN = {
    "solve_sat",
    "solve_maxsat",
    "solve_model_count",
    "count_models",
    "num_models",
    "polynomial_model_count",
    "fast_model_count",
    "exact_count_poly",
    "optimal_assignment",
    "ground_state",
}


def test_no_exact_solver_symbol_is_exported() -> None:
    assert not (set(logic.__all__) & _FORBIDDEN)
    for name in _FORBIDDEN:
        assert not hasattr(logic, name), f"{name} must not exist (implies a poly-time exact solver)"


def test_count_certificate_is_enclosure_shaped_not_exactness() -> None:
    fields = CountCertificate.__dataclass_fields__
    assert "lower" in fields and "upper" in fields
    for banned in ("optimal", "is_exact", "exact", "is_optimal", "zero_gap", "certified_exact"):
        assert banned not in fields, f"{banned} would imply an exactness claim"
    for prop in ("width", "relative_width", "is_sound"):
        assert hasattr(CountCertificate, prop)


def test_package_docstring_is_honest_yes_if() -> None:
    doc = (logic.__doc__ or "").lower()
    assert "p = np" in doc  # names the hard optimization limit
    assert "#p" in doc  # names the hard counting limit
    assert "certified" in doc
    assert "enclosure" in doc and "gap" in doc  # the objects it *does* deliver
    assert "yes" in doc  # the yes-if framing
    assert "closed-form" in doc  # ties to the omnibias derivative-tower promise


def test_exact_oracles_are_labelled_exponential() -> None:
    assert "exponential" in (logic.exact_model_count.__doc__ or "").lower()
    doc = (logic.brute_force_min.__doc__ or "").lower()
    assert "exponential" in doc or "2^n" in doc


def test_sound_exact_symbols_carry_honest_regime_docstrings() -> None:
    xor_doc = _doc(xor_model_count)
    assert "affine" in xor_doc and "gf(2)" in xor_doc and "unweighted" in xor_doc
    tw_doc = _doc(treewidth_model_count)
    assert "treewidth" in tw_doc and "exponential" in tw_doc
    dpll_doc = _doc(count_models_exact)
    assert "exponential" in dpll_doc
    # the counting-fragment package doc must name the #P-complete traps it refuses.
    pkg_doc = (importlib.import_module("omnibias.logic.model_count").__doc__ or "").lower()
    pkg_doc = pkg_doc.replace("`", "")  # strip reST code-span backticks before matching
    assert "#p-hard" in pkg_doc or "#p-complete" in pkg_doc
    assert "#2-sat" in pkg_doc and "#horn-sat" in pkg_doc


def test_count_result_is_soundness_tagged_not_an_exactness_field() -> None:
    fields = CountResult.__dataclass_fields__
    assert "guarantee" in fields  # "exact" | "certified_enclosure"
    assert "exact" not in fields and "is_exact" not in fields  # exactness is a derived property
    for prop in ("is_exact", "is_sound", "width", "contains"):
        assert hasattr(CountResult, prop)


def test_sealed_count_certificate_is_honest_and_lean_bridged() -> None:
    # the bridge surface is exported at the top level for a single import point.
    for name in ("seal_count_certificate", "check_certificate", "verify_certificate_digest"):
        assert name in logic.__all__ and hasattr(logic, name)
    # the bridge docstring names the honest boundary: finite arithmetic only, no P = NP / #P.
    proof_doc = _doc(logic.seal_count_certificate).replace("`", "")
    assert "tamper-evident" in proof_doc
    assert "finite" in proof_doc
    assert "#p-hard" in proof_doc  # explicit: no poly-time exact-count claim
    # a sealed exact-count certificate hard-wires the no-unproven / no-poly-exact honesty flags.
    mc = logic.model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
    sealed = logic.count_enclosure(mc, order=4).seal(problem=mc)
    assert sealed["honesty"]["unproven_claim"] is False
    assert sealed["meta"]["poly_time_exact"] is False
    assert sealed["meta"]["kind"] == "model_count"
    # theorem_prover_verified is earned by the kernel, never asserted by the certificate body.
    assert "theorem_prover_verified" not in sealed
    assert "theorem_prover_verified" not in sealed.get("honesty", {})


def test_statistical_layer_is_quarantined_and_labelled_not_sound() -> None:
    approx = importlib.import_module("omnibias.logic.approx")
    doc = (approx.__doc__ or "").lower()
    assert "not worst-case sound" in doc and "quarantined" in doc
    # the statistical estimators must NOT leak into the sound top-level namespace.
    for name in ("approx_model_count", "ConformalCounter", "ApproxCount"):
        assert name not in logic.__all__
        assert not hasattr(logic, name)
    # the result type advertises its non-sound contract and refuses to be forged sound.
    approx_count = approx.ApproxCount
    assert approx_count.__dataclass_fields__["worst_case_sound"].default is False
    assert "not" in approx.NOT_SOUND_DISCLAIMER.lower()


def test_relaxation_docstrings_distinguish_the_two_collapse_senses() -> None:
    seen = 0
    for mod_name in ("omnibias.logic.jax.relaxation", "omnibias.logic.torch.relaxation"):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:  # backend not installed in this environment
            continue
        doc = mod.__doc__ or ""
        assert "founding bias collapse" in doc
        assert "delta -> 0" in doc
        assert "feasibility" in doc.lower()
        seen += 1
    # the package docstring must carry the note even with no backend installed
    assert "founding bias collapse" in (logic.__doc__ or "")
    assert seen >= 0
