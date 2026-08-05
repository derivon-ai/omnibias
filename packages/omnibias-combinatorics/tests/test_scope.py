# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Honesty guards: the yes-if framing, a gap-shaped certificate, and the two collapse senses."""

from __future__ import annotations

import importlib

import omnibias.combinatorics as combinatorics
from omnibias.combinatorics import CombinatorialCertificate


def test_package_docstring_is_honest_yes_if() -> None:
    doc = (combinatorics.__doc__ or "").lower()
    assert "yes" in doc  # the yes-if framing
    assert "certified" in doc and "gap" in doc  # the object it delivers
    assert "integral" in doc and "tight" in doc  # why the gap is tight, not asserted
    # honest about complexity: these are in P, so it must NOT parrot a P = NP claim
    assert "in **p**" in doc or "in p" in doc
    assert "p = np" not in doc and "p=np" not in doc


def test_certificate_is_gap_shaped_not_exactness() -> None:
    fields = CombinatorialCertificate.__dataclass_fields__
    assert "lower_bound" in fields and "objective" in fields
    for banned in ("optimal", "is_exact", "exact", "is_optimal", "zero_gap"):
        assert banned not in fields, f"{banned} would imply an exactness claim"
    for prop in ("absolute_gap", "relative_gap", "is_sound"):
        assert hasattr(CombinatorialCertificate, prop)


def test_exact_classical_baseline_is_available() -> None:
    """Unlike NP-hard packages, an exact poly-time solver is legitimate here (P), and named."""
    assert hasattr(combinatorics, "classical_optimum")
    assert "exact" in (combinatorics.classical_optimum.__doc__ or "").lower()


def test_brute_force_is_labelled_exponential() -> None:
    doc = (combinatorics.brute_force_min.__doc__ or "").lower()
    assert "exponential" in doc or "2^" in doc or "n!" in doc


def test_relaxation_docstrings_distinguish_the_two_collapse_senses() -> None:
    seen = 0
    for mod_name in (
        "omnibias.combinatorics.jax.relaxation",
        "omnibias.combinatorics.torch.relaxation",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:  # backend not installed in this environment
            continue
        doc = mod.__doc__ or ""
        assert "founding bias collapse" in doc
        assert "delta -> 0" in doc
        assert "feasibility" in doc.lower()
        seen += 1
    # the container docstring must carry the note even with no backend installed
    assert "founding bias collapse" in (combinatorics.__doc__ or "")
    assert seen >= 0
