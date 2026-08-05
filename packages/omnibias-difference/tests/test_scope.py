# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Honesty / scope guards: founding-sense terminology and correct labels."""

from __future__ import annotations

import omnibias.difference as D
from omnibias.difference._core import euler, stencil


def test_founding_sense_terminology_in_package_docstring() -> None:
    doc = D.__doc__ or ""
    assert "founding bias collapse" in doc
    assert "sigma^(K-1)" in doc
    assert "conflate" in doc.lower()
    assert "beta -> inf" in doc  # cross-references the distinct penalty sense
    assert "delta -> 0" in doc


def test_stencil_module_states_the_founding_sense() -> None:
    doc = stencil.__doc__ or ""
    assert "founding bias collapse" in doc
    assert "sigma^(K-1)" in doc


def test_honesty_labels_are_correct() -> None:
    enc = D.certified_derivative_enclosure("tanh", 0.0, 2)
    assert enc.label == "closed-form"
    est = D.finite_difference_estimate("tanh", 0.0, 2, 1e-2)
    assert est.label == "numerical"
    cert = D.certified_fd_error("tanh", 0.0, 2, 1e-2)
    assert "closed-form" in cert.label and "numerical" in cert.label


def test_only_closed_form_and_numerical_registers_are_claimed() -> None:
    doc = D.__doc__ or ""
    normalized = " ".join(doc.split())  # collapse line wraps
    assert "closed-form" in normalized
    assert "numerical" in normalized
    # the package explicitly disclaims an autodiff-exact path
    assert "no ``autodiff-exact`` path" in normalized


def test_eulerian_numbers_are_not_claimed_off_the_sigmoid_tower() -> None:
    doc = euler.__doc__ or ""
    assert "Worpitzky" in doc
    assert "not" in doc.lower()
    assert "eulerian" in doc.lower() and "sigmoid" in doc.lower()
    fn_doc = D.eulerian_number.__doc__ or ""
    assert "Worpitzky" in fn_doc
