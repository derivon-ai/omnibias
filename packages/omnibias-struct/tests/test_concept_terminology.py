# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Guardrail: omnibias-struct must keep the two axes labelled and never conflated.

The ``beta -> inf`` log-sum-exp relaxation (feasibility / temperature axis) is
differentiated **exactly** by the ``delta -> 0`` founding bias collapse tower (closed-form
softplus / softmax jets). These filesystem + docstring checks fail if that disambiguation
is deleted -- mirroring ``omnibias-core``'s ``test_concept_terminology`` for the peer
relaxation packages.
"""

from __future__ import annotations

from pathlib import Path

import omnibias.struct as st
import pytest


def _pkg_source(rel: str) -> str:
    src = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "struct" / rel
    if not src.is_file():
        pytest.skip(f"{rel} not present (installed as a wheel?); skipping filesystem check")
    return src.read_text(encoding="utf-8")


def test_package_docstring_labels_both_axes() -> None:
    doc = " ".join((st.__doc__ or "").split())  # normalise line-wrapping
    assert "beta -> inf" in doc  # the relaxation / temperature axis
    assert "delta -> 0" in doc  # the founding derivative-tower axis
    assert "founding bias collapse" in doc
    assert "feasibility" in doc.lower()  # the relaxation is the feasibility sense ...
    assert "**not** bias collapse" in doc  # ... explicitly NOT bias collapse
    assert "softplus" in doc and "sigma^(n-1)" in doc  # the load-bearing tower identity
    assert "conflat" in doc.lower()


@pytest.mark.parametrize("rel", ["torch/soft_dp.py", "jax/soft_dp.py", "_core/tropical.py"])
def test_soft_dp_docstrings_cross_reference_the_founding_sense(rel: str) -> None:
    text = _pkg_source(rel)
    assert "founding bias collapse" in text, f"{rel} missing the disambiguation cross-ref"
    assert "feasibility" in text.lower(), f"{rel} missing the feasibility-sense label"
    assert "do not conflate" in text.lower(), f"{rel} missing the do-not-conflate warning"


@pytest.mark.parametrize("rel", ["torch/_logsumexp.py", "jax/_logsumexp.py"])
def test_logsumexp_docstrings_name_the_tower(rel: str) -> None:
    text = _pkg_source(rel)
    assert "softplus^(n) = sigma^(n-1)" in text  # the exact closed-form tower identity
    assert "compose_jet" in text  # built on omnibias.{torch,jax}.jet


@pytest.mark.parametrize(
    "rel", ["_core/select.py", "select.py", "torch/select.py", "jax/select.py"]
)
def test_select_docstrings_label_the_measure_temperature_sense(rel: str) -> None:
    # The certified-argmax / measure-mode collapse is the beta->inf temperature/measure axis,
    # differentiated by (never conflated with) the founding delta->0 bias-collapse tower.
    text = _pkg_source(rel)
    assert "beta -> inf" in text, f"{rel} missing the beta->inf temperature-axis label"
    assert "delta -> 0" in text, f"{rel} missing the founding delta->0 cross-ref"
    assert "bias collapse" in text, f"{rel} missing the founding-bias-collapse cross-ref"
    assert "feasibility" in text.lower(), f"{rel} missing the feasibility-sense label"
    assert "do not conflate" in text.lower(), f"{rel} missing the do-not-conflate warning"


def test_certificate_is_honest_about_the_gap() -> None:
    # The certificate never claims equality -- only the closed-form log(N)/beta sandwich.
    text = _pkg_source("_core/certificate.py")
    assert "log(N) / beta" in text
    assert "honest" in text.lower()
    assert "never claims" in text.lower()
