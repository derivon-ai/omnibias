# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Citation-path guards (theory 06-05 G1–G5).

This spec freezes a publish-and-use order for symbols that already ship.
It does not extract a new installable. ``PUBLIC_SURFACE`` here is the
document list, not a package.
"""

from __future__ import annotations

import re
from pathlib import Path

from omnibias.core.polynomials import (
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)
from omnibias.core.spec import ActivationSpec

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "theory" / "06-program" / "05-public-primitive-and-citation-path.md"
OPERATOR_SURFACE = REPO / "docs" / "operator-surface.md"
THEORY_README = REPO / "theory" / "README.md"
PACKAGES = REPO / "packages"

PUBLIC_SURFACE = (
    "omnibias.core.polynomials.sigmoid_polynomial_coeffs",
    "omnibias.core.polynomials.tanh_polynomial_coeffs",
    "omnibias.core.spec.ActivationSpec",
    "OperatorBlock.op in {identity, grad, laplacian, derivative, band, integral}",
    "omnibias.{torch,jax}.jet.compose_jet",
    "omnibias.{torch,jax}.jet.mlp_jet",
)
CITATION_ORDER = ("object", "theorem", "method_benchmark", "external_use")
NON_VEHICLE = ("ccf_stretch", "group_08_trainers_as_first_paper", "group_09")
ROLES = ("identity", "grad", "laplacian", "derivative", "band", "integral")
OBLIGATIONS = (
    "Obligation 1 — Use, not specs",
    "Obligation 2 — Standard computational object",
    "Obligation 3 — Method that replaces a default (narrow)",
    "Obligation 4 — Theorem about computation",
)


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_g1_four_obligations_are_complete() -> None:
    text = _spec_text()
    section = text.split("## 4. Mathematics", 1)[1].split("## 5. Worked example", 1)[0]
    for heading in OBLIGATIONS:
        assert heading in section, f"section 4 missing {heading!r}"
        _, rest = section.split(heading, 1)
        chunk = rest.split("#### Obligation", 1)[0]
        for label in ("- **Do.**", "- **Don't.**", "- **Done looks like.**"):
            assert label in chunk, f"{heading} missing {label}"


def test_g2_dependency_order_is_2_4_3_1() -> None:
    text = _spec_text()
    assert "2, then 4, then 3, then 1" in text
    assert "CITATION_ORDER = (\"object\", \"theorem\", \"method_benchmark\", \"external_use\")" in text
    assert CITATION_ORDER == ("object", "theorem", "method_benchmark", "external_use")


def test_g3_ccf_and_inventions_are_not_the_public_face() -> None:
    text = _spec_text()
    assert "not the public face" in text
    assert "NON_VEHICLE = (\"ccf_stretch\", \"group_08_trainers_as_first_paper\", \"group_09\")" in text
    for token in NON_VEHICLE:
        assert token in text
    assert "Fails G3" in text


def test_g4_no_prize_claim() -> None:
    text = _spec_text()
    honesty = text.split("## 10. Honesty and scope", 1)[1].split("## 11.", 1)[0]
    for word in ("Turing", "Clay", "Nobel"):
        assert word in honesty, f"honesty section dropped {word}"


def test_g5_does_not_mint_a_package() -> None:
    text = _spec_text()
    assert "No new package" in text
    assert re.search(r"This spec creates no\s+distribution", text)
    existing = {
        path.name
        for path in PACKAGES.glob("omnibias-*")
        if (path / "pyproject.toml").is_file()
    }
    assert "omnibias-public" not in existing
    assert not (PACKAGES / "omnibias-public").exists()


def test_public_surface_already_ships() -> None:
    """G5 / obligation 2: freeze what exists; do not invent a new extract."""
    assert callable(sigmoid_polynomial_coeffs)
    assert callable(tanh_polynomial_coeffs)
    assert ActivationSpec.__name__ == "ActivationSpec"
    surface = OPERATOR_SURFACE.read_text(encoding="utf-8")
    missing = [role for role in ROLES if f"`{role}`" not in surface]
    assert not missing, f"operator-surface.md missing roles {missing}"
    for backend in ("torch", "jax"):
        jet = (
            PACKAGES
            / f"omnibias-{backend}"
            / "src"
            / "omnibias"
            / backend
            / "jet.py"
        ).read_text(encoding="utf-8")
        assert re.search(r"^def compose_jet\(", jet, re.MULTILINE), backend
        assert re.search(r"^def mlp_jet\(", jet, re.MULTILINE), backend
    spec = _spec_text()
    for item in PUBLIC_SURFACE:
        assert item in spec, f"spec dropped PUBLIC_SURFACE item {item!r}"


def test_index_row_is_shipped_and_later_extract_stays_later() -> None:
    readme = THEORY_README.read_text(encoding="utf-8")
    row = re.search(
        r"\| \[06-05 public primitive and citation path\]\([^)]+\) \| (\w+) \|",
        readme,
    )
    assert row is not None, "theory/README.md lost the 06-05 index row"
    assert row.group(1) == "shipped"
    text = _spec_text()
    assert "PUBLIC_SURFACE_FROZEN = True" in text
    assert re.search(r"- \[x\] Freeze `PUBLIC_SURFACE`", text)
    assert re.search(
        r"- \[ \] Later: extract `PUBLIC_SURFACE`",
        text,
    ), "extract must stay a later item, not this pass"
    assert "Methods-paper outline" in text
    assert "1-D Poisson" in text
    assert "not CCF" in text.lower() or "not CCF stretch" in text


def test_citation_path_self_checks() -> None:
    assert len(OBLIGATIONS) == 4
    assert len(ROLES) == 6
    assert "identity" in ROLES and "integral" in ROLES
    assert "09-02" not in PUBLIC_SURFACE
    assert "ccf" not in " ".join(PUBLIC_SURFACE).lower()
