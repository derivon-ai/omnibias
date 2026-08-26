# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-family catalog guards (theory 01-13 G1–G5).

This spec is a document plus the existing ``op=`` alias. These tests
read markdown and shipped symbols; they do not mint a seventh role.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from omnibias.core.scan import resolve_scan_role

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "theory" / "01-geometry" / "13-operator-family.md"
THEORY_README = REPO / "theory" / "README.md"
PACKAGES = REPO / "packages"
BIAS_SCAN_SMOKE = REPO / "docs" / "benchmarks" / "bias_scan_smoke.json"

ROLES = ("identity", "grad", "laplacian", "derivative", "band", "integral")
POINTER_SPECS = (
    "09-14",
    "09-03",
    "09-17",
    "01-07",
    "09-04",
    "03-05",
    "01-08",
    "03-04",
    "09-13",
    "09-08",
    "09-10",
    "09-11",
    "01-04",
    "02-06",
    "01-05",
    "02-04",
    "09-27",
    "09-28",
)
NO_RESPEC = ("09-14", "03-05", "02-06")
REJECT_BANNED = (
    "seventh_operatorblock_role",
    "hilbert_convolution",
    "op_conv2d",
)
_SECTION = re.compile(
    r"^## (?P<n>\d+)\. [^\n]+\n(?P<body>.*?)(?=^## \d|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _section(text: str, number: int) -> str:
    for match in _SECTION.finditer(text):
        if int(match.group("n")) == number:
            return match.group("body")
    raise AssertionError(f"missing ## {number}.")


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_g1_section4_has_catalog_tables() -> None:
    section4 = _section(_spec_text(), 4)
    assert "### Generator" in section4
    assert "### Role × scan" in section4
    assert "### Other first-class operators (not scans)" in section4
    assert "### Inventable (pointers only)" in section4
    assert "### Rejected (no spec)" in section4
    for role in ROLES:
        assert f"`{role}`" in section4, f"role × scan dropped {role}"


def test_g2_inventable_rows_point_at_existing_specs() -> None:
    section4 = _section(_spec_text(), 4)
    inventable = section4.split("### Inventable (pointers only)", 1)[1].split(
        "### Rejected", 1
    )[0]
    for spec in POINTER_SPECS:
        assert spec in inventable, f"inventable table dropped {spec}"
    for spec in NO_RESPEC:
        assert spec in inventable
    assert "Do not open a second" in inventable
    assert "09-14" in inventable and "03-05" in inventable and "02-06" in inventable


def test_g3_rejects_are_named_and_unimplemented() -> None:
    section4 = _section(_spec_text(), 4)
    rejected = section4.split("### Rejected (no spec)", 1)[1]
    assert "seventh `OperatorBlock` role" in rejected
    assert "Translation equivariance on `R^D`" in rejected
    assert "Hilbert convolution" in rejected
    assert "Softmax / generic MoE" in rejected
    assert "`maxpool` or `vit`" in rejected
    assert "ImageNet ViT" in rejected
    hits = [
        path.relative_to(REPO).as_posix()
        for path in PACKAGES.rglob("*")
        if path.is_file() and any(token in path.name for token in REJECT_BANNED)
    ]
    assert not hits, f"rejected ideas grew implementation files: {hits}"
    assert not (PACKAGES / "omnibias-operators").exists()
    assert not (PACKAGES / "omnibias-operator-family").exists()


def test_g4_first_spend_is_integral_alias_then_09_14() -> None:
    text = _spec_text()
    assert "BiasScan(op=\"integral\")" in text or "BiasScan(op='integral')" in text
    assert "FIRST_SPEND = \"BiasScan(op='integral')\"" in text
    assert resolve_scan_role(op="integral", default="grad") == "integral"
    consumer = (
        REPO / "theory" / "09-inventions" / "14-integral-kernel-operator.md"
    )
    assert consumer.is_file()
    status = re.search(
        r"^\s*-\s+\*\*Status\*\*:\s*(\w+)",
        consumer.read_text(),
        re.MULTILINE,
    )
    assert status is not None and status.group(1) == "shipped"


def test_g5_no_new_package_and_bias_scan_g5_earned() -> None:
    existing = {
        path.name
        for path in PACKAGES.glob("omnibias-*")
        if (path / "pyproject.toml").is_file()
    }
    assert "omnibias-operators" not in existing
    payload = json.loads(BIAS_SCAN_SMOKE.read_text(encoding="utf-8"))
    assert payload["honesty"]["g5_earned"] is True
    assert payload["g5"]["name"] == "g5_integral_op_alias"
    assert payload["g5"]["passed"] is True


def test_index_row_is_shipped() -> None:
    row = re.search(
        r"\| \[01-13 operator family\]\([^)]+\) \| (\w+) \|",
        THEORY_README.read_text(encoding="utf-8"),
    )
    assert row is not None
    assert row.group(1) == "shipped"
