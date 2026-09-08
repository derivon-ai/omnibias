# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory-tree structural guards (theory 06-02 G3/G5, theory 07-01 G1–G5)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FRONTIER = REPO / "theory" / "07-frontier"
THEORY = REPO / "theory"
LEDGER = FRONTIER / "01-sub-obligation-ledger.md"
LEDGER_DOCS = REPO / "docs" / "frontier-ledger.md"
HONESTY_SRC = (
    REPO
    / "packages"
    / "omnibias-pinn"
    / "src"
    / "omnibias"
    / "pinn"
    / "solver"
    / "_core"
    / "honesty.py"
)
METHOD_LABELS = ("CLOSED_FORM", "AUTODIFF", "NUMERICAL", "SPECTRAL", "HIGH_ORDER")
LEDGER_TABLE_HEADER = (
    "| Parent key | Parent | Sub-obligation | Gate | Sealed scope | "
    "Never write | Entry | Distance |"
)
CLAIM_FLAG_MODULES = (
    "omnibias.pinn.certified.navier_stokes",
    "omnibias.pinn.certified.machine",
    "omnibias.geometry.gauge.transfer",
    "omnibias.core.verified.dirichlet",
    "omnibias.core.verified.debruijn_newman",
    "omnibias.core.verified.eig_operator",
    "omnibias.sos",
    "omnibias.qubo",
    "omnibias.discrete",
    "benchmarks/_gates.py",
    "omnibias.core.proof.obligations.convergence_ledger",
)
_BENCHMARK_PATH = re.compile(r"`(docs/benchmarks/[^`]+\.json)`")
_IMPORT_PADE = re.compile(
    r"(?:omnibias\.difference.*(?:pade|singularity)|"
    r"from omnibias\.difference[^\n]*(?:pade|singularity))",
    re.IGNORECASE,
)
_IMPORT_DIRICHLET = re.compile(r"omnibias\.core\.verified\.dirichlet")


def test_group_07_specs_have_section_13() -> None:
    files = sorted(p for p in FRONTIER.glob("*.md") if p.name != "README.md")
    assert files, "theory/07-frontier/ has no specs"
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^## 13\.", text, re.MULTILINE), f"{path.name} missing ## 13."
        assert re.search(r"(?i)\bparent\b", text), f"{path.name} does not name a parent"
        assert re.search(
            r"(?i)(stays an external|external obligation|does not claim|not claimed)",
            text,
        ), f"{path.name} does not state the non-claim"


def test_method_labels_exist_in_honesty_module() -> None:
    text = HONESTY_SRC.read_text(encoding="utf-8")
    missing = [label for label in METHOD_LABELS if label not in text]
    assert missing == [], f"honesty.py missing labels {missing}"


def test_claim_docs_name_the_five_method_labels() -> None:
    paths = (
        THEORY / "06-program" / "02-honesty-and-claim-boundaries.md",
        REPO / "docs" / "honesty.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        missing = [label for label in METHOD_LABELS if label not in text]
        assert missing == [], f"{path.name} missing method labels {missing}"


def _ledger_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith(LEDGER_TABLE_HEADER):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 8:
                rows.append(cells)
            continue
        if in_table:
            break
    return rows


def test_group_07_entries_are_in_the_ledger() -> None:
    """Every theory/07-frontier/*.md except the ledger itself names a parent
    that appears in the ledger's table, and the ledger's 'never write' column is
    non-empty for that parent."""
    ledger = LEDGER.read_text(encoding="utf-8")
    rows = _ledger_rows(ledger)
    assert rows, "ledger table is empty"
    parents = [row[1] for row in rows]
    never = {row[0]: row[5] for row in rows}
    assert all(never.values()), f"empty never-write column: {never}"
    entries = sorted(
        path for path in FRONTIER.glob("*.md") if path.name != "README.md"
    )
    ledger_name = LEDGER.name
    for path in entries:
        if path.name == ledger_name:
            continue
        section = path.read_text(encoding="utf-8").split("## 13.", 1)
        assert len(section) == 2, f"{path.name} missing section 13"
        blob = section[1]
        aliases = {
                "Navier-Stokes global regularity (Clay)": ("Navier-Stokes",),
                "finite-time singularity of 3D Euler / Navier-Stokes": (
                    "finite-time singularity",
                ),
                "Yang-Mills existence and mass gap (Clay)": ("Yang-Mills",),
                "the Riemann Hypothesis": ("Riemann Hypothesis",),
                "P versus NP": ("P versus NP", "P vs NP"),
                "turbulence closure (Nobel-adjacent)": ("turbulence",),
                "computer-assisted global dynamical structure": (
                    "dynamical systems",
                    "computer-assisted",
                ),
                "Nobel-adjacent scientific discovery": (
                    "quantum many-body",
                    "fusion",
                    "materials",
                ),
            }
        matched = [
            parent
            for parent in parents
            if any(alias.lower() in blob.lower() for alias in aliases.get(parent, (parent,)))
        ]
        assert matched, (
            f"{path.name} section 13 parent is not in the ledger table: "
            f"{blob[:180]!r}"
        )


def test_rh_entry_is_lambda_scoped() -> None:
    """The RH row may name only a de Bruijn--Newman Lambda research program."""
    rows = _ledger_rows(LEDGER.read_text(encoding="utf-8"))
    rh_rows = [row for row in rows if row[0] == "RH"]
    assert len(rh_rows) == 1, f"expected one RH ledger row, found {rh_rows!r}"
    _key, parent, sub_obligation, gate, scope, never_write, entry, distance = rh_rows[0]
    assert parent == "the Riemann Hypothesis"
    assert "de bruijn" in sub_obligation.lower()
    assert "lambda" in sub_obligation.lower()
    assert "lambda <=" in gate.lower()
    assert "not implemented" in scope.lower()
    forbidden_parent_claim = "we " + "prove / disprove" + " the Riemann Hypothesis"
    assert never_write == forbidden_parent_claim
    assert entry == "planned Lambda program"
    assert "non-entry" not in distance.lower()


def test_frontier_docs_mirror_the_ledger_table() -> None:
    spec_rows = _ledger_rows(LEDGER.read_text(encoding="utf-8"))
    docs_rows = _ledger_rows(LEDGER_DOCS.read_text(encoding="utf-8"))
    assert spec_rows, "spec lost the ledger table"
    assert spec_rows == docs_rows, "docs/frontier-ledger.md drifted from spec 07-01"


def test_claim_flag_modules_are_cross_referenced() -> None:
    text = LEDGER.read_text(encoding="utf-8") + "\n" + LEDGER_DOCS.read_text(
        encoding="utf-8"
    )
    missing = [name for name in CLAIM_FLAG_MODULES if name not in text]
    assert not missing, f"ledger missing claim-flag modules {missing}"


def test_distance_column_cites_existing_artifacts() -> None:
    rows = _ledger_rows(LEDGER.read_text(encoding="utf-8"))
    assert rows, "ledger table is empty"
    for key, _parent, _sub, _gate, _scope, never, entry, distance in rows:
        assert never, f"{key} has an empty never-write cell"
        assert distance, f"{key} has an empty distance cell"
        if key == "RH":
            assert entry == "planned Lambda program"
            assert "no implementation" in distance
            continue
        for rel in _BENCHMARK_PATH.findall(distance):
            assert (REPO / rel).is_file(), f"{key} cites missing artifact {rel}"


def test_pade_does_not_consume_dirichlet() -> None:
    """G4: no code path applies spec 03-10 Padé / Borel to Dirichlet output."""
    dirichlet = (
        REPO
        / "packages"
        / "omnibias-core"
        / "src"
        / "omnibias"
        / "core"
        / "verified"
        / "dirichlet.py"
    ).read_text(encoding="utf-8")
    assert not re.search(r"\b(pade|borel|singularity)\b", dirichlet, re.I)
    pade_roots = (
        REPO / "packages" / "omnibias-difference" / "src",
        REPO / "packages" / "omnibias-symbolic" / "src",
        REPO / "packages" / "omnibias-pinn" / "src" / "omnibias" / "pinn" / "certified",
    )
    offenders: list[str] = []
    for root in pade_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if _IMPORT_DIRICHLET.search(text) and _IMPORT_PADE.search(text):
                offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, "Padé and Dirichlet imported together:\n" + "\n".join(
        offenders
    )
    # Vacuity bait: the regexes must be able to fire.
    assert _IMPORT_DIRICHLET.search("from omnibias.core.verified.dirichlet import zeta")
    assert _IMPORT_PADE.search("from omnibias.difference._core.pade import pade_estimate")
