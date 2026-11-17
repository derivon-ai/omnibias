# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``docs/regimes.md`` is a faithful, current view of ``docs/benchmarks/full_scope.json``.

``scripts/build_regime_guide.py`` renders the page from the committed matrix and imports no package. These tests read
the committed JSON and page (cheap), recompute the totals and the loss regimes independently of the script, and render a
small hand-made matrix so the grouping rules are pinned: a win is shown with the losses of its regime, a baseline of
another kind is never mixed into the independent counts, and an incomplete cell is neither a win nor a loss.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_regime_guide.py"
PAGE = ROOT / "docs" / "regimes.md"
VERDICTS = ("big_win", "win", "parity", "loss", "incomplete")


@pytest.fixture(scope="module")
def guide():
    spec = importlib.util.spec_from_file_location("build_regime_guide", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def doc(guide):
    if not guide.SOURCE.exists():
        pytest.skip("no committed full_scope.json")
    return guide.load()


@pytest.fixture(scope="module")
def page(doc, guide):
    if not PAGE.exists():
        pytest.skip("no committed docs/regimes.md")
    return PAGE.read_text()


def _row(operator="op", backend="jax", device="A100", dtype="float64", endpoint="forward", cell="D=16,H=64,depth=2,B=32",
         baseline="folx_forward (folx)", kind="external", verdict="win", ratio=1.5, family="Family"):
    return {"operator": operator, "backend": backend, "device": device, "dtype": dtype, "endpoint": endpoint,
            "cell": cell, "baseline": baseline, "baseline_kind": kind, "verdict": verdict, "ratio": ratio,
            "lower": ratio, "upper": ratio, "family": family}


# -- the committed page ------------------------------------------------------------------------------------------


def test_the_committed_page_is_what_the_matrix_renders_to(guide, doc, page):
    assert guide.render(doc) == page, "docs/regimes.md is stale: run scripts/build_regime_guide.py"


def test_the_script_imports_no_package():
    text = SCRIPT.read_text()
    assert not re.search(r"^\s*(import|from)\s+(omnibias|jax|torch|numpy)\b", text, flags=re.M)


def _summary_numbers(page: str) -> list[int]:
    line = next(ln for ln in page.splitlines() if ln.startswith("| **all** | "))
    return [int(c.replace(",", "")) for c in re.findall(r"\|\s*([0-9][0-9,]*)\s", line + " ")]


def test_the_summary_totals_are_the_totals_of_the_matrix(doc, page):
    rows = doc["rows"]
    own = {"omnibias_variant", "hand_written"}
    ind = Counter(r["verdict"] for r in rows if r["baseline_kind"] == "external")
    mine = Counter(r["verdict"] for r in rows if r["baseline_kind"] in own)
    expected = [sum(ind.values()), *(ind[v] for v in VERDICTS), sum(mine.values()), *(mine[v] for v in VERDICTS)]
    assert _summary_numbers(page) == expected
    # engines on omnibias's projection are in neither column: they are shown, separately, in the family sections
    kinds = Counter(r["baseline_kind"] for r in rows)
    assert sum(kinds.values()) == len(rows)
    assert f"{len(rows):,} cells" in page


def test_every_independent_loss_is_listed_with_its_count(doc, page):
    groups: dict[tuple[str, ...], list[dict]] = {}
    for r in doc["rows"]:
        if r["baseline_kind"] != "external":
            continue
        key = (r["family"], r["operator"], r["backend"], r["device"], r["dtype"], r["endpoint"], r["baseline"].split(" (")[0])
        groups.setdefault(key, []).append(r)
    section = page.split("## Where an independent baseline is faster", 1)[1].split("\n## ", 1)[0]
    listed = {}
    for line in section.splitlines():
        if line.startswith("| ") and not line.startswith("| family") and not line.startswith("|---"):
            cols = [c.strip() for c in line.strip().strip("|").split(" | ")]
            listed[(*cols[:6], cols[6].strip("`"))] = (int(cols[7].replace(",", "")), int(cols[8].replace(",", "")))
    expected = {k: (len(v), sum(r["verdict"] == "loss" for r in v)) for k, v in groups.items() if any(r["verdict"] == "loss" for r in v)}
    assert listed == expected
    assert sum(loss for _, loss in expected.values()) == sum(
        r["verdict"] == "loss" for r in doc["rows"] if r["baseline_kind"] == "external"
    )


def test_every_regime_row_adds_up_and_no_cell_is_dropped(doc, page):
    total = 0
    for line in page.splitlines():
        if not line.startswith("| ") or line.startswith("| operator | back end") or line.startswith("|---"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cols) != 14:  # a group row of a family section: five labels, regime, six counts, ratio, loss regime
            continue
        cells, *verdicts = (int(c.replace(",", "")) for c in cols[6:12])
        assert cells == sum(verdicts), line
        total += cells
    assert total == len(doc["rows"])


# -- the rules, on a matrix small enough to read ---------------------------------------------------------------------


def test_a_win_is_shown_beside_the_losses_of_its_regime_and_kinds_are_not_mixed(guide):
    rows = [
        _row(cell="D=16,H=64,depth=2,B=32", verdict="big_win", ratio=3.0),
        _row(cell="D=1024,H=256,depth=4,B=2048", verdict="loss", ratio=0.6),
        _row(cell="D=64,H=64,depth=2,B=32", verdict="parity", ratio=1.0),
        _row(cell="D=64,H=64,depth=2,B=32", verdict="incomplete", ratio=None, baseline="none qualified"),
        _row(cell="D=64,H=64,depth=2,B=32", verdict="loss", ratio=0.8, kind="omnibias_variant", baseline="closed (c)"),
        _row(cell="D=64,H=64,depth=2,B=32", verdict="win", ratio=1.4, kind="engine_on_omnibias_projection", baseline="folx (f)"),
    ]
    text = guide.render({"rows": rows})
    independent = text.split("### Independent baselines", 1)[1].split("### ", 1)[0]
    row = next(ln for ln in independent.splitlines() if ln.startswith("| op | jax | A100 | float64 | forward"))
    cols = [c.strip() for c in row.strip().strip("|").split(" | ")]
    assert cols[5] == "D 16-1,024, H 64-256, depth 2-4, B 32-2,048"  # the regime the cells cover
    assert cols[6:12] == ["4", "1", "0", "1", "1", "1"]  # cells, big win, win, parity, loss, incomplete
    assert cols[12] == "0.60-3.00×"  # incomplete cells carry no ratio
    assert cols[13] == "D 1,024, H 256, depth 4, B 2,048"  # where the loss is
    # the other kinds have their own tables and do not enter the independent counts
    assert "### Generic engines fed omnibias's own projection" in text
    assert "### omnibias's own variants and hand-written closed forms" in text
    own = text.split("### omnibias's own variants and hand-written closed forms", 1)[1]
    assert "| op | jax | A100 | float64 | forward | D 64, H 64, depth 2, B 32 | 1 | 0 | 0 | 0 | 1 | 0 |" in own
    # the faster-baseline table lists the independent loss only, with its regime
    faster = text.split("## Where an independent baseline is faster", 1)[1].split("\n## ", 1)[0]
    assert "`folx_forward`" in faster and "`closed`" not in faster
    assert "| 3 | 1 | 1 | 0.60-0.60× | D 1,024, H 256, depth 4, B 2,048 |" in faster  # the baseline's own three cells


def test_bounds_are_read_from_the_cell_label_without_matching_inside_other_names(guide):
    assert guide.read_number("D=3072,H=96,depth=5", ("D", "dim")) == 3072.0
    assert guide.read_number("model=m,dim=41,hidden=64", ("D", "dim")) == 41.0
    assert guide.read_number("ND=7,H=96", ("D", "dim")) is None
    assert guide.read_number("D=2,rel_l2<=1e-3,B=1,024", ("B", "batch")) == 1024.0
    assert guide.regime_of([_row(cell="poisson_d100")]) == "-"
    assert guide.span([None, 4.0, 4096.0]) == "4-4,096"


def test_check_flags_a_stale_page_and_accepts_a_current_one(guide, monkeypatch, tmp_path):
    out = tmp_path / "regimes.md"
    monkeypatch.setattr(guide, "OUT", out)
    monkeypatch.setattr(guide, "load", lambda: {"rows": [_row()]})
    assert guide.main(["--check"]) == 1  # missing
    assert guide.main([]) == 0 and out.exists()
    assert guide.main(["--check"]) == 0
    out.write_text(out.read_text() + "edited by hand\n")
    assert guide.main(["--check"]) == 1
