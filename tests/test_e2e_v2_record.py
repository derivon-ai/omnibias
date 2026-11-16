# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The suite v2 section of docs/operator-planner.md quotes the committed record, shows its losses and claims no more than the headline holds."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "benchmarks" / "planner" / "e2e_suite_v2_a100.json"
FARM = ROOT / "docs" / "benchmarks" / "farm_time_to_target_v2.json"
FULL_SCOPE = ROOT / "docs" / "benchmarks" / "full_scope.json"
DOC = ROOT / "docs" / "operator-planner.md"
SECOND_ORDER = re.compile(r"^(gn|spring|lmtr|engd):")
LABEL = {"loss": "loss", "baseline_faster": "baseline faster or equal"}
TABLE_ROWS = {
    "excl": "compile excluded, every baseline (primary)",
    "exact_excl": "compile excluded, exact baselines only",
    "incl": "cold compile included, every baseline",
    "warm_incl": "warm compile included, every baseline",
    "exact_incl": "cold compile included, exact baselines only",
    "orig_incl": "the 175 registered pairs, cold compile included (the original rule)",
    "orig_excl": "the 175 registered pairs, compile excluded",
}


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("### Suite v2")
    return text[start : text.index("\n### Suite v1", start)]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _headline_line(h: dict) -> str:
    return (
        f"| `{h['problem']}` | {h['target']:g} | `{h['candidate']}` | {h['candidate_s']:.3g} | `{h['baseline']}` | "
        f"{h['baseline_s']:.3g} | {h['geomean']:.2f} | {h['lower95']:.2f} |"
    )


def _loss_line(h: dict) -> str:
    return (
        f"| `{h['problem']}` | {h['target']:g} | {LABEL[h['class']]} | `{h['candidate']}` | `{h['baseline']}` | "
        f"{h['geomean']:.2f} | {h['lower95']:.2f} |"
    )


def test_the_record_is_the_registered_float64_suite() -> None:
    rec = _record()
    setup = rec["setup"]
    assert (setup["problems"], setup["seeds"], setup["dtype"], setup["time_cap_s"]) == (13, 5, "float64", 900)
    assert setup["cells"] == 265 * 5
    assert re.fullmatch(r"[0-9a-f]{64}", setup["installed_tree_sha256"])
    assert "--xla_gpu_autotune_level=0" in setup["xla_flags"]
    assert rec["row_policy_check"]["rows_mismatch"] == [] and rec["row_policy_check"]["solver_unexpected"] == []
    assert set(rec["rows_by_comparison"]) == set(TABLE_ROWS)
    for body in rec["rows_by_comparison"].values():
        assert body["rows"] == 39 and body["eligible_rows"] == 32
        parts = ("significant", "not_significant", "baseline_faster", "loss", "no_pair_reached", "censored", "no_candidate_pair", "no_baseline_cells")
        assert sum(body.get(k, 0) for k in parts) == body["eligible_rows"]


def test_the_page_quotes_the_comparison_table_of_the_record() -> None:
    rec, text = _record(), _section()
    for tag, label in TABLE_ROWS.items():
        body = rec["rows_by_comparison"][tag]
        no_ratio = body.get("no_pair_reached", 0) + body.get("censored", 0) + body.get("no_candidate_pair", 0) + body.get("no_baseline_cells", 0)
        line = (
            f"| {label} | {body['significant']} | {body['not_significant']} | {body.get('baseline_faster', 0)} | "
            f"{body.get('loss', 0)} | {no_ratio} |"
        )
        assert line in text, line
    excl = rec["rows_by_comparison"]["excl"]
    assert f"{excl['significant']} of {excl['eligible_rows']} eligible rows meet the headline" in _flat(text)


def test_every_headline_row_is_on_the_page_and_follows_the_registered_rule() -> None:
    rec, text = _record(), _section()
    rows = rec["headline_rows_compile_excluded"]
    assert len(rows) == rec["rows_by_comparison"]["excl"]["significant"]
    assert all(h["lower95"] >= 2 for h in rows)
    for h in rows:
        assert _headline_line(h) in text, h
    # the table holds exactly the rows of the record
    table = [line for line in text.splitlines() if re.match(r"\| `\w+` \| [\d.]+ \| `\w+:\w+` \| ", line)]
    assert len(table) == len(rows)
    problems = {h["problem"] for h in rows}
    assert f"on {len(problems)} of the 13 problems" in _flat(text)
    # which optimizers win: all second order on the structured Gram, never a first-order candidate
    assert all(SECOND_ORDER.match(h["candidate"]) and h["candidate"].endswith(":structured") for h in rows)
    adam_or_lbfgs = sum(h["baseline"].startswith(("adam:", "lbfgs:")) for h in rows)
    assert len(rows) - adam_or_lbfgs == sum(h["baseline"].startswith("spring:") for h in rows) == 1
    flat = _flat(text)
    assert f"in {adam_or_lbfgs} of the {len(rows)} rows and an explicit-Jacobian SPRING pair in one" in flat
    # the sentence that quotes the primary count says that compile time is excluded
    assert f"With compile excluded, {len(rows)} of {rec['rows_by_comparison']['excl']['eligible_rows']} eligible rows meet the headline" in flat


def test_the_break_even_sentence_matches_the_record() -> None:
    rec, text = _record(), _flat(_section())
    rows = rec["headline_rows_compile_excluded"]
    none = sum(h["break_even_steps"] is None for h in rows)
    zero = sum(h["break_even_steps"] == 0 for h in rows)
    assert none + zero == len(rows)
    assert f"In {none} of the {len(rows)} rows the candidate compiles slower and is not faster per step" in text
    assert f"in the other {zero} it compiles no slower (break-even 0)" in text
    ratios = [h["geomean"] for h in rows]
    assert f"These ratios are {min(ratios):.2f} to {max(ratios):.2f}" in text
    extra = max(h["extra_compile_s"] for h in rows)
    nets = [h["net_at_target_s"] for h in rows]
    assert min(nets) > 0
    assert f"extra cold compile is at most {extra:.1f} s" in text
    assert f"the saving at the target is {min(nets):.1f} s to {max(nets):.1f} s, positive in every row" in text


def test_every_loss_and_parity_row_is_on_the_page() -> None:
    rec, text = _record(), _section()
    rows = rec["loss_rows_compile_excluded"]
    excl = rec["rows_by_comparison"]["excl"]
    assert len(rows) == excl["loss"] + excl["baseline_faster"]
    for h in rows:
        assert _loss_line(h) in text, h
    assert all(h["lower95"] < 1 for h in rows)
    assert any(h["problem"] == "triharmonic" for h in rows) and any(h["problem"] == "heat_hd_d1000" for h in rows)


def test_the_prediction_counts_are_quoted() -> None:
    rec, text = _record(), _flat(_section())
    pred = rec["predictions"]
    assert pred["hit"] + pred["missed"] + pred["not_scored"] == len(pred["verdicts"]) == 19
    assert f"Of 19 registered predictions (10 of the original registration, 9 of Addendum A), {pred['hit']} hit, {pred['missed']} missed and {pred['not_scored']} were not scored" in text


def test_the_full_scope_rows_restate_the_record_and_never_score_a_missing_ratio() -> None:
    rec = _record()
    farm = json.loads(FARM.read_text(encoding="utf-8"))
    assert farm["schema"] == "omnibias/farm-ratios/v1"
    rows = farm["rows"]
    assert all(r["basis"] == "t95_seeds" and r["cell"].startswith("suite_v2,") for r in rows)
    for r in rows:
        if r["ratio"] is None:
            assert r["lower"] is None and r["upper"] is None and r["baseline"] == "strongest baseline pair"
        else:
            assert r["lower"] <= r["ratio"] <= r["upper"]
    primary = [r for r in rows if r["dtype"] == "float64" and r["endpoint"] == "time to target, compile excluded (suite v2)"]
    assert len(primary) == 39
    assert sum(r["lower"] is not None and r["lower"] >= 2 for r in primary) == rec["rows_by_comparison"]["excl"]["significant"]
    matrix = json.loads(FULL_SCOPE.read_text(encoding="utf-8"))
    mine = [r for r in matrix["rows"] if r["record"].endswith("farm_time_to_target_v2.json")]
    assert len(mine) == len(rows)
    assert all(r["verdict"] == "incomplete" for r in mine if r["ratio"] is None)
    assert not any(r["verdict"] in {"big_win", "win"} for r in mine if r["ratio"] is None)


def test_the_page_does_not_state_a_speedup_the_headline_does_not_support() -> None:
    rec, text = _record(), _flat(_section())
    assert not re.search(r"\b\d+(\.\d+)?x faster (to|in) (train|converge)", text)
    # a first-order candidate never meets the bound, and the page says so only because that is true
    assert all(SECOND_ORDER.match(h["candidate"]) for h in rec["headline_rows_compile_excluded"])
    assert "No first-order candidate" in text
    assert "Because it was written knowing v1's outcome" in text and "not independent" in text
    assert "A faster first-order step is not what wins." in text
    # the float32 run is secondary and incomplete; the page must not widen the claim with it
    assert "does not extend it" in text and "Float32 meets the bound on no problem on which float64 does not" in text
    assert rec["float32_highest"]["rows_by_comparison"]["excl"]["significant"] < rec["rows_by_comparison"]["excl"]["significant"]


INCOMPLETE_FLOAT32 = {"heat1d", "allen_cahn", "cahn_hilliard"}
PRIMARY = "time to target, compile excluded (suite v2)"


def test_the_float32_run_is_quoted_for_its_ten_complete_problems_only() -> None:
    rec, text = _record(), _flat(_section())
    f32 = rec["float32_highest"]["rows_by_comparison"]
    assert f32["excl"]["rows"] == f32["incl"]["rows"] == 30 and f32["excl"]["eligible_rows"] == 22
    sig_excl, sig_incl = f32["excl"]["significant"], f32["incl"]["significant"]
    assert f"Over the 10 complete problems (30 rows, 22 eligible), {sig_excl} eligible rows meet the headline with compile excluded and {sig_incl} with the cold compile included" in text
    assert "complete for 10 of the 13 problems" in text and "no float32 row is reported for them or counted anywhere" in text
    farm = [r for r in json.loads(FARM.read_text(encoding="utf-8"))["rows"] if r["dtype"] == "float32"]
    problems = {r["cell"].split(",")[1] for r in farm}
    assert len(problems) == 10 and not problems & INCOMPLETE_FLOAT32
    primary = [r for r in farm if r["endpoint"] == PRIMARY]
    assert len(primary) == 30
    meets = [r for r in primary if r["lower"] is not None and r["lower"] >= 2]
    assert len(meets) == sig_excl
    assert f"ratios {min(r['ratio'] for r in meets):.2f} to {max(r['ratio'] for r in meets):.2f} (geometric means) and lower bounds {min(r['lower'] for r in meets):.2f} to {max(r['lower'] for r in meets):.2f}" in text
    assert {r["cell"].split(",")[1] for r in meets} == {"heat_hd_d100", "helmholtz", "hjb_d100", "poisson_d100"}
