# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The end-to-end section of docs/operator-planner.md quotes the committed suite record and does not claim a training speedup."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "benchmarks" / "planner" / "e2e_suite_a100.json"
DOC = ROOT / "docs" / "operator-planner.md"


def _section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## End-to-end training")
    return text[start : text.index("\n## ", start + 1)]


def test_the_page_quotes_the_counts_of_the_record() -> None:
    rec, text = json.loads(RECORD.read_text(encoding="utf-8")), _section()
    jax = rec["jax_suite"]
    assert f"{jax['headlines_compile_included']} of {jax['rows']} problem-target rows" in text
    assert f"{jax['losses']} rows are losses" in text
    assert f"{jax['no_pair_reached']} are rows in which no pair reached" in text
    assert f"{jax['headlines_compile_excluded']} rows reach the" in text
    pred = jax["predictions"]
    assert f"{pred['hit']} hit, {pred['missed']} missed" in text.replace("two", "2").replace("five", "5").replace("one was", "1 was")


def test_the_torch_subset_sentence_matches_the_record() -> None:
    rec, text = json.loads(RECORD.read_text(encoding="utf-8")), _section()
    sub = rec["torch_subset"]
    assert sub["headlines_compile_included"] == 0
    assert "no headline" in text and "no engine reached the loosest target at `D = 1000`" in text
    step = sub["post_hoc_step_and_memory"]
    omni, nested = step["poisson_d1000/omnibias"], step["poisson_d1000/nested"]
    assert f"{omni['step_ms']:.1f} ms and {omni['peak_device_gib']:.2f} GiB against {nested['step_ms']:.1f} ms and {nested['peak_device_gib']:.2f} GiB" in text


def test_a_registered_headline_that_fails_is_not_described_as_a_speedup() -> None:
    rec, text = json.loads(RECORD.read_text(encoding="utf-8")), _section()
    if rec["jax_suite"]["headlines_compile_included"] == 0:
        assert "not as a demonstrated end-to-end training speedup" in text
        assert not re.search(r"\b\d+(\.\d+)?x faster (to|in) (train|converge)", text)
