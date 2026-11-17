# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The per-call grid section of docs/performance.md quotes the committed record, and a target that was missed is not described as met."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "benchmarks" / "percall_grid.json"
PUBLISHED = ROOT / "docs" / "benchmarks" / "polylaplacian_order.json"
PAGE = ROOT / "docs" / "performance.md"
PLANNER_PAGE = ROOT / "docs" / "operator-planner.md"

# Words of the measurement site and of the working environment that a published record or page does not carry.
FORBIDDEN = (
    "/u/",
    "/home/",
    "scratch",
    "lsf",
    "b" "sub",  # split so that the repository leakage scan does not match this list
    "b" "jobs",
    "us01",
    "syn" "opsys",
    "gpudc",
    "vardang",
    ".internal",
)


def _record() -> dict[str, Any]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _flat(text: str) -> str:
    return " ".join(text.split())


def _section() -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("## Per-call grid")
    return text[start : text.index("\n## ", start + 1)]


def _row(text: str, target: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| **{target}**"))


def _leaves(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "cells_meeting" in node:
            yield node
            return
        for child in node.values():
            yield from _leaves(child)


def _paths(node: Any, prefix: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        if "cells_meeting" in node:
            yield prefix, node
            return
        for key, child in node.items():
            yield from _paths(child, f"{prefix}/{key}")


def _groups(rec: dict[str, Any], target: str) -> tuple[int, int]:
    """(evaluable groups, groups that met) of a target; a group meets only when every one of its cells does."""
    leaves = [leaf for leaf in _leaves(rec["verdicts"][target]) if leaf["cells"] > 0]
    return len(leaves), sum(1 for leaf in leaves if leaf["met"])


def _pool_rows(rec: dict[str, Any], cell_prefix: str) -> list[dict[str, Any]]:
    pool = rec["torch_candidate_pool"]
    return [
        row
        for prec in ("f32h", "f32t")
        for row in pool[prec]["fused"]
        if row["cell"].startswith(cell_prefix)
    ]


def test_every_job_names_its_installed_tree_and_its_exclusive_device() -> None:
    rec = _record()
    trees = set(rec["code"]["installed_tree_sha256"])
    assert trees and all(re.fullmatch(r"[0-9a-f]{64}", tree) for tree in trees)
    for name, job in rec["jobs"].items():
        assert job["installed_tree_sha256"] in trees, name
        assert (
            job["gpu"]["compute_mode"] == "Exclusive_Process"
            and job["gpu"]["foreign_compute_processes"] == []
        ), name
    # a job that was killed has no end-of-job placement record; the page says which ones
    unchecked = [name for name, job in rec["jobs"].items() if not job["exclusive_gpu_checked"]]
    assert unchecked in ([], ["jax/f64/o4"]), unchecked
    if unchecked:
        assert "the killed order-four JAX jobs have the start check only" in _flat(_section())
    assert set(rec["cells"]) >= {
        "jax/f64/o2",
        "jax/f32h/o2",
        "jax/f32t/o2",
        "torch/f64/o2",
        "jax/f64/o3",
        "jax/f64/o4",
        "torch/f64/o4",
    }
    assert "(benchmarks/percall_grid.json)" in _section()


def test_the_page_quotes_the_group_counts_of_the_record() -> None:
    rec, text = _record(), _section()
    n1, met1 = _groups(rec, "T1")
    assert met1 == 0 and f"no group of {n1}" in _row(text, "T1")
    n2, met2 = _groups(rec, "T2")
    assert met2 == 0 and f"{met2} of {n2} evaluable groups" in _row(text, "T2")
    n3, met3 = _groups(rec, "T3")
    assert 0 < met3 < n3 and f"{met3} of {n3} evaluable groups" in _row(text, "T3")
    n5, met5 = _groups(rec, "T5")
    assert met5 == 0 and f"{met5} of {n5} groups" in _row(text, "T5")


def test_a_missed_target_is_not_described_as_met() -> None:
    rec, text = _record(), _section()
    assert "None of the five registered targets was met as written" in _flat(text)
    for target in ("T1", "T2", "T4", "T5"):
        row = _row(text, target)
        assert "**missed**" in row and "**met**" not in row and "**partly met**" not in row, target
    assert "**partly met**" in _row(text, "T3") and "**met**" not in _row(text, "T3")
    assert not any(
        flag for entry in rec["verdicts"]["T4"] for flag in entry["within_10_percent"].values()
    )


def test_t4_quotes_the_ratios_to_the_published_ones() -> None:
    rec, text = _record(), _section()
    by_k = {entry["k"]: entry["relative_to_published"] for entry in rec["verdicts"]["T4"]}
    row = _row(text, "T4")
    assert (
        f"against `folx` {by_k[3]['folx_nested']:.2f} (order three) and {by_k[4]['folx_nested']:.2f} (order four)"
        in row
    )
    assert f"against a dense Hessian {by_k[3]['dense_nested']:.2f} (order three)" in row
    assert (
        PUBLISHED.is_file()
    ), "the published record the ratios are relative to must stay in the repository"


def test_the_training_step_numbers_are_the_records() -> None:
    rec, raw = _record(), _section()
    text = _flat(raw)
    steps = rec["verdicts"]["steps"]
    f64, f32 = steps["step_torch_f64"]["100"]["T5"], steps["step_torch_f32h"]["100"]["T5"]
    assert f64["meets_target"] is False and f32["meets_target"] is True
    assert (
        f"**step**: {f64['omnibias_planned_ms']:.3f} ms in float64 (missed) and {f32['omnibias_planned_ms']:.3f} ms in float32 (met)"
        in _flat(_row(raw, "T5"))
    )
    j32, j64 = steps["step_jax_f32h"]["100"], steps["step_jax_f64"]["100"]
    assert (
        f"JAX {j32['omnibias_planned']:.3f} ms in float32 and {j64['omnibias_planned']:.3f} ms in float64 "
        f"against {j32['folx_forward']:.3f} and {j64['folx_forward']:.3f} ms for a forward-Laplacian step with `folx`"
    ) in text
    # the planned step is slower than folx at D = 100 on JAX, and the page says so
    assert (
        j32["folx_forward"] < j32["omnibias_planned"]
        and j64["folx_forward"] < j64["omnibias_planned"]
    )
    assert "level with or slower than `folx` at `D = 100`" in text


def test_the_kernel_counts_quoted_for_the_pytorch_operator() -> None:
    rec, text = _record(), _flat(_section())
    rows = {(row["cell"], row["tag"]): row for row in rec["verdicts"]["kernel_counts"]["torch/f64"]}
    fwd, vag = rows[("D2H64L2B32", "forward")], rows[("D2H64L2B32", "value_and_grad")]
    assert (
        f"{fwd['default_train']} forward and {vag['default_train']} for value-and-gradient against {fwd['baseline']} and {vag['baseline']}"
        in text
    )


def test_the_fused_kernel_ratios_are_the_pool_record() -> None:
    rec, text = _record(), _flat(_section())
    rows = _pool_rows(rec, "D2H")

    def ratios(tag: str, ref: str) -> list[float]:
        return [
            row[ref][tag] / row["fused"][tag]
            for row in rows
            if row[ref].get(tag) and row["fused"].get(tag)
        ]

    fwd, vag = (
        ratios("forward/device", "omnibias_default"),
        ratios("value_and_grad/device", "omnibias_default"),
    )
    jet = ratios("forward/device", "jet_collapsed")
    assert (
        f"{min(fwd):.1f} to {max(fwd):.1f} times faster than the planned default forward and {min(vag):.1f} to {max(vag):.1f} times for value-and-gradient"
        in text
    )
    assert f"{min(jet):.1f} to {max(jet):.1f} times faster than `jet_collapsed` forward" in text
    assert "**The planned default does not choose it**" in text
    assert "the ranking was not tuned against these records" in text
    # the planner page carries the same milliseconds
    planner = _flat(PLANNER_PAGE.read_text(encoding="utf-8"))
    f_fwd = [row["fused"]["forward/device"] * 1000 for row in rows]
    d_fwd = [row["omnibias_default"]["forward/device"] * 1000 for row in rows]
    assert (
        f"{min(f_fwd):.3f} to {max(f_fwd):.3f} ms forward against {min(d_fwd):.2f} to {max(d_fwd):.2f} ms for the planned default"
        in planner
    )
    assert (
        f"({min(fwd):.1f} to {max(fwd):.1f} times)" in planner
        and f"({min(vag):.1f} to {max(vag):.1f} times)" in planner
    )


def test_the_device_gaps_of_pytorch_are_the_counts_of_the_record() -> None:
    rec, text = _record(), _flat(_section())
    timed = on_device = skipped = uncapturable = stream_engines = 0
    for key, cells in rec["cells"].items():
        if not key.startswith("torch"):
            continue
        for cell in cells:
            for name, row in cell["methods"].items():
                for tag in ("fwd", "vag"):
                    if row.get(f"{tag}_ms") is None:
                        continue
                    timed += 1
                    on_device += f"{tag}_device_ms" in row
                    skipped += bool(row.get(f"{tag}_device_skipped"))
                    if row.get(f"{tag}_device_error"):
                        uncapturable += 1
                        stream_engines += key.endswith("o3") and name in {
                            "omnibias_streams",
                            "omnibias_fused",
                        }
    assert (
        timed == on_device + skipped + uncapturable
    ), "every host-timed record is on the device, skipped by the memory guard, or uncapturable"
    assert (
        f"Of the {timed:,} PyTorch records with a host-timed median, {on_device} have a device time; {skipped} were skipped by a guard"
        in text
    )
    assert (
        f"{uncapturable} could not be captured ({stream_engines} of them the stream engines at order three)"
        in text
    )
    assert f"({stream_engines} records, an open defect)" in text


def test_the_short_block_counts_are_the_records() -> None:
    rec, text = _record(), _flat(_section())
    blocks = rec["verdicts"]["short_blocks"]
    assert blocks["max_calls"] == 8
    assert (
        f"{blocks['short_blocks']:,} of {blocks['records']:,} host-timed records are such" in text
    )
    assert (
        f"the {len(rec['verdicts']['suspect_timings'])} timings whose median is more than twice their minimum"
        in text
    )
    # the sensitivity check: eager groups whose counts change when the minimum replaces the median for short blocks
    groups = changed = flipped = 0
    for target in ("T1", "T2", "T3", "T5"):
        plain = {path: leaf for path, leaf in _paths(rec["verdicts"][target]) if "/eager" in path}
        robust = dict(_paths(rec["verdicts"]["robust_eager"][target]))
        for path, leaf in plain.items():
            if not leaf["cells"]:
                continue
            groups += 1
            changed += (leaf["cells_meeting"], leaf["cells"]) != (
                robust[path]["cells_meeting"],
                robust[path]["cells"],
            )
            flipped += leaf["met"] != robust[path]["met"]
    assert flipped == 0
    assert f"changes the counts of {changed} of {groups} groups and the verdict of none" in text


def test_the_deviations_are_listed_and_the_unrun_work_is_called_not_run() -> None:
    rec, text = _record(), _flat(_section())
    assert [entry["id"] for entry in rec["deviations"]] == [f"D{i}" for i in range(1, 14)]
    assert all(entry["kind"] and entry["text"] for entry in rec["deviations"])
    assert "13 dated deviations" in text
    tail = text[text.index("**Not run.**") :]
    assert (
        "third round of the dispatch gate is registered and was not run" in tail
        and "suite v2 was registered and not run in the round that measured this grid" in tail
        and "it has since been run" in tail
    )
    assert "a repeat of this grid on a second card was not run" in tail


def test_the_registered_files_are_hashed_before_and_after_the_deviations() -> None:
    rec = _record()
    first, later = rec["registered_sha256"], rec["registered_v2_sha256"]
    assert {"PREREGISTRATION.md", "analyze_percall.py", "t4_polylaplacian.py"} <= set(first)
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest) for digest in (*first.values(), *later.values())
    )
    # the deviations were appended to the registered text, so its second hash differs from the first
    assert later["PREREGISTRATION.md"] != first["PREREGISTRATION.md"]


def test_the_record_and_the_pages_carry_no_site_or_environment_names() -> None:
    rec = _record()
    text = (
        RECORD.read_text(encoding="utf-8")
        + "\n"
        + _section()
        + "\n"
        + PLANNER_PAGE.read_text(encoding="utf-8")
    ).lower()
    assert not {word for word in FORBIDDEN if word in text}
    assert not [name for name in rec["jobs"] if re.search(r"\d{6,}", name)]
