# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Build ``docs/regimes.md``: where omnibias is faster, level or slower, by regime, from ``full_scope.json``.

Nothing is measured and nothing is typed in by hand: the page is a view of ``docs/benchmarks/full_scope.json``
(see ``scripts/build_full_scope.py`` for the verdicts and their intervals). Cells are grouped by family, operator,
back end, device, dtype and endpoint. Each group shows the range of ``D``, ``H``, depth, order, batch and grid size
that its cells cover (parsed from the cell label), and the verdict counts side by side for each kind of baseline, so a
win is never shown without the losses of the same regime. The script imports no package. ``--check`` fails when
``docs/regimes.md`` is not what the JSON renders to.

The page is not a planner. It says what was measured; ``explain()`` and ``mode="find"`` say what a given problem does.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "benchmarks" / "full_scope.json"
OUT = ROOT / "docs" / "regimes.md"

VERDICTS = ("big_win", "win", "parity", "loss", "incomplete")
KINDS = (
    ("external", "Independent baselines"),
    ("engine_on_omnibias_projection", "Generic engines fed omnibias's own projection"),
    ("own", "omnibias's own variants and hand-written closed forms"),
)
KIND_NOTE = {
    "external": (
        "Nested autodiff, dense Hessian and Jacobian, `jax.experimental.jet`, `folx`, PyTorch `jet_collapsed`, "
        "Taylor mode, DeepXDE, GMRES and explicit grids. A loss here is a regime where the baseline is faster."
    ),
    "engine_on_omnibias_projection": (
        "The generic engine runs on the reduced function omnibias projects, so the projection is shared; the "
        "comparison isolates the engine."
    ),
    "own": (
        "A loss here is one omnibias program beating another (a closed form, a Riccati recurrence, an earlier "
        "compiler) on the same XLA. Say which variant won; it is not a loss against another package."
    ),
}
OWN_KINDS = frozenset({"omnibias_variant", "hand_written"})

# parsed from ``cell``; the first spelling found wins
BOUNDS = (
    ("D", ("D", "dim")),
    ("H", ("H", "hidden", "width")),
    ("depth", ("depth", "L")),
    ("order", ("order",)),
    ("B", ("B", "batch")),
    ("N", ("N",)),
)
GROUP_KEYS = ("operator", "backend", "device", "dtype", "endpoint")


def kind_of(row: dict[str, Any]) -> str:
    return "own" if row["baseline_kind"] in OWN_KINDS else str(row["baseline_kind"])


def read_number(cell: str, names: Sequence[str]) -> float | None:
    for name in names:
        found = re.search(rf"(?:^|,)\s*{re.escape(name)}=(-?[0-9][0-9_,]*(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)(?=,|$)", cell)
        if found:
            try:
                return float(found.group(1).replace(",", "").replace("_", ""))
            except ValueError:
                continue
    return None


def fmt_number(x: float) -> str:
    return f"{int(x):,}" if float(x).is_integer() else f"{x:g}"


def span(values: Iterable[float | None]) -> str:
    found = [v for v in values if v is not None]
    if not found:
        return "-"
    lo, hi = min(found), max(found)
    return fmt_number(lo) if lo == hi else f"{fmt_number(lo)}-{fmt_number(hi)}"


def regime_of(rows: Sequence[dict[str, Any]]) -> str:
    """``D 16-64, H 64, depth 2`` for the cells of ``rows`` (only the labels the cells carry)."""
    parts = []
    for label, names in BOUNDS:
        text = span(read_number(r["cell"], names) for r in rows)
        if text != "-":
            parts.append(f"{label} {text}")
    return ", ".join(parts) if parts else "-"


def fmt_ratio(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:,.0f}" if x >= 100 else f"{x:.2f}" if x < 10 else f"{x:.1f}"


def ratio_range(rows: Sequence[dict[str, Any]]) -> str:
    ratios = [r["ratio"] for r in rows if r["verdict"] != "incomplete" and r["ratio"] is not None]
    if not ratios:
        return "-"
    return f"{fmt_ratio(min(ratios))}-{fmt_ratio(max(ratios))}×"


def counts(rows: Sequence[dict[str, Any]]) -> Counter[str]:
    return Counter(r["verdict"] for r in rows)


def anchor(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def group(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def cell_key(r: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(r[k]) for k in GROUP_KEYS)


def table_row(key: tuple[str, ...], rows: Sequence[dict[str, Any]]) -> str:
    c = counts(rows)
    losses = [r for r in rows if r["verdict"] == "loss"]
    where = regime_of(losses) if losses else "-"
    return (
        f"| {' | '.join(key)} | {regime_of(rows)} | {len(rows):,} | {c['big_win']:,} | {c['win']:,} | {c['parity']:,} | "
        f"{c['loss']:,} | {c['incomplete']:,} | {ratio_range(rows)} | {where} |"
    )


HEADER = (
    "| operator | back end | device | dtype | endpoint | cells cover | cells | big win | win | parity | loss | "
    "incomplete | ratio range | loss cells cover |\n"
    "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"
)


def summary_table(rows: Sequence[dict[str, Any]], families: Sequence[str]) -> list[str]:
    lines = [
        "| family | independent cells | big win | win | parity | loss | incomplete | "
        "own-variant cells | big win | win | parity | loss | incomplete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_ind: Counter[str] = Counter()
    total_own: Counter[str] = Counter()
    for fam in families:
        ind = counts([r for r in rows if r["family"] == fam and kind_of(r) == "external"])
        own = counts([r for r in rows if r["family"] == fam and kind_of(r) == "own"])
        n_ind = sum(ind.values())
        n_own = sum(own.values())
        total_ind.update(ind)
        total_own.update(own)
        lines.append(
            f"| [{fam}](#{anchor(fam)}) | {n_ind:,} | {ind['big_win']:,} | {ind['win']:,} | {ind['parity']:,} | "
            f"{ind['loss']:,} | {ind['incomplete']:,} | {n_own:,} | {own['big_win']:,} | {own['win']:,} | "
            f"{own['parity']:,} | {own['loss']:,} | {own['incomplete']:,} |"
        )
    lines.append(
        f"| **all** | {sum(total_ind.values()):,} | {total_ind['big_win']:,} | {total_ind['win']:,} | "
        f"{total_ind['parity']:,} | {total_ind['loss']:,} | {total_ind['incomplete']:,} | {sum(total_own.values()):,} | "
        f"{total_own['big_win']:,} | {total_own['win']:,} | {total_own['parity']:,} | {total_own['loss']:,} | "
        f"{total_own['incomplete']:,} |"
    )
    return lines


def slower_than_baseline(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Every independent baseline that is faster in at least one cell, with the regime of those cells."""
    cells: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for r in rows:
        if kind_of(r) == "external":
            cells.setdefault((r["family"], *cell_key(r), str(r["baseline"]).split(" (")[0]), []).append(r)
    lines = [
        "| family | operator | back end | device | dtype | endpoint | baseline | cells | loss | parity | ratio of the losses "
        "| loss cells cover |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    ranked = []
    for key, members in cells.items():
        losses = [r for r in members if r["verdict"] == "loss"]
        if losses:
            ranked.append((-len(losses), key, members, losses))
    for _, key, members, losses in sorted(ranked, key=lambda t: (t[0], t[1])):
        family, operator, backend, device, dtype, endpoint, baseline = key
        ratios = [r["ratio"] for r in losses]
        lines.append(
            f"| {family} | {operator} | {backend} | {device} | {dtype} | {endpoint} | `{baseline}` | {len(members):,} | "
            f"{len(losses):,} | {sum(r['verdict'] == 'parity' for r in members):,} | "
            f"{min(ratios):.2f}-{max(ratios):.2f}× | {regime_of(losses)} |"
        )
    return lines


def render(doc: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = doc["rows"]
    families = sorted({str(r["family"]) for r in rows})
    overall = counts(rows)
    ind_all = counts([r for r in rows if kind_of(r) == "external"])
    out = [
        "# Regimes: where omnibias wins, ties and loses",
        "",
        "<!-- Generated by scripts/build_regime_guide.py from docs/benchmarks/full_scope.json. Do not edit; run the script. -->",
        "",
        "This page is a view of [`full_scope.json`](benchmarks/full_scope.json), the matrix of every measured cell against every "
        f"baseline ({len(rows):,} cells; {sum(ind_all.values()):,} against independent baselines). It says what was measured, "
        "by regime, with the cells that lose next to the cells that win. It does not choose a program for your problem: "
        "[`explain()` and `mode=\"find\"`](operator-planner.md#planning-for-training) do, on your shape and device.",
        "",
        "**Reading a cell.** The ratio is baseline time over omnibias time; each cell carries a 95 % interval and the verdict "
        "reads the interval: `big win` when the lower bound is at least 2, `win` when it is above 1, `parity` when the interval "
        "contains 1, `loss` when the upper bound is below 1 (the baseline is faster with confidence), `incomplete` when the "
        "record has no qualified baseline for the cell (the baseline failed, was refused or did not qualify; that is not a "
        "slow baseline and not a win). Cells are one operator at one configuration against one baseline. A regime is the "
        "range of `D`, `H`, depth, order, batch and grid size the cells of a row cover, read from the cell label; it is a "
        "bounding range, not a guarantee between its ends. Protocol, intervals and every row: "
        "[performance evidence](performance.md#full-scope-matrix).",
        "",
        "**Reading a baseline.** An independent baseline is another package or a textbook method. A generic engine fed "
        "omnibias's own projection shares that projection. A row against omnibias's own variants or a hand-written closed "
        "form compares two omnibias programs on the same XLA; a loss there names the variant that won and says nothing "
        "about other packages. Endpoints are as recorded: the device time of a captured program, an eager call, "
        "value-and-gradient, a solve to tolerance, or time to a target error with or without compilation.",
        "",
        f"Overall: {overall['big_win']:,} big wins, {overall['win']:,} wins, {overall['parity']:,} at parity, "
        f"{overall['loss']:,} losses and {overall['incomplete']:,} incomplete cells in {len(rows):,}.",
        "",
        "## Summary by family",
        "",
        *summary_table(rows, families),
        "",
        "## Where an independent baseline is faster",
        "",
        "Every (family, operator, back end, device, dtype, endpoint, baseline) with at least one loss cell, most losses first, "
        "with the regime those cells cover. Cells that tie are counted as parity. A baseline that is not here has no loss "
        "cell in this matrix; that is the matrix, not a proof for shapes outside the ranges.",
        "",
        *slower_than_baseline(rows),
    ]
    for fam in families:
        members = [r for r in rows if r["family"] == fam]
        out += ["", f"## {fam}", "", f"{len(members):,} cells."]
        for kind, title in KINDS:
            part = [r for r in members if kind_of(r) == kind]
            if not part:
                continue
            out += ["", f"### {title}", "", KIND_NOTE[kind], "", HEADER]
            for key, cells in group(part, cell_key).items():
                out.append(table_row(key, cells))
    return "\n".join(out) + "\n"


def load() -> dict[str, Any]:
    doc = json.loads(SOURCE.read_text())
    if doc.get("schema") != "omnibias/full-scope/v1":
        raise SystemExit(f"{SOURCE}: unexpected schema {doc.get('schema')!r}")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if docs/regimes.md is not what the JSON renders to")
    args = parser.parse_args(argv)
    text = render(load())
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("stale: docs/regimes.md: run scripts/build_regime_guide.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote docs/{OUT.name} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
