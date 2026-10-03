# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Keep agent benchmark summaries tied to measured workloads, without imports."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (".cursor/rules/omnibias.mdc",)
ARTIFACTS = ("derivative_order", "laplacian_scaling", "polylaplacian_order")
BEGIN = "<!-- BEGIN GENERATED CAPABILITY EVIDENCE -->"
END = "<!-- END GENERATED CAPABILITY EVIDENCE -->"


def read_artifacts(root: Path) -> dict[str, Any]:
    return {
        name: json.loads((root / "docs/benchmarks" / f"{name}.json").read_text())
        for name in ARTIFACTS
    }


def row_at(payload: dict[str, Any], key: str, value: int) -> dict[str, Any]:
    matches = [row for row in payload["rows"] if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one measured {key}={value} workload")
    result: dict[str, Any] = matches[0]
    return result


def median(result: dict[str, Any], repeats: int) -> float | None:
    status = result.get("status", "ok")
    if status in {"memory_budget", "timeout", "error"}:
        return None
    if status != "ok":
        raise ValueError(f"Unknown measurement status: {status}")
    samples = result["samples_ms"]
    value = float(result["median_ms"])
    if (
        len(samples) != repeats
        or not samples
        or not all(math.isfinite(sample) and sample > 0 for sample in samples)
        or not math.isfinite(value)
        or value <= 0
        or value != statistics.median(samples)
    ):
        raise ValueError("Measurement must have positive finite samples and their exact median")
    return value


def comparison(fast: dict[str, Any], slow: dict[str, Any], repeats: int) -> str:
    fast_ms, slow_ms = median(fast, repeats), median(slow, repeats)
    if fast_ms is None or slow_ms is None:
        failed = fast if fast_ms is None else slow
        return f"unavailable ({failed['status']}; no speedup)"
    ratio = slow_ms / fast_ms
    formatted = f"{ratio:,.0f}" if ratio >= 100 else f"{ratio:.1f}"
    return f"**{formatted}×**"


def evidence(data: dict[str, Any], prefix: str) -> str:
    """Render ratios from successful medians; failed baselines never yield claims."""
    activation, laplacian, repeated = (data[name] for name in ARTIFACTS)
    a, lap, poly = (item["config"] for item in (activation, laplacian, repeated))
    if any(item["dtype"] != a["dtype"] or item["repeats"] != a["repeats"]
           for item in (lap, poly)):
        raise ValueError("Summary requires a common dtype and repeat count")
    for payload in (activation, laplacian, repeated):
        if "CPU" not in payload["config"]["execution"]:
            raise ValueError("CPU benchmark summary cannot describe another execution device")
    for config in (lap, poly):
        if config["runtime_arguments"] != ["X", "W", "beta", "c"]:
            raise ValueError("Operator evidence requires runtime inputs and weights")
    counts = int(a["repeats"])
    first = row_at(activation, "n", 8)["measurements"]
    second = row_at(laplacian, "D", 60)["methods"]
    third, fourth = (row_at(repeated, "k", k) for k in (3, 4))

    def link(name: str) -> str:
        return f"{prefix}docs/benchmarks/{name}.json"

    activation_ratio = comparison(first["closed_form"], first["nested_autograd"], counts)
    laplacian_ratio = comparison(second["omnibias"], second["jax_hessian"], counts)
    third_ratio = comparison(third["omnibias"], third["dense_nested"], counts)
    fourth_ratio = comparison(fourth["omnibias"], fourth["folx_nested"], counts)
    dense = fourth["dense_nested"]
    dense_ms = median(dense, counts)
    dense_note = (
        f"Dense `Δ⁴`: `{dense['status']}` under {poly['rss_limit_mib']:,} MiB / "
        f"{poly['timeout_s']} s process budgets; no speedup for that run."
        if dense_ms is None else
        f"Dense `Δ⁴` completed: {dense_ms:.4f} ms; see the artifact for all baselines."
    )
    return "\n".join([
        f"Measured {a['dtype']} CPU medians, {counts} repeats; derivative evaluation, not training:",
        "",
        f"- [`σ⁽⁸⁾`]({link('derivative_order')}), {a['n_points']:,} {a['activation']} inputs, "
        f"Torch eager/{a['torch_threads']} thread: {activation_ratio} vs nested autograd.",
        f"- [`Δ`]({link('laplacian_scaling')}), D=60/B={lap['B']}/H={lap['H']}: "
        f"{laplacian_ratio} vs JAX dense Hessian.",
        f"- [`Δ³` / `Δ⁴`]({link('polylaplacian_order')}), "
        f"D={poly['D']}/B={poly['B']}/H={poly['H']}: {third_ratio} / {fourth_ratio} "
        "vs dense JAX / nested folx respectively.",
        "",
        "Operator timings use JAX JIT, runtime inputs/weights, compilation excluded. " + dense_note,
        f"[Full protocol, accuracy and baseline wins]({prefix}docs/performance.md).",
    ])


def replace_block(text: str, content: str) -> str:
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("Expected one capability-evidence marker pair")
    start, finish = text.index(BEGIN), text.index(END)
    if finish < start:
        raise ValueError("Capability-evidence markers are reversed")
    return text[:start] + BEGIN + "\n\n" + content + "\n\n" + text[finish:]


def render(root: Path) -> dict[Path, str]:
    data = read_artifacts(root)
    result = {}
    for relative in TARGETS:
        path = root / relative
        prefix = os.path.relpath(root, path.parent).replace(os.sep, "/") + "/"
        result[path] = replace_block(path.read_text(), evidence(data, prefix))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on stale agent evidence")
    args = parser.parse_args(argv)
    try:
        outputs = render(ROOT)
    except (KeyError, ValueError) as error:
        parser.exit(1, f"Invalid capability evidence: {error}\n")
    changed = [path for path, content in outputs.items() if path.read_text() != content]
    if args.check:
        for path in changed:
            print(f"Stale capability evidence: {path.relative_to(ROOT)}")
        return int(bool(changed))
    for path in changed:
        path.write_text(outputs[path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
