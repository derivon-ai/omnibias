# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Render the three specialized derivative workloads from committed measurements.

Usage: python benchmarks/render_specialized_benchmarks.py docs/benchmarks docs/img/specialized-derivatives.svg
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

SOURCES = ("derivative_order.json", "laplacian_scaling.json", "polylaplacian_order.json")
COLORS = {"omnibias": "#3654B3", "nested": "#30343B", "folx": "#829185", "torch": "#AF8065"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payloads = [json.loads((args.input / name).read_text()) for name in SOURCES]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
        "svg.hashsalt": "omnibias-specialized-v1", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#C9CACB", "text.color": "#1D2025",
        "axes.labelcolor": "#45494E", "xtick.color": "#45494E", "ytick.color": "#45494E",
        "figure.constrained_layout.w_pad": 0.12, "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.wspace": 0.06,
    })
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), layout="constrained")
    rows = payloads[0]["rows"]
    axes[0].plot([r["n"] for r in rows], [r["time_ms"]["closed_form"] for r in rows],
                 color=COLORS["omnibias"], label="omnibias", marker="o", linewidth=2.2, markersize=4)
    axes[0].plot([r["n"] for r in rows], [r["time_ms"]["nested_autograd"] for r in rows],
                 color=COLORS["nested"], label="PyTorch nested AD", marker="s", linewidth=1.6, markersize=4)
    axes[0].set_title("01  Activation derivatives\n20,000 tanh inputs · PyTorch eager",
                      loc="left", fontsize=11, pad=15, linespacing=1.6)
    axes[0].set_xlabel("Derivative order n")
    axes[0].set_xticks([1, 2, 4, 6, 8])
    rows = payloads[1]["rows"]
    for method, label, color, marker in (
        ("omnibias", "omnibias", COLORS["omnibias"], "o"),
        ("jax_hessian", "JAX dense Hessian", COLORS["nested"], "s"),
        ("folx", "folx", COLORS["folx"], "D"),
        ("torch_func_hessian", "PyTorch Hessian (eager)", COLORS["torch"], "^"),
    ):
        axes[1].plot([r["D"] for r in rows], [r["time_ms"][method] for r in rows],
                     label=label, color=color, marker=marker, linewidth=2.2 if method == "omnibias" else 1.6,
                     markersize=4)
    axes[1].set_title("02  Spatial Laplacian\n64 inputs · 32-unit ridge field · JAX JIT*",
                      loc="left", fontsize=11, pad=15, linespacing=1.6)
    axes[1].set_xlabel("Input dimension D")
    axes[1].set_xticks([3, 12, 30, 60])
    rows = payloads[2]["rows"]
    for method, label, color, marker in (
        ("omnibias", "omnibias", COLORS["omnibias"], "o"),
        ("dense_nested", "JAX dense Hessian, nested", COLORS["nested"], "s"),
        ("folx_nested", "folx, nested", COLORS["folx"], "D"),
    ):
        valid = [r for r in rows if r[method]["status"] == "ok"]
        axes[2].plot([r["k"] for r in valid], [r[method]["median_ms"] for r in valid],
                     label=label, color=color, marker=marker, linewidth=2.2 if method == "omnibias" else 1.6,
                     markersize=4)
    axes[2].set_title("03  Iterated Laplacian Δᵏ\n32 inputs · D = 16 · 16-unit ridge field · JAX JIT",
                      loc="left", fontsize=11, pad=15, linespacing=1.6)
    axes[2].set_xlabel("Laplacian order k")
    axes[2].set_xticks([1, 2, 3, 4])
    for axis in axes:
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylabel("Evaluation time (ms)")
        axis.grid(axis="y", which="major", color="#C9CACB", linewidth=0.5, alpha=0.5)
        axis.tick_params(axis="both", which="both", length=3)
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        axis.margins(x=0.06, y=0.4)
    labels = {"dense_nested": "Dense", "folx_nested": "folx"}
    config = payloads[2]["config"]
    status_labels = {"memory_budget": f"{config['rss_limit_mib'] / 1024:g} GiB budget exceeded",
                     "timeout": f"{config['timeout_s']} s timeout", "error": "execution error"}
    budget_results = [f"{labels[method]} k = {r['k']}: {status_labels[r[method]['status']]}"
                      for r in rows for method in ("dense_nested", "folx_nested")
                      if r[method]["status"] != "ok"]
    if budget_results:
        axes[2].text(0.98, 0.05, "\n".join(budget_results),
                     transform=axes[2].transAxes, ha="right", fontsize=8, color="#62666C")
    figure.suptitle("Specialized derivatives · runtime inputs · independent accuracy checks",
                    fontsize=15, x=0.015, ha="left", color="#1D2025")
    figure.supxlabel(
        "Float64 CPU · medians of 9 runs · lower is better · compilation excluded and reported separately\n"
        "*PyTorch Hessian uses eager execution; other spatial methods use JAX JIT. Derivative evaluation only, not training speedups.",
        fontsize=9, color="#62666C",
    )
    description = "; ".join(name + " SHA-256: " + hashlib.sha256((args.input / name).read_bytes()).hexdigest()
                            for name in SOURCES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={"Date": None, "Description": description}, facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
