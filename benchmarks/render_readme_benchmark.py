# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Render the README chart from its JSON evidence, without rerunning timings.

Run with matplotlib installed, passing the JSON input and SVG output paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    rows = payload["rows"]
    orders = [row["order"] for row in rows]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
        "svg.hashsalt": "omnibias-readme-v1", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#c8d3df", "text.color": "#122238",
        "axes.labelcolor": "#33465c", "xtick.color": "#33465c", "ytick.color": "#33465c",
    })
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    panels = [
        ("PyTorch · eager / one CPU thread", [
            ("torch_omnibias", "omnibias Riccati jet", "#087f72", "o"),
            ("torch_nested_ad", "Nested autograd", "#4253af", "s"),
        ]),
        ("JAX · compiled / CPU runtime defaults", [
            ("jax_omnibias", "omnibias Riccati jet", "#087f72", "o"),
            ("jax_nested_ad", "Nested grad", "#4253af", "s"),
            ("jax_taylor_ad", "jax.experimental.jet", "#bc5940", "D"),
        ]),
    ]
    for axis, (title, methods) in zip(axes, panels, strict=True):
        for key, label, color, marker in methods:
            values = [row["methods"][key]["median_ms"] for row in rows]
            axis.plot(orders, values, label=label, color=color, marker=marker,
                      linewidth=2.4, markersize=7)
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=16)
        axis.set_xlabel("Derivative order")
        axis.set_ylabel("Median evaluation time (ms) · lower is better")
        axis.set_xticks(orders)
        axis.set_yscale("log")
        axis.grid(axis="y", which="major", alpha=0.16)
        axis.legend(loc="upper left", fontsize=9, frameon=False)
    figure.suptitle("High-order derivatives, measured on the same network", fontsize=17,
                    fontweight="bold", x=0.01, ha="left")
    figure.supxlabel(
        "Tanh MLP 1→8→8→1 · 128 inputs · float64 · 9 repeats · JAX compile time excluded\n"
        "An 80-digit independent reference checks every method. These are local CPU timings, not training speedups.",
        fontsize=10, color="#52647a",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={
        "Date": None,
        "Description": "Measurement SHA-256: " + hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }, facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
