# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Render explanatory README animations from analytic curves, not timing estimates."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img/explain"


def render(kind: str) -> None:
    x = np.linspace(-3, 3, 321)
    frames = []
    for progress in np.linspace(0, 1, 32):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor("#f7f9fc")
        for ax in axes:
            ax.set_facecolor("#f7f9fc")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#cad3df")
            ax.tick_params(colors="#536174", labelsize=8)
            ax.set_xlabel("coordinate x", color="#536174", fontsize=9)
            ax.grid(alpha=.12)
        if kind == "bias-collapse":
            delta = 1.0 * (.04 ** progress)
            exact = 1 - np.tanh(x)**2
            shifted = (np.tanh(x + delta) - np.tanh(x)) / delta
            axes[0].plot(x, np.tanh(x), color="#12314b", lw=2, label="tanh(x)")
            axes[0].plot(x, np.tanh(x+delta), color="#087f72", lw=2, label="tanh(x + δ)")
            axes[1].plot(x, exact, color="#12314b", lw=2, label="analytic derivative")
            axes[1].plot(x, shifted, color="#087f72", lw=2, ls="--", label="normalized difference")
            axes[0].set_title("Bring the biases together", loc="left", fontsize=11)
            axes[1].set_title(f"δ = {delta:.3f}  →  derivative", loc="left", fontsize=11)
            axes[0].set_ylim(-1.2, 1.2)
            axes[1].set_ylim(-.06, 1.15)
            title = "BIAS COLLAPSE     From nearby activations to derivatives"
            caption = "The kernel evaluates the analytic limit directly. Finite differences here illustrate the mechanism."
        else:
            beta = .6 * (40 ** progress)
            gate = 1 / (1 + np.exp(-beta*x))
            left, right = -.45*x+.1, .22*x+.7
            axes[0].plot(x, gate, color="#087f72", lw=2.5, label="smooth gate")
            axes[0].plot(x, (x>0).astype(float), color="#8796a8", ls=":", label="hard decision")
            axes[1].plot(x, left, color="#8796a8", lw=1, ls="--", label="expert A")
            axes[1].plot(x, right, color="#536174", lw=1, ls="--", label="expert B")
            axes[1].plot(x, (1-gate)*left+gate*right, color="#087f72", lw=2.5, label="trainable mixture")
            axes[0].set_title(f"β = {beta:.2f}  →  sharper routing", loc="left", fontsize=11)
            axes[1].set_title("Blend experts through a smooth decision", loc="left", fontsize=11)
            axes[0].set_ylim(-.06, 1.1)
            axes[1].set_ylim(-1.4, 1.8)
            title = "TEMPERATURE COLLAPSE     Make decisions differentiable"
            caption = "Finite β keeps a smooth gate; the hard limit needs a tie policy. This is an illustration, not a training benchmark."
        for ax in axes:
            ax.legend(loc="lower right", frameon=False, fontsize=8)
        fig.suptitle(title, x=.07, ha="left", color="#12314b", fontsize=13, fontweight="bold")
        fig.text(.07, .03, caption, fontsize=8, color="#536174")
        fig.subplots_adjust(left=.07, right=.97, bottom=.19, top=.79, wspace=.23)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
        frames.append(frame)
        if progress == 1:
            frame.save(OUT / f"{kind}.png")
            fig.savefig(OUT / f"{kind}.svg", metadata={"Date": None})
        plt.close(fig)
    palette = frames[0].quantize(colors=128)
    quantized = [frame.quantize(palette=palette) for frame in frames]
    quantized[0].save(OUT / f"{kind}.gif", save_all=True, append_images=quantized[1:],
                      duration=[100]*31+[1400], loop=0, optimize=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "omnibias-explain-v1"})
    for name in ("bias-collapse", "temperature-collapse"):
        render(name)
