# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Render package-owned numerical storyboards into README animations and posters.

A scene module supplies build_scene(), returning title, subtitle, source and
frames. Each frame contains two plotting panels and a caption. Mathematical
values are computed by the owning package; this renderer only draws them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import textwrap
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLORS = ["#04776f", "#bb562f", "#536cb0", "#83939b"]
BG, INK, MUTED = "#f7f7f2", "#172e36", "#5c6b70"


def panel(ax: Any, data: dict[str, Any], mobile: bool) -> None:
    size = 14 if mobile else 11
    ax.set_facecolor(BG)
    ax.set_title(data["title"], loc="left", fontsize=size + 3, color=INK, pad=15)
    kind = data.get("kind", "lines")
    if kind == "lines":
        x = np.asarray(data["x"])
        for index, series in enumerate(data["series"]):
            y = np.asarray(series["y"])
            if not np.isfinite(y).all():
                raise ValueError("Scene contains non-finite data")
            ax.plot(
                x,
                y,
                lw=2.5,
                color=COLORS[index % len(COLORS)],
                label=series["label"],
                ls="--" if series.get("dashed") else "-",
            )
        if "lower" in data:
            ax.fill_between(x, data["lower"], data["upper"], color=COLORS[0], alpha=0.16)
        if "points" in data:
            ax.scatter(*np.asarray(data["points"]).T, s=35, color=COLORS[1], zorder=5)
        ax.legend(frameon=False, fontsize=size - 1, loc=data.get("legend", "best"))
    elif kind == "bars":
        values = data["values"]
        ax.bar(
            np.arange(len(values)),
            values,
            color=[COLORS[i % len(COLORS)] for i in range(len(values))],
            width=0.65,
        )
        ticks = np.arange(0, len(values), max(1, (len(values) + 7) // 8))
        labels = data.get("labels", list(map(str, range(len(values)))))
        ax.set_xticks(
            ticks,
            [labels[i] for i in ticks],
            fontsize=size,
        )
    elif kind == "heatmap":
        a = np.asarray(data["values"])
        plotted = ax.imshow(
            a, cmap="viridis", aspect="auto", vmin=data.get("vmin"), vmax=data.get("vmax")
        )
        ax.set_xticks(range(0, a.shape[1], max(1, (a.shape[1] + 7) // 8)))
        ax.set_yticks(range(0, a.shape[0], max(1, (a.shape[0] + 7) // 8)))
        if a.size <= 36:
            for (i, j), value in np.ndenumerate(a):
                ax.text(
                    j,
                    i,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=size,
                    color=INK if plotted.norm(value) > 0.55 else "white",
                )
    elif kind == "graph":
        nodes = np.asarray(data["nodes"])
        for index, edge in enumerate(data.get("edges", [])):
            i, j = edge[:2]
            weight = data.get("edge_weights", [])[index] if "edge_weights" in data else None
            if weight is not None and (not np.isfinite(weight) or weight < 0):
                raise ValueError("Graph illustration weights must be finite and nonnegative")
            ax.plot(
                nodes[[i, j], 0],
                nodes[[i, j], 1],
                color=COLORS[0] if len(edge) > 2 and edge[2] else "#b7c2c3",
                lw=1 + 1.5 * weight if weight is not None else (
                    3 if len(edge) > 2 and edge[2] else 1.3
                ),
            )
            if "edge_labels" in data:
                midpoint = nodes[[i, j]].mean(axis=0)
                ax.annotate(
                    data["edge_labels"][index], midpoint, xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=size,
                    color=INK, bbox={"facecolor": BG, "edgecolor": "none", "pad": 1},
                )
        ax.scatter(nodes[:, 0], nodes[:, 1], s=180, c=data.get("colors", COLORS[0]), zorder=5)
        for i, point in enumerate(nodes):
            ax.annotate(
                str(data.get("labels", list(range(len(nodes))))[i]),
                point,
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=size,
                color=INK,
            )
        ax.margins(0.25)
        ax.set_aspect("equal")
        ax.axis("off")
    elif kind == "flow":
        ax.axis("off")
        steps = data["steps"]
        for i, step in enumerate(steps):
            y = 0.9 - i * (0.8 / max(1, len(steps) - 1))
            active = i <= data.get("active", len(steps) - 1)
            ax.text(
                0.04,
                y,
                f"{i + 1:02d}",
                transform=ax.transAxes,
                color=COLORS[0] if active else "#a9b2b2",
                fontsize=size + 4,
                weight="bold",
            )
            ax.text(
                0.18,
                y,
                "\n".join(textwrap.wrap(step, 31 if mobile else 36)),
                transform=ax.transAxes,
                color=INK if active else "#a9b2b2",
                fontsize=size + 1,
                va="center",
            )
    elif kind == "landscape":
        ax.contour(data["x"], data["y"], data["values"], levels=12, cmap="viridis", linewidths=1)
        path = np.asarray(data["path"])
        ax.plot(path[:, 0], path[:, 1], "o-", color=COLORS[1], lw=2, markersize=4)
        ax.set_aspect("equal")
    elif kind == "vectors":
        x, y = np.asarray(data["x"]), np.asarray(data["y"])
        ax.quiver(
            x,
            y,
            data["u"],
            data["v"],
            color=COLORS[0],
            angles="xy",
            scale_units="xy",
            scale=data.get("scale", 4),
        )
        ax.set_aspect("equal")
    else:
        raise ValueError(f"Unknown scene panel: {kind}")
    if kind not in {"flow", "graph"}:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#bcc7c7")
        ax.tick_params(colors=MUTED, labelsize=size - 1)
        ax.set_xlabel(data.get("xlabel", ""), color=MUTED, fontsize=size)
        ax.set_ylabel(data.get("ylabel", ""), color=MUTED, fontsize=size)
        if "ylim" in data:
            ax.set_ylim(data["ylim"])
        if "xlim" in data:
            ax.set_xlim(data["xlim"])
        if kind in {"lines", "bars"}:
            ax.grid(axis="y", alpha=0.12)


def render(scene_path: Path, *, check: bool = False) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("package_scene", scene_path)
    if spec is None or spec.loader is None:
        raise ValueError(scene_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    story = module.build_scene()
    out = scene_path.parent
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": story["title"]})
    hashes = {}
    for mobile in (False, True):
        frames = []
        for index, frame in enumerate(story["frames"]):
            fig, axes = plt.subplots(
                2 if mobile else 1,
                1 if mobile else 2,
                figsize=(6, 9.4) if mobile else (11.8, 6.2),
                dpi=100,
            )
            fig.patch.set_facecolor(BG)
            fig.text(
                0.07,
                0.955,
                story["package"].upper(),
                fontsize=11 if mobile else 12,
                color=COLORS[0],
                weight="bold",
            )
            title = "\n".join(textwrap.wrap(story["title"], 26)) if mobile else story["title"]
            heading = fig.text(
                0.07,
                0.908,
                title,
                fontsize=25 if mobile else 29,
                color=INK,
                weight="bold",
                va="top",
            )
            if not mobile:
                # Fit real rendered glyph widths, not a character-count guess.
                fig.canvas.draw()
                while heading.get_window_extent().width > fig.bbox.width * 0.88:
                    heading.set_fontsize(heading.get_fontsize() - 0.5)
            if not mobile:
                fig.text(0.07, 0.80, story["subtitle"], fontsize=13, color=MUTED)
            for ax, data in zip(np.ravel(axes), frame["panels"], strict=True):
                panel(ax, data, mobile)
            fig.text(
                0.07,
                0.045,
                "\n".join(textwrap.wrap(frame["caption"], 52 if mobile else 112)),
                fontsize=11,
                color=MUTED,
            )
            fig.subplots_adjust(
                left=0.12 if mobile else 0.08,
                right=0.94,
                top=0.74 if mobile else 0.68,
                bottom=0.16 if mobile else 0.20,
                hspace=0.58,
                wspace=0.34,
            )
            fig.canvas.draw()
            frames.append(
                Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            )
            if index == len(story["frames"]) - 1:
                name = "poster-mobile" if mobile else "poster"
                for extension in ("svg", "png"):
                    buffer = io.BytesIO()
                    if extension == "svg":
                        fig.savefig(buffer, format="svg", metadata={"Date": None})
                    else:
                        frames[-1].save(buffer, format="PNG")
                    path = out / f"{name}.{extension}"
                    payload = buffer.getvalue()
                    if check:
                        if not path.exists() or path.read_bytes() != payload:
                            raise ValueError(f"Stale visual: {path}")
                    else:
                        path.write_bytes(payload)
            plt.close(fig)
        palette = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
        for i, frame in enumerate(frames):
            palette.paste(frame, (0, i * frame.height))
        palette = palette.quantize(colors=96)
        quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

        buffer = io.BytesIO()
        quantized[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=quantized[1:],
            duration=[150] * (len(frames) - 1) + [1200],
            loop=0,
            optimize=False,
        )
        name = "story-mobile.gif" if mobile else "story.gif"
        payload = buffer.getvalue()
        if len(payload) > 1.5 * 1024 * 1024:
            raise ValueError(f"{scene_path}: animation exceeds size budget")
        if check:
            if not (out / name).exists() or (out / name).read_bytes() != payload:
                raise ValueError(f"Stale visual: {out / name}")
        else:
            (out / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    if not check:
        (out / "provenance.json").write_text(
            json.dumps(
                {
                    "source": story["source"],
                    "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                    "render_environment": {
                        name: version(name) for name in ("numpy", "matplotlib", "Pillow")
                    },
                    "purpose": "computed illustration; not a performance measurement",
                    "sha256": hashes,
                },
                indent=2,
            )
            + "\n"
        )
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for scene in args.scenes:
        print(scene, render(scene, check=args.check), flush=True)


if __name__ == "__main__":
    main()
