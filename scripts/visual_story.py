# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Small drawing descriptions shared by package-owned README scenes."""

from __future__ import annotations

from typing import Any

import numpy as np


def line(title: str, x: Any, *series: tuple[str, Any], **options: Any) -> dict[str, Any]:
    return {
        "kind": "lines",
        "title": title,
        "x": np.asarray(x),
        "series": [{"label": label, "y": np.asarray(values)} for label, values in series],
        **options,
    }


def bars(title: str, values: Any, labels: Any = None, **options: Any) -> dict[str, Any]:
    result = {"kind": "bars", "title": title, "values": np.asarray(values), **options}
    if labels is not None:
        result["labels"] = labels
    return result


def flow(title: str, steps: list[str], active: int) -> dict[str, Any]:
    return {"kind": "flow", "title": title, "steps": steps, "active": active}


def frame(left: dict, right: dict, caption: str) -> dict[str, Any]:
    return {"panels": [left, right], "caption": caption}


def story(
    package: str, title: str, subtitle: str, source: str, frames: list[dict]
) -> dict[str, Any]:
    return {
        "package": "omnibias-" + package,
        "title": title,
        "subtitle": subtitle,
        "source": source,
        "frames": frames,
    }
