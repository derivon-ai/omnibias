# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

from omnibias.boolean import anf_from_truth_table, truth_table_from_anf, truth_table_from_callable
from visual_story import bars, flow, frame, story


def build_scene():
    values = truth_table_from_callable(lambda a, b: a ^ b, 2)
    poly = anf_from_truth_table(values)
    assert truth_table_from_anf(poly) == values
    frames = []
    for i in range(20):
        frames.append(
            frame(
                bars("XOR truth table", values, ["00", "01", "10", "11"], ylim=[0, 1.2]),
                flow(
                    "Exact representation",
                    [
                        "Truth values over {0,1}²",
                        "Transform over the binary field",
                        "ANF: a ⊕ b",
                        "Round trip reproduces all rows",
                    ],
                    min(i // 5, 3),
                ),
                "Exact Boolean algebra; optional soft gates are a separate differentiable realization.",
            )
        )
    return story(
        "boolean",
        "From truth values to exact algebra.",
        "Representations can change while every Boolean result stays fixed.",
        "truth_table_from_callable / anf_from_truth_table / truth_table_from_anf",
        frames,
    )
