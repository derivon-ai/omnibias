# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

from fractions import Fraction

import numpy as np
from omnibias.qcalculus import q_derivative_poly
from visual_story import frame, line, story


def build_scene():
    x = np.linspace(0, 2, 81)
    frames = []
    for q in np.linspace(0.25, 1, 20):
        coeff = q_derivative_poly([0, 0, 1], Fraction(float(q)).limit_denominator(10000))
        y = np.polynomial.polynomial.polyval(x, [float(v) for v in coeff])
        nodes = q ** np.arange(8)
        frames.append(
            frame(
                {
                    "kind": "graph",
                    "title": "Multiplicative sampling grid",
                    "nodes": np.column_stack([nodes, np.zeros(8)]),
                    "edges": [[i, i + 1] for i in range(7)],
                    "labels": [""] * 8,
                },
                line(
                    "Jackson derivative of x²",
                    x,
                    ("Dq x²", y),
                    ("ordinary derivative", 2 * x),
                    xlabel="x",
                    ylim=[0, 4.2],
                ),
                f"q = {q:.3f}: Dq x² = (1 + q)x; q → 1 is a separate limit from temperature hardening.",
            )
        )
    return story(
        "qcalculus",
        "Calculus on a geometric grid.",
        "Exact q-polynomial coefficients approach ordinary derivatives.",
        "omnibias.qcalculus.q_derivative_poly",
        frames,
    )
