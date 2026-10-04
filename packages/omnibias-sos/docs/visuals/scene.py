# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.sos import Polynomial, certify_sos
from visual_story import flow, frame, line, story


def build_scene():
    x = np.linspace(-2, 2, 81)
    v = Polynomial.variable(0, 1)
    frames = []
    for c in np.linspace(0.2, 1.2, 20):
        cert = certify_sos(v * v + Polynomial.constant(float(c), 1))
        frames.append(
            frame(
                line(
                    "Polynomial and nonnegative pieces",
                    x,
                    ("x² + c", x * x + c),
                    ("x²", x * x),
                    ("c", np.full_like(x, c)),
                    ylim=[0, 5.5],
                    xlabel="x",
                ),
                flow(
                    "Certificate boundary",
                    [
                        "Propose an SOS decomposition",
                        "Check the coefficient identity",
                        "Check positive semidefiniteness",
                        "Certified" if cert.certified else "Inconclusive",
                    ],
                    3,
                ),
                f"c = {c:.2f}; certified = {cert.certified}. An unsuccessful search would be inconclusive.",
            )
        )
    return story(
        "sos",
        "Turn positivity into a checkable witness.",
        "A decomposition earns a scoped certificate through explicit checks.",
        "omnibias.sos.Polynomial / certify_sos",
        frames,
    )
