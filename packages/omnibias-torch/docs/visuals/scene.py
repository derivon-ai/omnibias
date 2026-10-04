# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import torch
from omnibias.torch.jet import jet_to_tower, mlp_jet
from visual_story import frame, line, story


def build_scene():
    torch.set_num_threads(1)
    x = torch.linspace(-1, 1, 41, dtype=torch.float64)
    a = torch.tensor([[0.8]], dtype=torch.float64, requires_grad=True)
    frames = []
    losses = []
    for step in range(20):
        layers = [
            (
                torch.tensor([[0.7]], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
                "tanh",
            ),
            (a, None, None),
        ]
        t = jet_to_tower(mlp_jet(x[:, None], torch.ones_like(x[:, None]), layers, order=4))
        r = t[4] + t[0]
        loss = r.square().mean()
        losses.append(float(loss.detach()))
        g = torch.autograd.grad(loss, a)[0]
        frames.append(
            frame(
                line(
                    "Fourth-order residual",
                    x.numpy(),
                    ("u⁽⁴⁾ + u", r.detach().numpy().ravel()),
                    ylim=[-2, 2],
                    xlabel="collocation coordinate",
                ),
                line(
                    "Parameter optimization",
                    range(len(losses)),
                    ("mean squared residual", losses),
                    xlim=[0, 19],
                    ylim=[0, 2],
                    xlabel="gradient step",
                ),
                f"Step {step}: parameter gradient {float(g):.3g}; this is a small residual example, not a PDE convergence guarantee.",
            )
        )
        with torch.no_grad():
            a -= 0.08 * g
    return story(
        "torch",
        "Derivatives that keep training.",
        "Spatial jets forward. Parameter gradients backward.",
        "omnibias.torch.jet.mlp_jet + torch.autograd.grad",
        frames,
    )
