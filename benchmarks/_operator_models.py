# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic operands and independent references for operator benchmarks."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Any


def model_data(family: str, order: int, batch: int, dim: int, hidden: int,
               latent: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    dim = order if family == "separable" else dim
    points = [[rng.uniform(-.25, .25) for _ in range(dim)] for _ in range(batch)]
    if family == "separable":
        return {"family": family, "points": points, "dimension": dim,
                "weights": [rng.uniform(.8, 1.2) for _ in range(dim)],
                "biases": [rng.uniform(-.1, .1) for _ in range(dim)]}
    widths = {"ridge": (dim, hidden, 1), "coupled": (dim, hidden, hidden, 1),
              "bottleneck": (dim, latent, hidden, 1)}[family]
    layers = []
    for i, (din, dout) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        weight = [[rng.gauss(0., .5 / math.sqrt(din)) for _ in range(din)] for _ in range(dout)]
        bias = [rng.uniform(-.15, .15) for _ in range(dout)]
        layers.append((weight, bias, i < len(widths) - 2))
    return {"family": family, "points": points, "dimension": dim, "layers": layers}


def torch_model(data: dict[str, Any]) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if data["family"] == "separable":
                values = [data["weights"], data["biases"]]
            else:
                values = [v for w, b, _ in data["layers"] for v in (w, b)]
            self.operands = nn.ParameterList([nn.Parameter(torch.tensor(v, dtype=torch.float64)) for v in values])

        def forward(self, x: Any) -> Any:
            if data["family"] == "separable":
                return torch.sin(torch.tanh(x * self.operands[0] + self.operands[1]).sum(-1, keepdim=True))
            for i, (_, _, nonlinear) in enumerate(data["layers"]):
                x = F.linear(x, self.operands[2*i], self.operands[2*i+1])
                if nonlinear:
                    x = torch.tanh(x)
            return x

    return Model()


def high_precision(data: dict[str, Any], order: int, indices: list[int]) -> list[str]:
    """80-digit formulas/AD independent of every tensor derivative kernel."""
    import mpmath as mp

    with mp.workdps(80):
        points = [[mp.mpf(v) for v in data["points"][i]] for i in indices]
        if data["family"] == "separable":
            weights, biases = map(lambda row: list(map(mp.mpf, row)), (data["weights"], data["biases"]))
            return [mp.nstr(mp.diff(mp.sin, sum(mp.tanh(w*x+b) for w, x, b in zip(weights, point, biases, strict=True)), order)
                           * mp.fprod(w / mp.cosh(w*x+b)**2 for w, x, b in zip(weights, point, biases, strict=True)), 80)
                    for point in points]
        layers = [([[mp.mpf(v) for v in row] for row in w], list(map(mp.mpf, b)), act)
                  for w, b, act in data["layers"]]

        def evaluate(point: Any, selected_layers: Any = layers) -> Any:
            for weight, bias, act in selected_layers:
                point = [sum(a*b for a, b in zip(row, point, strict=True)) + offset
                         for row, offset in zip(weight, bias, strict=True)]
                if act:
                    point = list(map(mp.tanh, point))
            return point[0]

        if data["family"] == "coupled":
            return [mp.nstr(mp.diff(lambda t, p=point: evaluate([x+t for x in p]), mp.mpf(0), order), 80) for point in points]
        weight, bias, _ = layers[0]
        if data["family"] == "ridge":
            readout = layers[1][0][0]
            return [mp.nstr(sum(c * mp.diff(mp.tanh, sum(w*x for w, x in zip(row, point, strict=True))+b, order)
                               * sum(w*w for w in row)**(order//2)
                               for row, b, c in zip(weight, bias, readout, strict=True)), 80) for point in points]
        dim = len(weight)
        gram = [[sum(a*b for a, b in zip(left, right, strict=True)) for right in weight] for left in weight]
        symbol: dict[tuple[int, ...], Any] = {(0,)*dim: mp.mpf(1)}
        for _ in range(order//2):
            expanded: dict[tuple[int, ...], Any] = defaultdict(lambda: mp.mpf(0))
            for alpha, coefficient in symbol.items():
                for i, j in itertools.product(range(dim), repeat=2):
                    beta = list(alpha)
                    beta[i] += 1
                    beta[j] += 1
                    expanded[tuple(beta)] += coefficient * gram[i][j]
            symbol = dict(expanded)

        def outer(*z: Any) -> Any:
            return evaluate(list(map(mp.tanh, z)), layers[1:])

        values = []
        for point in points:
            z = tuple(sum(w*x for w, x in zip(row, point, strict=True))+b for row, b in zip(weight, bias, strict=True))
            values.append(mp.nstr(sum(coefficient * mp.diff(outer, z, alpha) for alpha, coefficient in symbol.items()), 80))
        return values
