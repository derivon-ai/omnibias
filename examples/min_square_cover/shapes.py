# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bridge from the discrete instance to the continuous ``omnibias-shape`` parameterisation.

The ``K`` candidate squares are represented by continuous centers ``(K, 2)`` and existence
gate logits ``(K,)``; :func:`round_to_squares` snaps the active soft squares back onto the
pixel grid. This is the only place the example knows how the two registers connect.
"""

from __future__ import annotations

import torch
from torch import Tensor

from examples.min_square_cover.data import Instance


def grid_axes(shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Row / column coordinate axes for a ``shape`` image, in the default dtype."""
    rows = torch.arange(shape[0], dtype=torch.get_default_dtype())
    cols = torch.arange(shape[1], dtype=torch.get_default_dtype())
    return rows, cols


def init_centers(
    instance: Instance, greedy_squares: list[tuple[int, int]], k: int, *, seed: int = 0
) -> Tensor:
    """Initialise ``k`` square *centers* from the greedy top-left corners (+ jittered extras).

    A square with top-left ``(r, c)`` has center ``(r + side/2, c + side/2)``; if ``k`` exceeds
    the greedy count the extra centers are seeded at random 1-pixels so every candidate starts
    somewhere useful.
    """
    side = instance.side
    half = side / 2.0
    m, n = instance.shape
    g = torch.Generator().manual_seed(seed)
    centers = torch.zeros(k, 2, dtype=torch.get_default_dtype())
    ones_idx = instance.image.nonzero(as_tuple=False)
    for i in range(k):
        if i < len(greedy_squares):
            r, c = greedy_squares[i]
            centers[i, 0] = r + half
            centers[i, 1] = c + half
        elif len(ones_idx) > 0:
            pick = int(torch.randint(len(ones_idx), (1,), generator=g))
            centers[i, 0] = float(ones_idx[pick, 0])
            centers[i, 1] = float(ones_idx[pick, 1])
        else:
            centers[i, 0] = float(torch.randint(m, (1,), generator=g))
            centers[i, 1] = float(torch.randint(n, (1,), generator=g))
    return centers


def lp_init_centers(
    instance: Instance,
    positions: list[tuple[int, int]],
    weights: list[float],
    k: int,
    *,
    init_gate: float = 2.0,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Warm-start ``k`` centers + gate logits from the LP-relaxation fractional cover.

    The ``k`` highest-weight candidate positions become the centers: the LP is the tightest
    convex relaxation, so its heaviest fractional squares mark the load-bearing locations (a more
    globally informed start than the greedy corners). Gates all start *on* (``init_gate``) so the
    initial soft cover already covers well and the count penalty prunes the redundant candidates
    during annealing. If fewer than ``k`` positions exist, the remainder are seeded at random
    1-pixels.
    """
    side = instance.side
    half = side / 2.0
    m, n = instance.shape
    dtype = torch.get_default_dtype()
    order = sorted(range(len(positions)), key=lambda i: weights[i], reverse=True)
    g = torch.Generator().manual_seed(seed)
    ones_idx = instance.image.nonzero(as_tuple=False)
    centers = torch.zeros(k, 2, dtype=dtype)
    for i in range(k):
        if i < len(order):
            r, c = positions[order[i]]
            centers[i, 0] = r + half
            centers[i, 1] = c + half
        elif len(ones_idx) > 0:
            pick = int(torch.randint(len(ones_idx), (1,), generator=g))
            centers[i, 0] = float(ones_idx[pick, 0])
            centers[i, 1] = float(ones_idx[pick, 1])
        else:
            centers[i, 0] = float(torch.randint(m, (1,), generator=g))
            centers[i, 1] = float(torch.randint(n, (1,), generator=g))
    gate_logits = torch.full((k,), float(init_gate), dtype=dtype)
    return centers, gate_logits


def round_to_squares(
    centers: Tensor, gates: Tensor, instance: Instance, *, threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Snap the active soft squares (gate >= ``threshold``) to valid pixel-grid top-lefts.

    A center ``(cy, cx)`` maps to the top-left ``round(cy - side/2)`` clamped to
    ``[0, M - side]`` so the square stays fully inside the image; duplicate placements are
    removed.
    """
    side = instance.side
    m, n = instance.shape
    half = side / 2.0
    max_r = max(0, m - side)
    max_c = max(0, n - side)
    placements: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    active = gates >= threshold
    for i in range(centers.shape[0]):
        if not bool(active[i]):
            continue
        r = int(round(float(centers[i, 0]) - half))
        c = int(round(float(centers[i, 1]) - half))
        r = min(max(r, 0), max_r)
        c = min(max(c, 0), max_c)
        if (r, c) not in seen:
            seen.add((r, c))
            placements.append((r, c))
    return placements


__all__ = ["grid_axes", "init_centers", "lp_init_centers", "round_to_squares"]
