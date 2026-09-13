# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit, applicable hidden-atom symmetries; no inferred empirical quotient."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class HiddenAction:
    incoming: Array
    bias: Array
    outgoing: Array
    next_bias: Array
    optimizer_policy: str
    identity: str


def hidden_permutation(
    incoming: Array, bias: Array, outgoing: Array, next_bias: Array, permutation: tuple[int, ...]
) -> HiddenAction:
    n = len(bias)
    if sorted(permutation) != list(range(n)) or incoming.shape[0] != n or outgoing.shape[1] != n:
        raise ValueError("permutation and adjacent layer dimensions must agree")
    p = list(permutation)
    return HiddenAction(
        incoming[p].copy(),
        bias[p].copy(),
        outgoing[:, p].copy(),
        next_bias.copy(),
        "transport_permutation",
        "neuron_permutation",
    )


def hidden_sign(
    incoming: Array,
    bias: Array,
    outgoing: Array,
    next_bias: Array,
    signs: Array,
    *,
    activation: str,
    derivative_order: int = 0,
    role: str = "identity",
) -> HiddenAction:
    """Apply tanh parity or sigmoid complement to explicit activation atoms.

    ``derivative_order`` refers to sigma**(n)(w*x+b), without a hidden
    spatial-chain prefactor. Window/band roles require endpoint transformations
    and are deliberately rejected by this atom-level action.
    """
    if role not in ("identity", "derivative") or derivative_order < 0:
        raise ValueError("this action supports identity and explicit derivative atoms")
    if activation not in ("tanh", "sigmoid") or not np.isin(signs, (-1, 1)).all():
        raise ValueError("tanh/sigmoid and signs in {-1,1} required")
    if (
        signs.shape != bias.shape
        or incoming.shape[0] != len(bias)
        or outgoing.shape[1] != len(bias)
    ):
        raise ValueError("adjacent layer dimensions must agree")
    incoming_new = incoming * signs[:, None]
    bias_new = bias * signs
    parity = np.where(signs < 0, (-1) ** (derivative_order + 1), 1)
    outgoing_new = outgoing * parity[None, :]
    shift = np.zeros_like(next_bias)
    complement = activation == "sigmoid" and derivative_order == 0
    if complement:
        shift = outgoing[:, signs < 0].sum(axis=1)
    return HiddenAction(
        incoming_new,
        bias_new,
        outgoing_new,
        next_bias + shift,
        "reset_affected" if complement else "transport_sign",
        "sigmoid_complement" if complement else "activation_parity",
    )


def homogeneous_scaling(
    incoming: Array,
    bias: Array,
    outgoing: Array,
    next_bias: Array,
    scale: Array,
    *,
    degree: int,
    activation: str = "monomial",
) -> HiddenAction:
    """Exact sigma(z)=z**degree scaling; compatibility must be declared."""
    if activation != "monomial" or degree < 1 or scale.shape != bias.shape or np.any(scale == 0):
        raise ValueError("a declared positive-degree monomial and nonzero scales are required")
    return HiddenAction(
        incoming * scale[:, None],
        bias * scale,
        outgoing / (scale**degree)[None, :],
        next_bias.copy(),
        "reset_affected",
        "homogeneous_scaling",
    )


@dataclass(frozen=True)
class HomogeneousAnchorChart:
    degree: int
    anchors: tuple[int, ...]
    margins: tuple[float, ...]

    def normalize(
        self, incoming: Array, bias: Array, outgoing: Array, next_bias: Array
    ) -> HiddenAction:
        augmented = np.column_stack((incoming, bias))
        if len(self.anchors) != len(bias):
            raise ValueError("one anchor per neuron required")
        values = np.array([augmented[i, j] for i, j in enumerate(self.anchors)])
        if any(abs(v) <= m for v, m in zip(values, self.margins, strict=False)):
            raise ValueError("anchor margin crossed; quotient chart is invalid")
        return homogeneous_scaling(
            incoming, bias, outgoing, next_bias, 1 / values, degree=self.degree
        )


def homogeneous_anchor_chart(
    incoming: Array, bias: Array, *, degree: int, margin: float = 1e-8
) -> HomogeneousAnchorChart:
    if degree < 1 or margin <= 0:
        raise ValueError("positive degree and margin required")
    augmented = np.column_stack((incoming, bias))
    anchors = tuple(int(i) for i in np.argmax(abs(augmented), axis=1))
    if any(abs(augmented[i, j]) <= margin for i, j in enumerate(anchors)):
        raise ValueError("inactive neuron has no regular scaling anchor")
    return HomogeneousAnchorChart(degree, anchors, (margin,) * len(anchors))


def duplicate_fibers(incoming: Array, bias: Array, outgoing: Array) -> tuple[tuple[int, ...], ...]:
    """Exact stored-parameter duplicates and zero-output fibers, reported separately."""
    augmented = np.column_stack((incoming, bias))
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for i in range(len(bias)):
        if i in seen:
            continue
        same = tuple(j for j in range(i, len(bias)) if np.array_equal(augmented[i], augmented[j]))
        seen.update(same)
        if len(same) > 1:
            groups.append(same)
        elif np.all(outgoing[:, i] == 0):
            groups.append((i,))
    return tuple(groups)


__all__ = [
    "HiddenAction",
    "HomogeneousAnchorChart",
    "duplicate_fibers",
    "hidden_permutation",
    "hidden_sign",
    "homogeneous_anchor_chart",
    "homogeneous_scaling",
]
