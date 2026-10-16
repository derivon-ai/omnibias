# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Immutable scalar MLP objectives with interval value and gradient callbacks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import jet_gradient, mlp_jet_mv
from omnibias.core.verified.transcend import require_rigorous_backend, set_strict_backend

LayerLike = tuple[Sequence[Sequence[int | float]], Sequence[int | float] | None, str | None]
StoredLayer = tuple[tuple[tuple[float, ...], ...], tuple[float, ...] | None, str | None]
_ACTIVATIONS = frozenset(("sigmoid", "tanh", "sin", "cos"))


def _coefficient(value: int | float) -> float:
    if type(value) not in (int, float):
        raise TypeError("MLP coefficients must be Python int or float, excluding bool")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("MLP coefficients must be finite binary64 values") from exc
    if not isfinite(result):
        raise ValueError("MLP coefficients must be finite binary64 values")
    if isinstance(value, int) and int(result) != value:
        raise ValueError("integer MLP coefficients must be exactly representable as binary64")
    return result


def _snapshot(layers: Sequence[LayerLike]) -> tuple[StoredLayer, ...]:
    frozen: list[StoredLayer] = []
    previous_width: int | None = None
    for layer in layers:
        if len(layer) != 3:
            raise ValueError("each MLP layer must contain (weight, bias, activation)")
        weight, bias, activation = layer
        if activation is not None and (not isinstance(activation, str) or activation not in _ACTIVATIONS):
            raise ValueError("activation must be None, sigmoid, tanh, sin or cos")
        rows = tuple(tuple(_coefficient(value) for value in row) for row in weight)
        if not rows or not rows[0]:
            raise ValueError("MLP layer dimensions must be positive")
        input_width = len(rows[0])
        if any(len(row) != input_width for row in rows):
            raise ValueError("MLP weights must be rectangular")
        if previous_width is not None and input_width != previous_width:
            raise ValueError("MLP layer input width must match the preceding output width")
        offsets = None if bias is None else tuple(_coefficient(value) for value in bias)
        if offsets is not None and len(offsets) != len(rows):
            raise ValueError("MLP bias width must match its layer output width")
        frozen.append((rows, offsets, activation))
        previous_width = len(rows)
    if not frozen:
        raise ValueError("an interval MLP needs at least one layer")
    if previous_width != 1:
        raise ValueError("an interval MLP objective must have exactly one scalar output")
    return tuple(frozen)


@dataclass(frozen=True, init=False)
class IntervalMLP:
    """Frozen real-valued MLP suitable for interval global optimization.

    Construct with :meth:`from_layers`; each layer is ``(W, b, activation)``.
    Weights and biases are copied to immutable tuples. Python floats are
    treated as their exact stored binary64 values; integers must be exactly
    representable as binary64. Booleans, nonfinite numbers, complex numbers,
    rationals and interval-valued coefficients are refused. This encloses the
    corresponding real mathematical network, not hardware rounding of a
    Torch/JAX forward pass. Later mutations of the source cannot change it.

    Supported activations are sigmoid, tanh, sin and cos; ``None`` is affine.
    Every input box is a sequence of finite :class:`Interval` values whose
    endpoints obey the same binary64 representability policy. The output is
    scalar. Transcendental networks require the rigorous mpmath backend and
    strict evaluation without clearing existing fallback provenance;
    affine-only networks need no optional backend. Bounds retain
    interval dependency overestimation, which can grow with width and depth.
    First-order gradients use ``input_dim + 1`` coefficients. Network size
    and subdivision budgets belong to the calling optimizer; numerical range
    failures propagate rather than manufacturing a finite enclosure.
    """

    layers: tuple[StoredLayer, ...]

    def __init__(self, layers: Sequence[LayerLike]) -> None:
        object.__setattr__(self, "layers", _snapshot(layers))

    @classmethod
    def from_layers(cls, layers: Sequence[LayerLike]) -> IntervalMLP:
        """Snapshot validated layers, fixing all parameters for later calls."""
        return cls(layers)

    @property
    def input_dim(self) -> int:
        """Number of input coordinates of the frozen objective."""
        return len(self.layers[0][0][0])

    def _box(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        frozen = tuple(box)
        if len(frozen) != self.input_dim:
            raise ValueError("box dimension must match the interval MLP input dimension")
        if any(not isinstance(value, Interval) for value in frozen):
            raise TypeError("box entries must be Interval objects")
        try:
            return tuple(Interval(_coefficient(value.lo), _coefficient(value.hi)) for value in frozen)
        except TypeError as exc:
            raise TypeError("box endpoints must be Python int or float, excluding bool") from exc
        except ValueError as exc:
            raise ValueError("interval MLP boxes must have finite endpoints exactly representable as binary64") from exc

    def _jet(self, box: Sequence[Interval], order: int) -> list[list[Interval]]:
        frozen = self._box(box)
        if any(activation is not None for _, _, activation in self.layers):
            require_rigorous_backend()
            previous = set_strict_backend(True)
            try:
                return mlp_jet_mv(frozen, self.layers, order)
            finally:
                set_strict_backend(previous)
        return mlp_jet_mv(frozen, self.layers, order)

    def value(self, box: Sequence[Interval]) -> Interval:
        """Enclose the scalar objective over a finite input box."""
        return self._jet(box, 0)[0][0]

    def gradient(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        """Enclose all first partial derivatives throughout a finite box."""
        jet = self._jet(box, 1)
        return tuple(row[0] for row in jet_gradient(jet, self.input_dim, 1))


__all__ = ["IntervalMLP"]
