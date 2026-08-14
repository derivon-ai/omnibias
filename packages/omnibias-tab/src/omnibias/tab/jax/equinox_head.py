# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Optional Equinox wrappers around the JAX tab kernels.

Import this module only when Equinox is installed (the ``[equinox]`` extra).
:mod:`omnibias.tab.jax` itself stays Equinox-free. ``__call__`` delegates to
:func:`~omnibias.tab.jax.arrangement.arrangement_forward` /
:func:`~omnibias.tab.jax.arrangement.boosted_forward` /
:func:`~omnibias.tab.jax.model.forward_arrays`.
"""

from __future__ import annotations

from typing import Any

try:
    import equinox as eqx
except ImportError as exc:  # pragma: no cover - exercised via importorskip
    raise ImportError(
        "omnibias.tab.jax.equinox_head requires the optional [equinox] extra "
        "(pip install equinox)"
    ) from exc

import jax
from omnibias.tab.jax.arrangement import arrangement_forward, boosted_forward
from omnibias.tab.jax.model import forward_arrays


class ArrangementHead(eqx.Module):
    """Equinox module wrapping arrangement arrays ``W, t, cell_logits, beta``."""

    W: jax.Array
    t: jax.Array
    cell_logits: jax.Array
    beta: jax.Array

    def __call__(self, x: Any) -> Any:
        return arrangement_forward(self.W, self.t, self.cell_logits, x, self.beta)


class BoostedHead(eqx.Module):
    """Equinox module wrapping stacked arrangement members + ``lr`` / ``base``."""

    W_stack: jax.Array
    t_stack: jax.Array
    logits_stack: jax.Array
    beta: jax.Array
    learning_rate: jax.Array
    base: jax.Array

    def __call__(self, x: Any) -> Any:
        return boosted_forward(
            self.W_stack,
            self.t_stack,
            self.logits_stack,
            x,
            self.beta,
            self.learning_rate,
            self.base,
        )


class SoftTreeHead(eqx.Module):
    """Equinox module wrapping SoftTree arrays ``W, t, leaves, b0, beta``."""

    W: jax.Array
    t: jax.Array
    leaves: jax.Array
    b0: jax.Array
    beta: jax.Array
    depth: int = eqx.field(static=True)

    def __call__(self, x: Any) -> Any:
        return forward_arrays(
            self.W, self.t, self.leaves, self.b0, x, self.beta, self.depth
        )


__all__ = [
    "ArrangementHead",
    "BoostedHead",
    "SoftTreeHead",
]
