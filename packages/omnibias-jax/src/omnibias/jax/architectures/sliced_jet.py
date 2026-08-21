# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sliced-jet encoder (jax; theory 09-28).

Tokens are scans, not patches. founding bias collapse
(``delta -> 0``) is the scan jet. Temperature collapse
(``beta -> inf``, feasibility) is labelled. do not conflate the two.
Not a ViT. Not ImageNet.
"""

from __future__ import annotations

from omnibias.core import sliced_jet as core

from jax import Array
from jax import numpy as jnp

DISCLAIMER = core.DISCLAIMER
SlicedJetConfig = core.SlicedJetConfig
honesty_payload = core.honesty_payload


def _as_image(image: Array) -> core.Image:
    rows = image.tolist()
    return tuple(tuple(float(v) for v in row) for row in rows)


class SlicedJetEncoder:
    """Scan-jet tokens. Not a patch ViT."""

    def __init__(self, config: SlicedJetConfig | None = None) -> None:
        self.config = core.DEFAULT_CONFIG if config is None else config
        self._core = core.SlicedJetEncoder(self.config)

    def encode(self, image: Array) -> Array:
        tokens = self._core.encode(_as_image(image))
        return jnp.asarray(tokens)

    def decode(self, tokens: Array, *, height: int, width: int) -> Array:
        bank = tuple(
            tuple(tuple(float(c) for c in cell) for cell in token) for token in tokens.tolist()
        )
        return jnp.asarray(self._core.decode(bank, height=height, width=width))

    def reconstruct(self, image: Array) -> Array:
        return jnp.asarray(self._core.reconstruct(_as_image(image)))


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "SlicedJetConfig",
    "SlicedJetEncoder",
    "honesty_payload",
    "worked_example",
]
