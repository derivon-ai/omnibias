# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sliced-jet encoder (torch; theory 09-28).

Tokens are scans, not patches. founding bias collapse
(``delta -> 0``) is the scan jet. Temperature collapse
(``beta -> inf``, feasibility) is labelled. do not conflate the two.
Not a ViT. Not ImageNet.
"""

from __future__ import annotations

from omnibias.core import sliced_jet as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
SlicedJetConfig = core.SlicedJetConfig
honesty_payload = core.honesty_payload


def _as_image(image: Tensor) -> core.Image:
    rows = image.detach().tolist()
    return tuple(tuple(float(v) for v in row) for row in rows)


class SlicedJetEncoder:
    """Scan-jet tokens. Not a patch ViT."""

    def __init__(self, config: SlicedJetConfig | None = None) -> None:
        self.config = core.DEFAULT_CONFIG if config is None else config
        self._core = core.SlicedJetEncoder(self.config)

    def encode(self, image: Tensor) -> Tensor:
        tokens = self._core.encode(_as_image(image))
        return image.new_tensor(tokens)

    def decode(self, tokens: Tensor, *, height: int, width: int) -> Tensor:
        bank = tuple(
            tuple(tuple(float(c) for c in cell) for cell in token) for token in tokens.detach().tolist()
        )
        recon = self._core.decode(bank, height=height, width=width)
        return tokens.new_tensor(recon)

    def reconstruct(self, image: Tensor) -> Tensor:
        rows = _as_image(image)
        recon = self._core.reconstruct(rows)
        return image.new_tensor(recon)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "SlicedJetConfig",
    "SlicedJetEncoder",
    "honesty_payload",
    "worked_example",
]
