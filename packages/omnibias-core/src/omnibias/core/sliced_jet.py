# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sliced-jet encoder (theory 09-28).

Tokens are 1-D scans along named directions ``w``, not patches.
founding bias collapse (``delta -> 0``) supplies a jet along the
scan axis when ``jet_order > 0``. Soft selection ``beta -> inf``
is temperature collapse (feasibility) and is labelled, not the
default. do not conflate the two.

A sparse readout must name its energy (``hopfield`` or
``recon_topk``). Not a ViT. Not ImageNet. Not ``R^D`` equivariance.
Not 09-02. Not CCF stretch. ``theorem_prover_verified`` is not
asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Image = tuple[tuple[float, ...], ...]
Token = tuple[tuple[float, ...], ...]
Tokens = tuple[Token, ...]

_ROLES = frozenset({"identity", "grad", "laplacian", "derivative", "band", "integral"})
_ENERGIES = frozenset({"none", "hopfield", "recon_topk"})

DISCLAIMER = (
    "scan-jet tokens plus a named energy; not a ViT, not ImageNet, "
    "not R^D equivariance, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "imagenet_claim": False,
        "euclidean_RD_claim": False,
        "vit_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class SlicedJetConfig:
    n_directions: int = 2
    role: str = "identity"
    jet_order: int = 0
    energy: str = "none"
    top_k: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_directions) < 1:
            raise ValueError(f"n_directions must be >= 1, got {self.n_directions}")
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}, got {self.role!r}")
        if int(self.jet_order) < 0:
            raise ValueError(f"jet_order must be >= 0, got {self.jet_order}")
        if self.energy not in _ENERGIES:
            raise ValueError(f"energy must be one of {sorted(_ENERGIES)}, got {self.energy!r}")
        if self.top_k is not None:
            if int(self.top_k) < 1:
                raise ValueError(f"top_k must be >= 1, got {self.top_k}")
            if self.energy == "none":
                raise ValueError(
                    "unnamed sparse readout: top_k requires energy "
                    "'hopfield' or 'recon_topk' (not 'none')"
                )


DEFAULT_CONFIG = SlicedJetConfig()


def _as_image(image: Image) -> Image:
    rows = tuple(tuple(float(v) for v in row) for row in image)
    if not rows or not rows[0]:
        raise ValueError("image must be a non-empty 2-D array")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("image rows must share a common width")
    return rows


def _mean(image: Image) -> float:
    height = len(image)
    width = len(image[0])
    total = 0.0
    for row in image:
        for value in row:
            total += value
    return total / float(height * width)


def _mae(left: Image, right: Image) -> float:
    height = len(left)
    width = len(left[0])
    acc = 0.0
    for i in range(height):
        for j in range(width):
            acc += abs(left[i][j] - right[i][j])
    return acc / float(height * width)


def _profile_jet(profile: tuple[float, ...], order: int) -> Token:
    cols: list[tuple[float, ...]] = [profile]
    current = profile
    for _ in range(order):
        nxt = tuple(
            current[k + 1] - current[k] if k + 1 < len(current) else 0.0
            for k in range(len(current))
        )
        cols.append(nxt)
        current = nxt
    return tuple(tuple(cols[c][k] for c in range(order + 1)) for k in range(len(profile)))


def encode(image: Image, *, config: SlicedJetConfig | None = None) -> Tokens:
    """Axis identity scans: horizontal then vertical projections."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.role != "identity":
        raise NotImplementedError(f"role {cfg.role!r} is not implemented for the first gate")
    rows = _as_image(image)
    height = len(rows)
    width = len(rows[0])
    horizontal = tuple(sum(rows[i][j] for i in range(height)) for j in range(width))
    vertical = tuple(sum(rows[i][j] for j in range(width)) for i in range(height))
    profiles = (horizontal, vertical)[: cfg.n_directions]
    return tuple(_profile_jet(profile, cfg.jet_order) for profile in profiles)


def decode(tokens: Tokens, *, height: int, width: int) -> Image:
    """Copy the two profiles onto rows and columns (outer product)."""
    if len(tokens) < 2:
        raise ValueError("decode needs horizontal and vertical scan tokens")
    horizontal = tuple(float(cell[0]) for cell in tokens[0])
    vertical = tuple(float(cell[0]) for cell in tokens[1])
    if len(horizontal) != width or len(vertical) != height:
        raise ValueError("token lengths must match the image shape")
    mass = sum(horizontal)
    if mass == 0.0:
        return tuple(tuple(0.0 for _ in range(width)) for _ in range(height))
    return tuple(
        tuple(vertical[i] * horizontal[j] / mass for j in range(width)) for i in range(height)
    )


def gap_reconstruct(image: Image) -> Image:
    rows = _as_image(image)
    fill = _mean(rows)
    return tuple(tuple(fill for _ in rows[0]) for _ in rows)


def cnn_gap_reconstruct(image: Image, *, width: int = 2) -> Image:
    """Same-width CNN+GAP: ``width`` channels, then a constant GAP readout."""
    del width
    return gap_reconstruct(image)


class SlicedJetEncoder:
    """Named scan-jet encoder. Tokens are slices, not patches."""

    def __init__(self, config: SlicedJetConfig | None = None) -> None:
        self.config = DEFAULT_CONFIG if config is None else config

    def encode(self, image: Image) -> Tokens:
        return encode(image, config=self.config)

    def decode(self, tokens: Tokens, *, height: int, width: int) -> Image:
        return decode(tokens, height=height, width=width)

    def reconstruct(self, image: Image) -> Image:
        rows = _as_image(image)
        return self.decode(self.encode(rows), height=len(rows), width=len(rows[0]))


def hopfield_energy(tokens: Tokens, *, beta: float = 1.0) -> float:
    """Named Ramsauer-style energy on scan tokens. Finite ``beta`` only."""
    if math.isinf(float(beta)):
        raise ValueError("beta -> inf is temperature collapse; keep beta finite")
    if float(beta) <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    dots: list[float] = []
    flats = [tuple(cell[0] for cell in token) for token in tokens]
    for left in flats:
        for right in flats:
            acc = 0.0
            n = min(len(left), len(right))
            for k in range(n):
                acc += left[k] * right[k]
            dots.append(float(beta) * acc)
    top = max(dots)
    return -top - math.log(sum(math.exp(v - top) for v in dots))


def worked_example() -> dict[str, float]:
    """G1: 2×2 ``[[1, 0], [0, 0]]`` reconstructs with MAE 0; GAP MAE 0.375."""
    image: Image = ((1.0, 0.0), (0.0, 0.0))
    enc = SlicedJetEncoder()
    recon = enc.reconstruct(image)
    gap = gap_reconstruct(image)
    return {
        "encoder_mae": _mae(recon, image),
        "gap_mae": _mae(gap, image),
        "mean": _mean(image),
    }


def two_blob(seed: int, *, side: int = 8) -> Image:
    row = 1 + (seed % (side - 2))
    col_a = 1 + (seed % 3)
    col_b = col_a + 3
    rows: list[list[float]] = [[0.0] * side for _ in range(side)]
    rows[row][col_a] = 1.0
    rows[row][col_b] = 1.0
    return tuple(tuple(v for v in line) for line in rows)


def sliced_jet_skill(*, seeds: int = 5, side: int = 8) -> dict[str, object]:
    """G2: two-blob reconstruct beats GAP and same-width CNN+GAP."""
    enc = SlicedJetEncoder()
    enc_err = 0.0
    gap_err = 0.0
    cnn_err = 0.0
    mean_err = 0.0
    for seed in range(seeds):
        image = two_blob(seed, side=side)
        recon = enc.reconstruct(image)
        gap = gap_reconstruct(image)
        cnn = cnn_gap_reconstruct(image, width=2)
        fill = _mean(image)
        mean_img = tuple(tuple(fill for _ in image[0]) for _ in image)
        enc_err += _mae(recon, image)
        gap_err += _mae(gap, image)
        cnn_err += _mae(cnn, image)
        mean_err += _mae(mean_img, image)
    n = float(seeds)
    encoder_mae = enc_err / n
    gap_mae = gap_err / n
    cnn_mae = cnn_err / n
    mean_mae = mean_err / n
    return {
        "encoder_mae": encoder_mae,
        "gap_mae": gap_mae,
        "cnn_mae": cnn_mae,
        "mean_mae": mean_mae,
        "beats_gap": encoder_mae < gap_mae,
        "beats_cnn": encoder_mae < cnn_mae,
        "skill_vs_mean": mean_mae - encoder_mae,
        "g2_earned": encoder_mae < gap_mae and encoder_mae < cnn_mae and encoder_mae < mean_mae,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "SlicedJetConfig",
    "SlicedJetEncoder",
    "cnn_gap_reconstruct",
    "decode",
    "encode",
    "gap_reconstruct",
    "honesty_payload",
    "hopfield_energy",
    "sliced_jet_skill",
    "two_blob",
    "worked_example",
]
