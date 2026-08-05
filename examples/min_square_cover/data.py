# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Binary-image instances and the greedy set-cover baseline.

Every instance is a ``(M, N)`` 0/1 image plus a fixed square ``side``; the task is to
cover all 1-pixels with the fewest axis-aligned ``side x side`` squares (aligned to the
pixel grid, fully inside the image). The synthetic generators are deterministic so the
example runs fully offline. :func:`greedy_cover` is the classic max-overlap heuristic
(the accepted Stack Overflow answer) and the first-order baseline every arm is measured
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

#: The synthetic shape families (deterministic; no download, no research tree).
SHAPES: tuple[str, ...] = ("blob", "ring", "l_shape", "scatter", "reference")

#: Default square side per shape (chosen so several squares are needed).
_DEFAULT_SIDE: dict[str, int] = {
    "blob": 6,
    "ring": 5,
    "l_shape": 6,
    "scatter": 5,
    "reference": 8,
}


@dataclass(frozen=True)
class Instance:
    """One covering instance: a binary ``image`` and the fixed square ``side``."""

    name: str
    image: Tensor  # (M, N), values in {0.0, 1.0}
    side: int

    @property
    def n_ones(self) -> int:
        """Number of 1-pixels that must be covered."""
        return int(self.image.sum())

    @property
    def shape(self) -> tuple[int, int]:
        """Image ``(rows, cols)``."""
        return (int(self.image.shape[0]), int(self.image.shape[1]))


def _disk(size: int, cy: float, cx: float, radius: float) -> Tensor:
    ys = torch.arange(size, dtype=torch.get_default_dtype()).reshape(-1, 1)
    xs = torch.arange(size, dtype=torch.get_default_dtype()).reshape(1, -1)
    return ((ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2).to(torch.get_default_dtype())


def make_image(name: str, *, size: int = 24, seed: int = 0) -> Tensor:
    """Build one deterministic ``size x size`` binary image from :data:`SHAPES`."""
    if name not in SHAPES:
        raise ValueError(f"unknown shape {name!r}; choose from {SHAPES}")
    c = (size - 1) / 2.0
    if name == "blob":
        return _disk(size, c, c, size * 0.34)
    if name == "ring":
        outer = _disk(size, c, c, size * 0.42)
        inner = _disk(size, c, c, size * 0.22)
        return (outer - inner).clamp(0.0, 1.0)
    if name == "l_shape":
        img = torch.zeros(size, size, dtype=torch.get_default_dtype())
        arm = max(3, size // 4)
        img[size // 6 :, size // 6 : size // 6 + arm] = 1.0
        img[size - size // 6 - arm : size - size // 6, size // 6 :] = 1.0
        return img
    if name == "scatter":
        g = torch.Generator().manual_seed(seed)
        img = torch.zeros(size, size, dtype=torch.get_default_dtype())
        n_blobs = 6
        for _ in range(n_blobs):
            cy = float(torch.randint(2, size - 2, (1,), generator=g))
            cx = float(torch.randint(2, size - 2, (1,), generator=g))
            img = img + _disk(size, cy, cx, 1.4)
        return img.clamp(0.0, 1.0)
    # reference: two offset filled blocks (a compact multi-square instance).
    img = torch.zeros(size, size, dtype=torch.get_default_dtype())
    q = size // 3
    img[1 : 1 + q, 1 : 1 + q + 2] = 1.0
    img[size - 1 - q : size - 1, size - 1 - q - 2 : size - 1] = 1.0
    img[q : q + 3, q : size - q] = 1.0
    return img.clamp(0.0, 1.0)


def make_instance(name: str, *, size: int = 24, side: int | None = None, seed: int = 0) -> Instance:
    """Assemble an :class:`Instance` (image + square side) for shape ``name``."""
    image = make_image(name, size=size, seed=seed)
    resolved = _DEFAULT_SIDE[name] if side is None else side
    return Instance(name=name, image=image, side=int(resolved))


def synthetic_instances(
    shapes: tuple[str, ...] = SHAPES, *, size: int = 24, seed: int = 0
) -> list[Instance]:
    """One :class:`Instance` per requested shape (deterministic, offline)."""
    return [make_instance(s, size=size, seed=seed) for s in shapes]


def _load_grayscale(path: str, max_size: int | None) -> Tensor:
    """Load ``path`` as a 2-D grayscale float tensor in ``[0, 1]`` (lazy Pillow, then skimage)."""
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore[assignment]
    if Image is not None:
        img = Image.open(path).convert("L")
        if max_size is not None and max(img.size) > max_size:
            scale = max_size / max(img.size)
            new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(new)
        import numpy as np

        return torch.as_tensor(np.asarray(img, dtype="float64") / 255.0)
    try:
        import numpy as np
        from skimage.color import rgb2gray
        from skimage.io import imread
        from skimage.transform import resize as sk_resize
    except ImportError as exc:  # pragma: no cover - exercised only without both libraries
        raise ImportError(
            "real-image loading needs Pillow or scikit-image; install one, e.g. `pip install pillow`"
        ) from exc
    arr = imread(path)
    if arr.ndim == 3:
        arr = rgb2gray(arr[..., :3])
    arr = arr.astype("float64")
    if float(arr.max()) > 1.0:
        arr = arr / 255.0
    if max_size is not None and max(arr.shape) > max_size:
        scale = max_size / max(arr.shape)
        target = (max(1, round(arr.shape[0] * scale)), max(1, round(arr.shape[1] * scale)))
        arr = sk_resize(arr, target, anti_aliasing=True)
    return torch.as_tensor(arr)


def load_binary_image(
    path: str, *, threshold: float = 0.5, invert: bool = False, max_size: int | None = None
) -> Tensor:
    """Load a real image file as a ``(M, N)`` binary ``{0.0, 1.0}`` tensor (lazy Pillow/skimage).

    The image is converted to grayscale in ``[0, 1]`` and thresholded: pixels ``>= threshold``
    become 1-pixels to cover (pass ``invert=True`` for dark-foreground-on-light images). Give
    ``max_size`` to downscale the longest side first (keeps the candidate-square count tractable
    for the certified LP register). Requires ``Pillow`` or ``scikit-image`` -- neither is a
    dependency of the example, so this raises a helpful ``ImportError`` if both are missing.
    """
    gray = _load_grayscale(path, max_size)
    fg = gray >= threshold
    if invert:
        fg = ~fg
    return fg.to(torch.get_default_dtype())


def load_instance(
    path: str,
    side: int,
    *,
    name: str | None = None,
    threshold: float = 0.5,
    invert: bool = False,
    max_size: int | None = None,
) -> Instance:
    """Load a real image file into an :class:`Instance` with square ``side`` (see
    :func:`load_binary_image` for the binarisation options)."""
    image = load_binary_image(path, threshold=threshold, invert=invert, max_size=max_size)
    label = name if name is not None else Path(path).stem
    return Instance(name=label, image=image, side=int(side))


def _window_sums(image: Tensor, side: int) -> Tensor:
    """Sliding ``side x side`` window sums at every valid top-left, shape ``(M-s+1, N-s+1)``."""
    x = image.reshape(1, 1, *image.shape)
    kernel = torch.ones(1, 1, side, side, dtype=image.dtype)
    return F.conv2d(x, kernel).reshape(image.shape[0] - side + 1, image.shape[1] - side + 1)


def _clamp_side(image: Tensor, side: int) -> int:
    return max(1, min(side, image.shape[0], image.shape[1]))


def greedy_cover(image: Tensor, side: int) -> list[tuple[int, int]]:
    """Max-overlap greedy cover: repeatedly place the square covering the most uncovered 1s.

    Returns the placed squares as ``(row, col)`` top-left corners. This is the accepted
    Stack Overflow heuristic and the baseline the continuous solver must match or beat.
    """
    side = _clamp_side(image, side)
    remaining = image.clone()
    squares: list[tuple[int, int]] = []
    budget = image.shape[0] * image.shape[1]
    while float(remaining.sum()) > 0.0 and len(squares) <= budget:
        sums = _window_sums(remaining, side)
        flat = int(torch.argmax(sums))
        r, cc = divmod(flat, sums.shape[1])
        squares.append((r, cc))
        remaining[r : r + side, cc : cc + side] = 0.0
    return squares


def squares_to_mask(shape: tuple[int, int], squares: list[tuple[int, int]], side: int) -> Tensor:
    """Boolean mask of pixels covered by ``squares`` (each a ``side x side`` block)."""
    mask = torch.zeros(shape, dtype=torch.bool)
    for r, cc in squares:
        mask[r : r + side, cc : cc + side] = True
    return mask


def is_feasible(image: Tensor, squares: list[tuple[int, int]], side: int) -> bool:
    """``True`` iff every 1-pixel of ``image`` is covered by some square."""
    covered = squares_to_mask(image.shape, squares, side)
    return bool((image.to(torch.bool) & ~covered).sum() == 0)


def coverage_fraction(image: Tensor, squares: list[tuple[int, int]], side: int) -> float:
    """Fraction of 1-pixels covered by ``squares`` (1.0 == feasible)."""
    ones = image.to(torch.bool)
    total = int(ones.sum())
    if total == 0:
        return 1.0
    covered = squares_to_mask(image.shape, squares, side)
    return float((ones & covered).sum()) / total


__all__ = [
    "Instance",
    "SHAPES",
    "coverage_fraction",
    "greedy_cover",
    "is_feasible",
    "load_binary_image",
    "load_instance",
    "make_image",
    "make_instance",
    "squares_to_mask",
    "synthetic_instances",
]
