# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Part-A octic target selection among the two open (19,3) schemes.

Geiselmann et al. (arXiv:2602.06888) list two algebraically open maximal
octics with ``(p,n)=(19,3)``. The paper's nested-box T-curve is depth-3 but
narrow, which is evidence that patchworking reaches shallow nests first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omnibias.geometry.patchwork import RootedTree, target_octic_region_tree

PartATargetName = Literal["wide_deep", "sibling_narrow"]

__all__ = [
    "PARTA_SELECTED_TARGET",
    "PartATargetName",
    "PartATargetSelection",
    "count_octic_ovals",
    "sibling_octic_region_tree",
    "target_tree",
    "wide_deep_octic_region_tree",
]


def count_octic_ovals(tree: RootedTree) -> int:
    """Count ovals encoded by a rooted complement-region tree.

    Each leaf ``()`` is one unnested oval. Each nonempty child tuple is one
    nested oval containing its descendant ovals (scheme notation ``1<...>``).
    """
    if not tree:
        return 0
    total = 0
    for child in tree:
        if child == ():
            total += 1
        else:
            total += 1 + count_octic_ovals(child)
    return total


def wide_deep_octic_region_tree() -> RootedTree:
    """Return the rooted-tree signature of ``4 + 1<2 + 1<14>>``."""
    return target_octic_region_tree()


def sibling_octic_region_tree() -> RootedTree:
    """Return the rooted-tree signature of ``14 + 1<2 + 1<4>>``."""
    four = tuple(() for _ in range(4))
    nested = tuple(sorted(((), (), four), key=repr))
    fourteen = tuple(() for _ in range(14))
    return tuple(sorted((*fourteen, nested), key=repr))


@dataclass(frozen=True)
class PartATargetSelection:
    name: PartATargetName
    scheme: str
    tree: RootedTree
    rationale: str

    @property
    def oval_count(self) -> int:
        return count_octic_ovals(self.tree)


def target_tree(name: PartATargetName = "sibling_narrow") -> RootedTree:
    if name == "wide_deep":
        return wide_deep_octic_region_tree()
    return sibling_octic_region_tree()


PARTA_SELECTED_TARGET = PartATargetSelection(
    name="sibling_narrow",
    scheme="14 + 1<2 + 1<4>>",
    tree=sibling_octic_region_tree(),
    rationale=(
        "The paper's own (19,3) nested-box T-curve is depth-3 but narrow; "
        "the sibling scheme is less wide than 4 + 1<2 + 1<14>> and is the "
        "recommended first search target before spending extended budget."
    ),
)
