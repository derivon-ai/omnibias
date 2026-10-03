# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tree-directed polygonal annulus layouts for Route-2 Part-A searches."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, sqrt

from omnibias.geometry.algebraic import PolygonalAnnulus, RationalPolygon, _nesting
from omnibias.geometry.part_a_target import count_octic_ovals
from omnibias.geometry.patchwork import RootedTree

__all__ = [
    "LayoutParams",
    "fourteen_orbit_centers",
    "layout_annuli_for_tree",
    "normalize_annulus_layout",
    "parents_to_rooted_tree",
    "square_polygon_xy",
    "validate_annulus_layout",
]


def square_polygon_xy(cx: Fraction, cy: Fraction, half: Fraction) -> RationalPolygon:
    """Axis-aligned square with exact rational center and half-width."""
    if half <= 0:
        raise ValueError("half must be positive")
    return RationalPolygon(
        (
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        )
    )


@dataclass(frozen=True)
class LayoutParams:
    """Geometric knobs for tree-directed annulus placement."""

    exterior_center: Fraction = Fraction(30)
    exterior_half: Fraction = Fraction(3, 2)
    exterior_inner_ratio: Fraction = Fraction(5, 6)
    center_outer: Fraction = Fraction(11)
    inner_ratio: Fraction = Fraction(4, 5)
    leaf_half: Fraction = Fraction(1, 2)
    nested_leaf_half: Fraction = Fraction(2, 5)
    deep_leaf_half: Fraction = Fraction(1, 8)
    branch_outer_ratio: Fraction = Fraction(1, 3)
    branch_offset: Fraction = Fraction(0)
    deep_branch_scale: Fraction = Fraction(2, 3)
    leaf_fill: Fraction = Fraction(7, 10)


def parents_to_rooted_tree(parents: tuple[int, ...]) -> RootedTree:
    """Rebuild the rooted complement tree from ``barrier_parents`` indices."""
    children: list[list[int]] = [[] for _ in range(len(parents) + 1)]
    for index, parent in enumerate(parents):
        children[0 if parent < 0 else parent + 1].append(index + 1)

    def signature(node: int) -> RootedTree:
        return tuple(sorted((signature(child) for child in children[node]), key=repr))

    return signature(0)


def validate_annulus_layout(
    annuli: tuple[PolygonalAnnulus, ...],
    target: RootedTree,
) -> None:
    """Raise if annuli are not pairwise valid or do not realize ``target``."""
    if count_octic_ovals(target) != len(annuli):
        raise ValueError(
            f"expected {count_octic_ovals(target)} annuli for the target tree, got {len(annuli)}"
        )
    for index, annulus in enumerate(annuli):
        for other_index in range(index + 1, len(annuli)):
            if not (
                annulus.outer.disjoint(annuli[other_index].outer)
                or annulus.outer.strictly_contains(annuli[other_index].outer)
                or annuli[other_index].outer.strictly_contains(annulus.outer)
            ):
                raise ValueError(
                    f"annuli {index} and {other_index} have intersecting non-nested boundaries"
                )
    parents = _nesting(annuli)
    realized = parents_to_rooted_tree(parents)
    if realized != target:
        raise ValueError(f"layout tree {realized!r} does not match target {target!r}")


def _distribute_centers(
    count: int,
    radius: Fraction,
    center: tuple[Fraction, Fraction],
    *,
    min_separation: Fraction | None = None,
) -> tuple[tuple[Fraction, Fraction], ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (center,)
    if count == 2:
        offset = max(radius / Fraction(3), (min_separation or Fraction(0)) / Fraction(2))
        return (
            (center[0] - offset, center[1]),
            (center[0] + offset, center[1]),
        )
    cols = 7 if count >= 14 else max(1, ceil(sqrt(count)))
    rows = max(1, ceil(count / cols))
    span_x = radius * Fraction(2)
    span_y = radius * Fraction(2)
    step_x = span_x / Fraction(cols + 1)
    step_y = span_y / Fraction(rows + 1)
    if min_separation is not None:
        step_x = max(step_x, min_separation)
        step_y = max(step_y, min_separation)
    positions: list[tuple[Fraction, Fraction]] = []
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= count:
                break
            cx = center[0] - span_x / 2 + step_x * (col + 1)
            cy = center[1] - span_y / 2 + step_y * (row + 1)
            positions.append((cx, cy))
    return tuple(positions)


def _leaf_annulus(
    center: tuple[Fraction, Fraction],
    outer_half: Fraction,
    params: LayoutParams,
) -> PolygonalAnnulus:
    inner = outer_half * params.exterior_inner_ratio
    return PolygonalAnnulus(
        square_polygon_xy(center[0], center[1], inner),
        square_polygon_xy(center[0], center[1], outer_half),
    )


def _paired_leaf_centers(
    center: tuple[Fraction, Fraction],
    radius: Fraction,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    return (
        (center[0] - radius, center[1]),
        (center[0] + radius, center[1]),
    )


def fourteen_orbit_centers(
    center: tuple[Fraction, Fraction],
    radius: Fraction,
    leaf_outer_half: Fraction,
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Place ``14 = 8 + 4 + 2`` centers in exact sign-symmetric orbits."""
    if radius <= 0 or leaf_outer_half <= 0:
        raise ValueError("orbit radius and leaf half-width must be positive")
    step = radius / 4
    if step <= 2 * leaf_outer_half:
        raise ValueError("14-orbit placement has insufficient leaf separation")
    cx, cy = center
    orbit8 = tuple(
        (cx + sx * x, cy + sy * y)
        for x, y in ((4 * step, step), (step, 4 * step))
        for sx in (-1, 1)
        for sy in (-1, 1)
    )
    orbit4 = (
        (cx - 2 * step, cy),
        (cx + 2 * step, cy),
        (cx, cy - 2 * step),
        (cx, cy + 2 * step),
    )
    orbit2 = (
        (cx - step / 2, cy),
        (cx + step / 2, cy),
    )
    return orbit8 + orbit4 + orbit2


def _layout_nested(
    node: RootedTree,
    center: tuple[Fraction, Fraction],
    outer_half: Fraction,
    params: LayoutParams,
    depth: int = 0,
) -> tuple[PolygonalAnnulus, ...]:
    if not node:
        raise ValueError("nested layout requires a nonempty node")
    inner_half = outer_half * params.inner_ratio
    annuli: list[PolygonalAnnulus] = [
        PolygonalAnnulus(
            square_polygon_xy(center[0], center[1], inner_half),
            square_polygon_xy(center[0], center[1], outer_half),
        )
    ]
    leaves = tuple(child for child in node if child == ())
    branches = tuple(child for child in node if child != ())
    if depth == 0:
        leaf_outer = params.leaf_half
    elif len(leaves) > 3:
        leaf_outer = params.deep_leaf_half
    elif depth == 1:
        leaf_outer = params.nested_leaf_half
    else:
        leaf_outer = params.deep_leaf_half
    if len(leaves) == 2:
        positions = _paired_leaf_centers(center, inner_half * params.leaf_fill)
    elif len(leaves) == 14:
        margin = leaf_outer + Fraction(1, 64)
        fill_radius = min(inner_half * params.leaf_fill, inner_half - margin)
        positions = fourteen_orbit_centers(center, fill_radius, leaf_outer)
    else:
        margin = leaf_outer + Fraction(1, 64)
        fill_radius = min(inner_half * params.leaf_fill, inner_half - margin)
        if fill_radius <= margin:
            raise ValueError("leaf placement does not fit inside the declared inner collar")
        separation = min(leaf_outer * Fraction(9, 4), fill_radius * Fraction(2, max(len(leaves), 1)))
        positions = _distribute_centers(
            len(leaves),
            fill_radius,
            center,
            min_separation=separation,
        )
    for position in positions:
        annuli.append(_leaf_annulus(position, leaf_outer, params))
    branch_scale = params.deep_branch_scale if depth >= 1 else Fraction(1)
    branch_outer = inner_half * params.branch_outer_ratio * branch_scale
    for branch in branches:
        # The offset is dimensionless and exact; zero is the symmetry-fixed tier.
        branch_center = (
            center[0],
            center[1] + inner_half * params.branch_offset,
        )
        annuli.extend(_layout_nested(branch, branch_center, branch_outer, params, depth + 1))
    return tuple(annuli)


def _layout_top_level(tree: RootedTree, params: LayoutParams) -> tuple[PolygonalAnnulus, ...]:
    leaves = tuple(child for child in tree if child == ())
    branches = tuple(child for child in tree if child != ())
    annuli: list[PolygonalAnnulus] = []
    quadrant_offsets = (
        (-params.exterior_center, -params.exterior_center),
        (params.exterior_center, -params.exterior_center),
        (-params.exterior_center, params.exterior_center),
        (params.exterior_center, params.exterior_center),
    )
    for index, _leaf in enumerate(leaves):
        cx, cy = quadrant_offsets[index % len(quadrant_offsets)]
        outer = params.exterior_half
        annuli.append(_leaf_annulus((cx, cy), outer, params))
    center = (Fraction(0), Fraction(0))
    for branch in branches:
        annuli.extend(_layout_nested(branch, center, params.center_outer, params))
    return tuple(annuli)


def _scaled_params(params: LayoutParams, scale: Fraction) -> LayoutParams:
    return LayoutParams(
        exterior_center=params.exterior_center,
        exterior_half=params.exterior_half * scale,
        exterior_inner_ratio=params.exterior_inner_ratio,
        center_outer=params.center_outer * scale,
        inner_ratio=params.inner_ratio,
        leaf_half=params.leaf_half * scale,
        nested_leaf_half=params.nested_leaf_half * scale,
        deep_leaf_half=params.deep_leaf_half * scale,
        branch_outer_ratio=params.branch_outer_ratio,
        branch_offset=params.branch_offset,
        deep_branch_scale=params.deep_branch_scale,
        leaf_fill=params.leaf_fill,
    )


def _scale_polygon(polygon: RationalPolygon, scale: Fraction) -> RationalPolygon:
    return RationalPolygon(tuple((x * scale, y * scale) for x, y in polygon.vertices))


def normalize_annulus_layout(
    annuli: tuple[PolygonalAnnulus, ...],
    *,
    chart_radius: Fraction = Fraction(1),
) -> tuple[PolygonalAnnulus, ...]:
    """Apply one exact positive scale so every affine vertex is ``O(1)``."""
    if chart_radius <= 0:
        raise ValueError("chart_radius must be positive")
    maximum = max(
        abs(coordinate)
        for annulus in annuli
        for polygon in (annulus.inner, annulus.outer)
        for vertex in polygon.vertices
        for coordinate in vertex
    )
    if maximum <= 0:
        raise ValueError("cannot normalize a zero-size annulus layout")
    scale = chart_radius / maximum
    return tuple(
        PolygonalAnnulus(
            _scale_polygon(annulus.inner, scale),
            _scale_polygon(annulus.outer, scale),
        )
        for annulus in annuli
    )


def layout_annuli_for_tree(
    tree: RootedTree,
    *,
    params: LayoutParams | None = None,
    base_half: Fraction = Fraction(1, 16),
    growth: Fraction = Fraction(5, 4),
    normalize: bool = True,
    shrink_to_fit: bool = True,
) -> tuple[PolygonalAnnulus, ...]:
    """Build a tree-directed 22-oval layout and validate it against ``tree``.

    ``base_half`` and ``growth`` are retained for API compatibility with the
    legacy staircase layout but are not used by the tree-directed generator.
    """
    _ = (base_half, growth)
    expected = count_octic_ovals(tree)
    if expected < 1:
        raise ValueError("tree must encode at least one oval")
    base_params = params or LayoutParams()
    last_error: ValueError | None = None
    steps = range(11) if shrink_to_fit else range(1)
    for step in steps:
        scale = Fraction(11 - step, 11)
        try:
            annuli = _layout_top_level(tree, _scaled_params(base_params, scale))
            validate_annulus_layout(annuli, tree)
            normalized = normalize_annulus_layout(annuli) if normalize else annuli
            validate_annulus_layout(normalized, tree)
            return normalized
        except ValueError as error:
            last_error = error
    raise ValueError(f"could not realize a valid layout for tree {tree!r}: {last_error}")
