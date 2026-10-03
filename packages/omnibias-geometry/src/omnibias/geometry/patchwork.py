# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact combinatorics for planar Viro patchworks.

The module verifies a unimodular triangulation of the degree-``d`` Newton
triangle, reflects its signs and triangles into the Viro chart, and computes
the projective T-curve and its complement-region tree.  Every predicate is
over integers or :class:`fractions.Fraction`; no sampled geometry is used.

For even degree, the rooted complement tree is exactly the nesting tree of
the projective ovals.  The root is identified topologically: its lift already
connects an antipodal pair on the boundary before the projective quotient.
Odd-degree T-curves can still be constructed, but their pseudoline complement
does not carry the oval-only region-tree contract used here.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

LatticePoint = tuple[int, int]
RationalPoint = tuple[Fraction, Fraction]
Triangle = tuple[LatticePoint, LatticePoint, LatticePoint]


def lattice_points(degree: int) -> tuple[LatticePoint, ...]:
    """Return the degree-``degree`` Newton-triangle lattice points."""
    if type(degree) is not int or degree < 1:
        raise ValueError("degree must be a positive integer")
    return tuple(
        (i, j)
        for total in range(degree + 1)
        for i in range(total + 1)
        for j in (total - i,)
    )


def _det(a: LatticePoint, b: LatticePoint, c: LatticePoint) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _edge(a: LatticePoint, b: LatticePoint) -> tuple[LatticePoint, LatticePoint]:
    return (a, b) if a < b else (b, a)


def _on_newton_boundary(point: LatticePoint, degree: int) -> bool:
    return point[0] == 0 or point[1] == 0 or point[0] + point[1] == degree


def _boundary_edge(a: LatticePoint, b: LatticePoint, degree: int) -> bool:
    return (
        (a[0] == b[0] == 0)
        or (a[1] == b[1] == 0)
        or (a[0] + a[1] == b[0] + b[1] == degree)
    )


def _orientation(a: LatticePoint, b: LatticePoint, c: LatticePoint) -> int:
    value = _det(a, b, c)
    return (value > 0) - (value < 0)


def _on_segment(a: LatticePoint, b: LatticePoint, p: LatticePoint) -> bool:
    return (
        _orientation(a, b, p) == 0
        and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(
    a: LatticePoint,
    b: LatticePoint,
    c: LatticePoint,
    d: LatticePoint,
) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (o1 == 0 and _on_segment(a, b, c))
        or (o2 == 0 and _on_segment(a, b, d))
        or (o3 == 0 and _on_segment(c, d, a))
        or (o4 == 0 and _on_segment(c, d, b))
    )


@dataclass(frozen=True)
class Triangulation:
    """An exact unimodular triangulation of a Newton triangle."""

    degree: int
    triangles: tuple[Triangle, ...]

    def __post_init__(self) -> None:
        points = frozenset(lattice_points(self.degree))
        normalized: list[Triangle] = []
        for raw in self.triangles:
            if len(raw) != 3 or any(
                len(point) != 2 or any(type(value) is not int for value in point)
                for point in raw
            ):
                raise TypeError("triangles require three integer lattice points")
            triangle = tuple(raw)
            if len(set(triangle)) != 3 or any(point not in points for point in triangle):
                raise ValueError("triangle vertices must be distinct Newton-triangle points")
            determinant = _det(*triangle)
            if abs(determinant) != 1:
                raise ValueError("every triangle must be unimodular")
            if determinant < 0:
                triangle = (triangle[0], triangle[2], triangle[1])
            normalized.append(triangle)
        canonical = tuple(sorted(normalized, key=lambda t: tuple(sorted(t))))
        if len(set(tuple(sorted(t)) for t in canonical)) != len(canonical):
            raise ValueError("duplicate triangles are not allowed")
        object.__setattr__(self, "triangles", canonical)
        self._validate_cover(points)

    def _validate_cover(self, points: frozenset[LatticePoint]) -> None:
        if len(self.triangles) != self.degree**2:
            raise ValueError("unimodular triangle count does not cover the Newton triangle")
        used = frozenset(point for triangle in self.triangles for point in triangle)
        if used != points:
            raise ValueError("the triangulation must use every Newton-triangle lattice point")
        incidence: dict[tuple[LatticePoint, LatticePoint], int] = defaultdict(int)
        for triangle in self.triangles:
            for i in range(3):
                incidence[_edge(triangle[i], triangle[(i + 1) % 3])] += 1
        if len(incidence) != len(points) + len(self.triangles) - 1:
            raise ValueError("edge count violates the disk Euler identity")
        for (a, b), count in incidence.items():
            expected = 1 if _boundary_edge(a, b, self.degree) else 2
            if count != expected:
                raise ValueError("triangle faces do not form a complete face-to-face cover")
        edges = tuple(incidence)
        for i, (a, b) in enumerate(edges):
            for c, d in edges[i + 1:]:
                common = {a, b} & {c, d}
                if common:
                    if len(common) == 1:
                        shared = next(iter(common))
                        other_ab = b if a == shared else a
                        other_cd = d if c == shared else c
                        if _on_segment(a, b, other_cd) or _on_segment(c, d, other_ab):
                            raise ValueError("edges overlap beyond a common endpoint")
                    continue
                if _segments_intersect(a, b, c, d):
                    raise ValueError("triangulation edges cross")

    @property
    def vertices(self) -> tuple[LatticePoint, ...]:
        return lattice_points(self.degree)

    @property
    def edges(self) -> tuple[tuple[LatticePoint, LatticePoint], ...]:
        return tuple(sorted({
            _edge(triangle[i], triangle[(i + 1) % 3])
            for triangle in self.triangles
            for i in range(3)
        }))


def staircase_triangulation(degree: int) -> Triangulation:
    """Return the standard regular unimodular staircase triangulation."""
    triangles: list[Triangle] = []
    for i in range(degree):
        for j in range(degree - i):
            triangles.append(((i, j), (i + 1, j), (i, j + 1)))
            if i + j <= degree - 2:
                triangles.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
    return Triangulation(degree, tuple(triangles))


def flip_neighbors(triangulation: Triangulation) -> tuple[Triangulation, ...]:
    """Return every valid unimodular interior-edge flip."""
    incidence: dict[
        tuple[LatticePoint, LatticePoint],
        list[int],
    ] = defaultdict(list)
    for index, triangle in enumerate(triangulation.triangles):
        for i in range(3):
            incidence[_edge(triangle[i], triangle[(i + 1) % 3])].append(index)
    neighbors: dict[tuple[tuple[LatticePoint, ...], ...], Triangulation] = {}
    for edge, adjacent in incidence.items():
        if len(adjacent) != 2:
            continue
        a, b = edge
        c = next(point for point in triangulation.triangles[adjacent[0]] if point not in edge)
        d = next(point for point in triangulation.triangles[adjacent[1]] if point not in edge)
        if _orientation(c, d, a) * _orientation(c, d, b) >= 0:
            continue
        if abs(_det(c, d, a)) != 1 or abs(_det(d, c, b)) != 1:
            continue
        triangles = [
            triangle
            for index, triangle in enumerate(triangulation.triangles)
            if index not in adjacent
        ]
        triangles.extend(((c, d, a), (d, c, b)))
        try:
            candidate = Triangulation(triangulation.degree, tuple(triangles))
        except ValueError:
            continue
        key = tuple(tuple(sorted(triangle)) for triangle in candidate.triangles)
        neighbors[key] = candidate
    return tuple(neighbors[key] for key in sorted(neighbors))


@dataclass(frozen=True)
class SignDistribution:
    """Sign bits on the Newton triangle, in :func:`lattice_points` order."""

    degree: int
    bits: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = len(lattice_points(self.degree))
        if len(self.bits) != expected or any(type(bit) is not int or bit not in (0, 1) for bit in self.bits):
            raise ValueError(f"exactly {expected} bits are required")

    @classmethod
    def create(
        cls,
        degree: int,
        values: Mapping[LatticePoint, int] | Sequence[int],
    ) -> SignDistribution:
        """Create a distribution from a complete mapping or ordered bits."""
        points = lattice_points(degree)
        if isinstance(values, Mapping):
            if set(values) != set(points):
                raise ValueError("the sign mapping must cover exactly the Newton lattice")
            bits = tuple(values[point] for point in points)
        else:
            bits = tuple(values)
        return cls(degree, bits)

    def bit(self, point: LatticePoint) -> int:
        """Return a base-triangle sign bit."""
        try:
            index = lattice_points(self.degree).index(point)
        except ValueError as exc:
            raise ValueError("point is outside the Newton triangle") from exc
        return self.bits[index]

    def reflected_bit(self, point: LatticePoint) -> int:
        """Return the quadrant-reflected Viro sign bit."""
        x, y = point
        base = (abs(x), abs(y))
        if sum(base) > self.degree:
            raise ValueError("point is outside the reflected Newton polygon")
        return (self.bit(base) + (abs(x) if x < 0 else 0) + (abs(y) if y < 0 else 0)) % 2


@dataclass(frozen=True)
class PatchworkSegment:
    """One exact straight segment of the reflected projective T-curve."""

    start: RationalPoint
    end: RationalPoint

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("a patchwork segment must have distinct endpoints")


RootedTree = tuple["RootedTree", ...]


@dataclass(frozen=True)
class PatchworkResult:
    """Exact T-curve components and the even-degree rooted region tree."""

    degree: int
    segments: tuple[PatchworkSegment, ...]
    components: tuple[tuple[int, ...], ...]
    region_children: tuple[tuple[int, ...], ...]
    rooted_tree: RootedTree

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def region_count(self) -> int:
        return len(self.region_children)


class _UnionFind:
    def __init__(self, items: Sequence[object]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: object) -> object:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: object, b: object) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _reflected_mesh(triangulation: Triangulation) -> tuple[Triangle, ...]:
    triangles: set[tuple[LatticePoint, ...]] = set()
    result: list[Triangle] = []
    for sx, sy in product((-1, 1), repeat=2):
        for triangle in triangulation.triangles:
            reflected = tuple((sx * x, sy * y) for x, y in triangle)
            key = tuple(sorted(reflected))
            if key in triangles:
                continue
            triangles.add(key)
            if _det(*reflected) < 0:
                reflected = (reflected[0], reflected[2], reflected[1])
            result.append(reflected)
    return tuple(sorted(result, key=lambda t: tuple(sorted(t))))


def _midpoint(a: LatticePoint, b: LatticePoint) -> RationalPoint:
    return (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))


def _on_reflected_boundary(point: tuple[Fraction, Fraction], degree: int) -> bool:
    return abs(point[0]) + abs(point[1]) == degree


def _projective_point(point: RationalPoint, degree: int) -> RationalPoint:
    if not _on_reflected_boundary(point, degree):
        return point
    antipode = (-point[0], -point[1])
    return min(point, antipode)


def _tree_signature(
    adjacency: Mapping[int, set[int]],
    node: int,
    parent: int | None,
) -> RootedTree:
    return tuple(sorted(
        (_tree_signature(adjacency, child, node) for child in adjacency[node] if child != parent),
        key=repr,
    ))


def _canonical_children(
    adjacency: Mapping[int, set[int]],
    root: int,
) -> tuple[tuple[int, ...], ...]:
    children: list[list[int]] = []

    def visit(node: int, parent: int | None) -> int:
        index = len(children)
        children.append([])
        ordered = sorted(
            (child for child in adjacency[node] if child != parent),
            key=lambda child: repr(_tree_signature(adjacency, child, node)),
        )
        children[index].extend(visit(child, node) for child in ordered)
        return index

    visit(root, None)
    return tuple(tuple(row) for row in children)


def patchwork_curve(
    triangulation: Triangulation,
    signs: SignDistribution,
) -> PatchworkResult:
    """Construct the exact reflected T-curve and its projective region tree."""
    if triangulation.degree != signs.degree:
        raise ValueError("triangulation and signs must have the same degree")
    degree = triangulation.degree
    mesh = _reflected_mesh(triangulation)
    vertices = tuple(sorted({point for triangle in mesh for point in triangle}))
    mesh_edges = tuple(sorted({
        _edge(triangle[i], triangle[(i + 1) % 3])
        for triangle in mesh
        for i in range(3)
    }))

    segment_records: list[tuple[RationalPoint, RationalPoint, Triangle]] = []
    for triangle in mesh:
        changed = [
            _projective_point(_midpoint(triangle[i], triangle[(i + 1) % 3]), degree)
            for i in range(3)
            if signs.reflected_bit(triangle[i]) != signs.reflected_bit(triangle[(i + 1) % 3])
        ]
        if changed:
            if len(changed) != 2 or changed[0] == changed[1]:
                raise ArithmeticError("a nonmonochromatic triangle must contribute one segment")
            a, b = sorted(changed)
            segment_records.append((a, b, triangle))

    unique_records: dict[tuple[RationalPoint, RationalPoint], Triangle] = {}
    for a, b, triangle in segment_records:
        if (a, b) in unique_records:
            raise ArithmeticError("duplicate patchwork segments are not allowed")
        unique_records[(a, b)] = triangle
    segment_keys = tuple(sorted(unique_records))
    segments = tuple(PatchworkSegment(a, b) for a, b in segment_keys)

    curve_points = tuple(sorted({point for segment in segment_keys for point in segment}))
    curve_uf = _UnionFind(curve_points)
    degree_count: dict[RationalPoint, int] = defaultdict(int)
    for a, b in segment_keys:
        curve_uf.union(a, b)
        degree_count[a] += 1
        degree_count[b] += 1
    if any(value != 2 for value in degree_count.values()):
        raise ArithmeticError("the projective patchwork curve is not a disjoint union of cycles")
    component_groups: dict[object, list[int]] = defaultdict(list)
    for index, (a, _) in enumerate(segment_keys):
        component_groups[curve_uf.find(a)].append(index)
    components = tuple(sorted((tuple(indices) for indices in component_groups.values()), key=lambda x: x[0]))

    if degree % 2:
        return PatchworkResult(degree, segments, components, (), ())

    pre_uf = _UnionFind(vertices)
    for a, b in mesh_edges:
        if signs.reflected_bit(a) == signs.reflected_bit(b):
            pre_uf.union(a, b)
    region_uf = _UnionFind(vertices)
    for a, b in mesh_edges:
        if signs.reflected_bit(a) == signs.reflected_bit(b):
            region_uf.union(a, b)
    boundary = tuple(
        point
        for point in vertices
        if _on_reflected_boundary((Fraction(point[0]), Fraction(point[1])), degree)
    )
    for point in boundary:
        antipode = (-point[0], -point[1])
        if signs.reflected_bit(point) != signs.reflected_bit(antipode):
            raise ArithmeticError("even-degree antipodal signs must agree")
        region_uf.union(point, antipode)

    region_roots = sorted({region_uf.find(point) for point in vertices}, key=repr)
    region_index = {root: i for i, root in enumerate(region_roots)}
    exterior_candidates: set[int] = set()
    for point in boundary:
        antipode = (-point[0], -point[1])
        if pre_uf.find(point) == pre_uf.find(antipode):
            exterior_candidates.add(region_index[region_uf.find(point)])
    if len(exterior_candidates) != 1:
        raise ArithmeticError("the nonorientable exterior region is not uniquely identified")
    exterior = next(iter(exterior_candidates))

    curve_component_for_point = {
        point: component
        for component, indices in enumerate(components)
        for index in indices
        for point in segment_keys[index]
    }
    component_adjacency: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for a, b in segment_keys:
        triangle = unique_records[(a, b)]
        bits = tuple(signs.reflected_bit(point) for point in triangle)
        minority = next(i for i, bit in enumerate(bits) if bits.count(bit) == 1)
        majority = next(i for i, bit in enumerate(bits) if bits.count(bit) == 2)
        left = region_index[region_uf.find(triangle[minority])]
        right = region_index[region_uf.find(triangle[majority])]
        component_adjacency[curve_component_for_point[a]].add(tuple(sorted((left, right))))
    if any(len(pairs) != 1 for pairs in component_adjacency.values()):
        raise ArithmeticError("a T-curve component does not separate one fixed region pair")

    adjacency: dict[int, set[int]] = {i: set() for i in range(len(region_roots))}
    for component in range(len(components)):
        pairs = component_adjacency.get(component)
        if not pairs:
            raise ArithmeticError("a T-curve component has no adjacent complement regions")
        left, right = next(iter(pairs))
        if left == right or right in adjacency[left]:
            raise ArithmeticError("region adjacency must have one edge per oval")
        adjacency[left].add(right)
        adjacency[right].add(left)
    if len(components) != len(region_roots) - 1:
        raise ArithmeticError("oval components do not give a region tree")
    reached: set[int] = set()
    queue = deque([exterior])
    while queue:
        node = queue.popleft()
        if node in reached:
            continue
        reached.add(node)
        queue.extend(adjacency[node] - reached)
    if len(reached) != len(region_roots):
        raise ArithmeticError("the complement-region graph is disconnected")

    children = _canonical_children(adjacency, exterior)
    return PatchworkResult(
        degree,
        segments,
        components,
        children,
        _tree_signature(adjacency, exterior, None),
    )


def target_octic_region_tree() -> RootedTree:
    """Return the rooted-tree signature of ``<4 + 1<2 + 1<14>>>``."""
    fourteen = tuple(() for _ in range(14))
    nested = tuple(sorted(((), (), fourteen), key=repr))
    return tuple(sorted(((), (), (), (), nested), key=repr))


__all__ = [
    "LatticePoint",
    "PatchworkResult",
    "PatchworkSegment",
    "RationalPoint",
    "RootedTree",
    "SignDistribution",
    "Triangle",
    "Triangulation",
    "flip_neighbors",
    "lattice_points",
    "patchwork_curve",
    "staircase_triangulation",
    "target_octic_region_tree",
]
