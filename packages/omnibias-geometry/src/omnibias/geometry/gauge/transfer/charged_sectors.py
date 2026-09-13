# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact SU(2) Gauss-law dimensions at fixed edge representations.

All spins are nonnegative integer ``two_j = 2*j`` labels. Clebsch--Gordan
fusion counts invariant tensors at every vertex, including arbitrary
valence and external representation factors. This identifies a finite
electric representation sector; it supplies neither magnetic matrix
elements nor a Hamiltonian, energy bound, or continuum statement.

For an oriented finite graph, the fixed-edge Peter--Weyl block is
``tensor_e (V_j(e) tensor V_j(e)^*)``. Tensoring external representations
and collecting factors at vertices makes the gauge-invariant dimension
the product of local singlet multiplicities. SU(2) irreducibles are
self-dual, so orientation does not affect these dimensions. A self-loop
still supplies two representation factors at its vertex.

The general fusion computation is exact integer arithmetic. Its state
space can grow with the total twice-spin, rather than just its bit length;
it is not a polynomial-time claim for binary-encoded large spins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a nonnegative Python integer (not bool or float)")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _spin_labels(values: Sequence[int], label: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of twice-spin integers")
    return tuple(_nonnegative_integer(value, f"{label}[{i}]") for i, value in enumerate(values))


def _fuse(
    multiplicities: dict[int, int],
    spin: int,
    remaining_sum: int,
    remaining_max: int,
) -> dict[int, int]:
    """Fuse one irrep and retain channels that the remaining factors can cancel.

    Each old channel k contributes its multiplicity to every
    ``abs(k-spin), abs(k-spin)+2, ..., k+spin``. Range additions implement
    the same Clebsch--Gordan rule without a nested loop over all channels.
    Remaining tensor factors can only contain spins in
    ``[max(2*remaining_max-remaining_sum, remaining_sum%2), remaining_sum]``.
    Discarding channels outside this necessary range cannot remove a singlet.
    """
    lower_remaining = max(2 * remaining_max - remaining_sum, remaining_sum % 2)
    changes: dict[int, int] = {}
    for channel, count in multiplicities.items():
        lower = max(abs(channel - spin), lower_remaining)
        upper = min(channel + spin, remaining_sum)
        if (lower - channel - spin) % 2:
            lower += 1
        if lower <= upper:
            changes[lower] = changes.get(lower, 0) + count
            changes[upper + 2] = changes.get(upper + 2, 0) - count
    if not changes:
        return {}
    result: dict[int, int] = {}
    running = 0
    previous = min(changes)
    for endpoint in sorted(changes):
        if running:
            for channel in range(previous, endpoint, 2):
                result[channel] = running
        running += changes[endpoint]
        previous = endpoint
    return result


def su2_singlet_multiplicity(two_spins: Sequence[int]) -> int:
    r"""Return ``dim Inv_SU(2)(tensor_i V_(two_spins[i]/2))`` exactly.

    The empty tensor product has dimension one. Zero-spin factors are
    trivial and may be present anywhere. All entries are validated before
    any admissibility shortcut; floats, bools, negative spins, text and
    unordered collections are rejected rather than interpreted as labels.

    Fusion uses ``V_a tensor V_b = direct_sum_(k=|a-b| step 2)^(a+b) V_k``
    in twice-spin notation. Multiplicities of repeated channels are added,
    so a vertex of valence greater than three can have several singlets.
    """
    labels = tuple(sorted(spin for spin in _spin_labels(two_spins, "two_spins") if spin))
    count = len(labels)
    if count == 0:
        return 1
    total = sum(labels)
    if total % 2 or 2 * labels[-1] > total:
        return 0
    if count == 2:
        return int(labels[0] == labels[1])
    if count == 3:
        # After the parity and largest-spin checks, the triangle is admissible.
        return 1
    if count == 4:
        # Fuse two pairs: a singlet requires equal intermediate irreps.
        a, b, c, d = labels
        lower = max(abs(a - b), abs(c - d))
        upper = min(a + b, c + d)
        return max(0, (upper - lower) // 2 + 1)
    remaining_sum = total
    multiplicities = {0: 1}
    for index, spin in enumerate(labels):
        remaining_sum -= spin
        remaining_max = labels[-1] if index + 1 < count else 0
        multiplicities = _fuse(multiplicities, spin, remaining_sum, remaining_max)
        if not multiplicities:
            return 0
    return multiplicities.get(0, 0)


def charged_spin_network_dimension(
    n_vertices: int,
    edges: Sequence[tuple[int, int]],
    two_edge_spins: Sequence[int],
    external_two_spins: Mapping[int, Sequence[int]] | None = None,
) -> int:
    r"""Exact invariant dimension on a labelled finite graph with external charges.

    Vertices are ``0,...,n_vertices-1``; each edge is an oriented endpoint
    pair with one corresponding twice-spin. Distinct parallel links are
    legal. A self-loop contributes twice at its endpoint. External charges
    are *labelled representation factors*, not fixed color vectors, and
    a vertex may carry zero, one, or several such factors.

    This computes the product of local singlet multiplicities for the
    fixed edge labels. It does not sum over spins, divide by representation
    dimensions, identify permutations of identical edges/charges, or
    construct plaquette couplings. Empty graphs and isolated neutral
    vertices contribute the one-dimensional trivial representation.
    """
    n_vertices = _nonnegative_integer(n_vertices, "n_vertices")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes, bytearray)):
        raise TypeError("edges must be a sequence of endpoint pairs")
    edge_spins = _spin_labels(two_edge_spins, "two_edge_spins")
    if len(edges) != len(edge_spins):
        raise ValueError("one twice-spin is required for each edge")
    local: dict[int, list[int]] = {}

    def vertex_id(value: object, label: str) -> int:
        vertex = _nonnegative_integer(value, label)
        if vertex >= n_vertices:
            raise ValueError(f"{label} must be smaller than n_vertices")
        return vertex

    for index, (edge, spin) in enumerate(zip(edges, edge_spins, strict=True)):
        if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes, bytearray)):
            raise TypeError(f"edges[{index}] must be an endpoint pair")
        if len(edge) != 2:
            raise ValueError(f"edges[{index}] must have exactly two endpoints")
        source = vertex_id(edge[0], f"edges[{index}][0]")
        target = vertex_id(edge[1], f"edges[{index}][1]")
        if spin:
            local.setdefault(source, []).append(spin)
            local.setdefault(target, []).append(spin)
    if external_two_spins is not None:
        if not isinstance(external_two_spins, Mapping):
            raise TypeError("external_two_spins must map vertices to charge sequences")
        for vertex, charges in external_two_spins.items():
            vertex = vertex_id(vertex, "external charge vertex")
            labels = _spin_labels(charges, f"external_two_spins[{vertex}]")
            local.setdefault(vertex, []).extend(spin for spin in labels if spin)

    # Validation is complete before a zero local dimension can return early.
    dimension = 1
    cache: dict[tuple[int, ...], int] = {}
    for spins in local.values():
        signature = tuple(sorted(spins))
        if signature not in cache:
            cache[signature] = su2_singlet_multiplicity(signature)
        dimension *= cache[signature]
        if dimension == 0:
            return 0
    return dimension


__all__ = ["charged_spin_network_dimension", "su2_singlet_multiplicity"]
