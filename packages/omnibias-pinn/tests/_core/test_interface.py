# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Interface geometry and interface-point sampling (backend-free)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.interface import (
    Interface,
    InterfaceSpec,
    interface_points,
    split_by_interface,
)

BOX_2D = ((-1.0, 1.0), (0.0, 2.0))


def _oblique(label: str = "oblique") -> Interface:
    return Interface(normal=(1.0, 2.0), offset=1.5, label=label)


# ------------------------------------------------------------ geometry ---


def test_the_normal_is_stored_unit_length_without_moving_the_plane() -> None:
    """Rescaling ``(n, c)`` together leaves the point set alone."""
    scaled = Interface(normal=(3.0, 6.0), offset=4.5)
    assert float(np.linalg.norm(scaled.unit_normal)) == pytest.approx(1.0)
    x = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(
        scaled.signed_distance(x), _oblique().signed_distance(x), atol=1e-15
    )


def test_signed_distance_is_a_distance() -> None:
    iface = _oblique()
    x = np.random.default_rng(1).normal(size=(30, 2))
    d = iface.signed_distance(x)
    foot = iface.project(x)
    np.testing.assert_allclose(np.linalg.norm(x - foot, axis=1), np.abs(d), atol=1e-14)
    assert np.all(iface.contains(foot))


def test_orientation_is_the_sign_of_the_normal() -> None:
    iface = Interface.from_axis(ndim=2, axis=0, value=0.5)
    x = np.array([[0.9, 0.0], [0.1, 0.0]])
    assert iface.side(x).tolist() == [1, -1]
    assert iface.flip().side(x).tolist() == [-1, 1]
    np.testing.assert_allclose(
        iface.flip().signed_distance(x), -iface.signed_distance(x), atol=1e-15
    )


def test_the_tangent_basis_spans_the_interface() -> None:
    iface = Interface(normal=(1.0, -2.0, 0.5), offset=0.3)
    basis = iface.tangent_basis()
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(basis.T @ iface.unit_normal, np.zeros(2), atol=1e-14)


def test_a_one_dimensional_interface_is_a_point_with_no_tangents() -> None:
    iface = Interface.from_axis(ndim=1, axis=0, value=0.25)
    assert iface.tangent_basis().shape == (1, 0)
    pts = interface_points(iface, ((0.0, 1.0),), n_points=4)
    np.testing.assert_allclose(pts, np.full((4, 1), 0.25))


def test_from_spec_names_the_axis() -> None:
    spec = CoordinateSpec(axes=("t", "x", "y"), time_axis="t")
    iface = Interface.from_spec(spec, axis="y", value=0.5)
    assert iface.unit_normal.tolist() == [0.0, 0.0, 1.0]
    assert iface.label == "y=0.5"


def test_from_split_reads_a_partition_gate_zero_set() -> None:
    """The gate is 1/2 exactly on the returned hyperplane."""
    dirs = np.array([[1.0, 0.0], [0.6, -0.8]])
    thresh = np.array([0.25, -0.1])
    iface = Interface.from_split(dirs, thresh, row=1)
    x = interface_points(iface, BOX_2D, n_points=16, seed=3)
    gate = 1.0 / (1.0 + np.exp(-30.0 * (x @ dirs[1] - thresh[1])))
    np.testing.assert_allclose(gate, np.full(16, 0.5), atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "err", "match"),
    [
        (dict(normal=(0.0, 0.0), offset=1.0), ValueError, "non-zero"),
        (dict(normal=(), offset=1.0), ValueError, "non-empty"),
        (dict(normal=(1.0, np.inf), offset=0.0), ValueError, "finite"),
    ],
)
def test_a_degenerate_interface_is_rejected(kwargs, err, match) -> None:
    with pytest.raises(err, match=match):
        Interface(**kwargs)


def test_a_mismatched_point_set_is_rejected() -> None:
    iface = _oblique()
    with pytest.raises(ValueError, match="interface is 2-D"):
        iface.signed_distance(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="must be 2-D"):
        iface.signed_distance(np.zeros(4))


def test_from_axis_checks_its_arguments() -> None:
    with pytest.raises(IndexError, match="axis 2"):
        Interface.from_axis(ndim=2, axis=2, value=0.0)
    with pytest.raises(ValueError, match="ndim"):
        Interface.from_axis(ndim=0, axis=0, value=0.0)


# ------------------------------------------------------------ sampling ---


@pytest.mark.parametrize("method", ["random", "grid"])
def test_sampled_points_lie_on_the_interface_to_round_off(method: str) -> None:
    """The whole point: on the seam, not near it."""
    iface = _oblique()
    x = interface_points(iface, BOX_2D, n_points=64, method=method, seed=5)
    assert x.shape == (64, 2)
    assert float(np.abs(iface.signed_distance(x)).max()) < 1e-14
    lo, hi = np.asarray(BOX_2D)[:, 0], np.asarray(BOX_2D)[:, 1]
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)


def test_grid_sampling_is_deterministic_and_random_sampling_is_seeded() -> None:
    iface = _oblique()
    a = interface_points(iface, BOX_2D, n_points=12, method="grid")
    b = interface_points(iface, BOX_2D, n_points=12, method="grid", seed=99)
    np.testing.assert_array_equal(a, b)  # seed is irrelevant to the lattice
    r1 = interface_points(iface, BOX_2D, n_points=12, seed=1)
    r2 = interface_points(iface, BOX_2D, n_points=12, seed=1)
    r3 = interface_points(iface, BOX_2D, n_points=12, seed=2)
    np.testing.assert_array_equal(r1, r2)
    assert not np.allclose(r1, r3)


def test_shrink_keeps_points_clear_of_the_box_faces() -> None:
    iface = Interface.from_axis(ndim=2, axis=0, value=0.0)
    x = interface_points(iface, BOX_2D, n_points=64, seed=7, shrink=0.25)
    assert float(x[:, 1].min()) >= 0.5 - 1e-9
    assert float(x[:, 1].max()) <= 1.5 + 1e-9


def test_a_three_dimensional_interface_samples_a_plane() -> None:
    iface = Interface(normal=(1.0, 1.0, 1.0), offset=0.0)
    box = ((-1.0, 1.0),) * 3
    x = interface_points(iface, box, n_points=50, seed=2)
    assert x.shape == (50, 3)
    assert float(np.abs(iface.signed_distance(x)).max()) < 1e-14
    # A plane, not a line: both tangent directions are actually explored.
    spread = (x @ iface.tangent_basis()).std(axis=0)
    assert float(spread.min()) > 0.1


def test_an_interface_that_misses_the_box_is_an_error_not_silence() -> None:
    far = Interface.from_axis(ndim=2, axis=0, value=50.0)
    with pytest.raises(ValueError, match="barely meets the box"):
        interface_points(far, BOX_2D, n_points=8)
    with pytest.raises(ValueError, match="too few lattice points"):
        interface_points(far, BOX_2D, n_points=8, method="grid")
    with pytest.raises(ValueError, match="outside the box"):
        interface_points(
            Interface.from_axis(ndim=1, axis=0, value=9.0), ((0.0, 1.0),), n_points=2
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(n_points=0), "n_points"),
        (dict(n_points=4, shrink=0.5), "shrink"),
        (dict(n_points=4, method="spiral"), "unknown method"),
    ],
)
def test_sampling_arguments_are_validated(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        interface_points(_oblique(), BOX_2D, **kwargs)


def test_a_malformed_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        interface_points(_oblique(), ((1.0, 1.0), (0.0, 2.0)), n_points=4)
    with pytest.raises(ValueError, match="interface is 2-D"):
        interface_points(_oblique(), ((0.0, 1.0),) * 3, n_points=4)


# -------------------------------------------------------------- splits ---


def test_splitting_routes_every_point_to_exactly_one_side() -> None:
    iface = _oblique()
    x = np.random.default_rng(11).uniform(-1.0, 2.0, size=(200, 2))
    plus, minus = split_by_interface(iface, x)
    assert plus.shape[0] + minus.shape[0] == 200
    assert np.all(iface.signed_distance(plus) >= 0.0)
    assert np.all(iface.signed_distance(minus) < 0.0)
    flipped_plus, flipped_minus = split_by_interface(iface.flip(), x)
    assert flipped_plus.shape[0] + flipped_minus.shape[0] == 200
    assert flipped_minus.shape[0] <= plus.shape[0]


def test_an_exact_tie_is_owned_by_the_plus_side() -> None:
    """Documented tie-break, checked where the arithmetic is exact.

    An oblique seam cannot be tested this way and should not be: ``n.x - c``
    for a projected point is a round-off-sized number of either sign, so which
    patch owns an on-seam point is arbitrary there. Route collocation with
    points drawn off the seam, and put the seam itself in the interface
    residual, where both sides are evaluated anyway.
    """
    iface = Interface.from_axis(ndim=2, axis=0, value=0.5)
    plus, minus = split_by_interface(iface, np.array([[0.5, 0.3], [0.5, -0.2]]))
    assert plus.shape[0] == 2 and minus.shape[0] == 0


# ---------------------------------------------------------------- spec ---


def test_the_spec_carries_material_data_and_defaults_to_matched_media() -> None:
    spec = InterfaceSpec(interface=_oblique("seam"))
    assert spec.conductivity == (1.0, 1.0)
    assert spec.weights == (1.0, 1.0)
    assert spec.label == "seam"
    assert InterfaceSpec(_oblique(), conductivity=(2, 5)).conductivity == (2.0, 5.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(conductivity=(1.0,)), "k_plus"),
        (dict(weights=(1.0, 1.0, 1.0)), "value"),
        (dict(weights=(1.0, -1.0)), "non-negative"),
    ],
)
def test_a_malformed_spec_is_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        InterfaceSpec(interface=_oblique(), **kwargs)
