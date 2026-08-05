# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Domain, conditions, sampling, and System validation tests."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn.solver import (
    BoundaryCondition,
    CollocationSpec,
    Domain,
    Field,
    InitialCondition,
    System,
)
from omnibias.pinn.solver._core.sampling import (
    interior_points,
    spatial_boundary_points,
)


def test_domain_basics() -> None:
    dom = Domain(("x", "y", "t"), ((0.0, 1.0), (0.0, 2.0), (0.0, 0.5)))
    assert dom.ndim == 3
    assert dom.spatial_axes == ("x", "y")
    assert dom.time_axis == "t"
    assert dom.is_time_dependent
    assert dom.time_bounds() == (0.0, 0.5)
    assert dom.bound("y") == (0.0, 2.0)


def test_domain_requires_ordered_bounds() -> None:
    with pytest.raises(ValueError):
        Domain(("x",), ((1.0, 0.0),))


def test_boundary_condition_validation() -> None:
    BoundaryCondition("u", "dirichlet", 0.0)
    with pytest.raises(ValueError):
        BoundaryCondition("u", "bogus")
    with pytest.raises(ValueError):
        BoundaryCondition("u", "neumann")  # needs an axis


def test_initial_condition_order() -> None:
    InitialCondition("u", 1.0, order=1)
    with pytest.raises(ValueError):
        InitialCondition("u", 1.0, order=3)


def test_sampling_shapes() -> None:
    dom = Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    spec = CollocationSpec(n_interior=8, n_boundary=8, method="grid")
    pts = interior_points(dom, spec)
    assert pts.shape == (64, 2)
    # interior points strictly inside the box
    assert np.all(pts > 0.0) and np.all(pts < 1.0)
    bnd = spatial_boundary_points(dom, spec)
    assert bnd.shape[1] == 2
    # every boundary point sits on a face
    on_face = (
        np.isclose(bnd[:, 0], 0.0)
        | np.isclose(bnd[:, 0], 1.0)
        | np.isclose(bnd[:, 1], 0.0)
        | np.isclose(bnd[:, 1], 1.0)
    )
    assert np.all(on_face)


def test_periodic_axis_has_no_boundary_face() -> None:
    dom = Domain(("x", "t"), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))
    spec = CollocationSpec(n_interior=8, n_boundary=8)
    bnd = spatial_boundary_points(dom, spec)
    # x is periodic and t is the time axis -> no spatial boundary faces
    assert bnd.shape[0] == 0


def test_system_validation() -> None:
    dom = Domain(("x", "t"), ((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError):
        System(domain=dom, fields=(), residuals=(lambda s: s,))
    with pytest.raises(ValueError):
        System(domain=dom, fields=(Field("u"),), residuals=())
    # initial condition on a steady domain is rejected
    steady = Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError):
        System(
            domain=steady,
            fields=(Field("u"),),
            residuals=(lambda s: s,),
            initial=(InitialCondition("u", 0.0),),
        )
