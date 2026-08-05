# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Taxonomy / classification tests for the canonical problems."""

from __future__ import annotations

import math

from omnibias.pinn.solver import (
    Arity,
    Domain,
    Linearity,
    PDEType,
    ProblemKind,
    advection_diffusion,
    array_namespace,
    burgers,
    heat,
    poisson,
    reaction_diffusion,
    wave,
)


def _sin_source(c):
    xp = array_namespace(c)
    return -2.0 * math.pi ** 2 * xp.sin(math.pi * c[:, 0])


def test_poisson_classification() -> None:
    dom = Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    sys = poisson(dom, source=_sin_source)
    c = sys.classify()
    assert c.pde_type is PDEType.ELLIPTIC
    assert c.linearity is Linearity.LINEAR
    assert c.kind is ProblemKind.BOUNDARY_VALUE
    assert c.arity is Arity.SCALAR
    assert str(c) == "elliptic / linear / boundary_value / scalar"


def test_heat_and_wave_are_time_dependent() -> None:
    dom = Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)))
    h = heat(dom, diffusivity=0.1, initial=1.0)
    assert h.classify().kind is ProblemKind.INITIAL_VALUE
    assert h.classify().pde_type is PDEType.PARABOLIC
    w = wave(dom, speed=1.0)
    assert w.classify().pde_type is PDEType.HYPERBOLIC
    assert len(w.initial) == 2  # u and u_t


def test_burgers_is_nonlinear_scalar() -> None:
    dom = Domain(("x", "t"), ((0.0, 2.0), (0.0, 0.5)), periodic=(True, False))
    b = burgers(dom, viscosity=0.05, initial=0.0)
    c = b.classify()
    assert c.linearity is Linearity.NONLINEAR
    assert c.arity is Arity.SCALAR


def test_coupled_systems_have_system_arity() -> None:
    dom = Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)), periodic=(True, False))
    rd = reaction_diffusion(dom, reaction=lambda u, v: (u * v, -u * v))
    assert rd.classify().arity is Arity.SYSTEM
    assert rd.classify().linearity is Linearity.NONLINEAR
    assert rd.component_names() == ("u", "v")

    ad = advection_diffusion(dom, velocity=1.0, coupling=0.3)
    assert ad.classify().arity is Arity.SYSTEM
    assert ad.classify().linearity is Linearity.LINEAR
