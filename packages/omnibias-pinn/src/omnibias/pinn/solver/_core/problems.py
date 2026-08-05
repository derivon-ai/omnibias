# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Canonical, neutral PDE problems as :class:`System` builders.

Six problems span the taxonomy cross-product (type x linearity x kind x arity):

* :func:`poisson`             -- elliptic,   linear,    BVP,  scalar
* :func:`heat`                -- parabolic,  linear,    IVP,  scalar
* :func:`wave`                -- hyperbolic, linear,    IVP,  scalar
* :func:`burgers`             -- parabolic,  nonlinear, IVP,  scalar
* :func:`reaction_diffusion`  -- parabolic,  nonlinear, IVP,  system (coupled)
* :func:`advection_diffusion` -- parabolic,  linear,    IVP,  system (coupled)

Residuals are backend-agnostic closures over ``state.ops.*`` (the closed-form
operator surface). Sign conventions are documented per builder.

Every physical coefficient accepts ``float | Unknown``: pass a float for a
forward solve, or an :class:`~omnibias.pinn.solver._core.unknowns.Unknown` to
recover it from observations with :func:`omnibias.pinn.solver.torch.solve_inverse`.
The residual closures read coefficients through
:func:`~omnibias.pinn.solver._core.unknowns.coefficient`, so a single ``System``
serves both modes and the inverse path costs a forward solve nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from omnibias.pinn.solver._core.conditions import (
    BoundaryCondition,
    InitialCondition,
    ValueLike,
)
from omnibias.pinn.solver._core.domain import Domain
from omnibias.pinn.solver._core.system import Field, System
from omnibias.pinn.solver._core.taxonomy import Linearity, PDEType
from omnibias.pinn.solver._core.unknowns import (
    Coefficient,
    Unknown,
    coefficient,
    collect_unknowns,
)


def _eval(value: ValueLike, coords: Any) -> Any:
    """Evaluate a float / callable condition target on ``coords``."""
    if callable(value):
        return value(coords)
    return value


def _advection(state: Any, name: str, velocity: Sequence[Coefficient]) -> Any:
    """``sum_i v_i du/dx_i`` over spatial axes."""
    spatial = state.coordinate_spec.spatial_axes
    total = None
    for axis, vi in zip(spatial, velocity, strict=True):
        d = state.ops.derivative(state, name, axis=axis, order=1)
        term = coefficient(vi) * d
        total = term if total is None else total + term
    return total


def poisson(
    domain: Domain,
    *,
    source: ValueLike = 0.0,
    boundary: ValueLike = 0.0,
    component: str = "u",
) -> System:
    r"""Poisson problem ``Delta u = source`` with Dirichlet boundary data.

    Residual: ``laplacian(u) - source``. To solve ``-Delta u = f`` pass
    ``source = lambda c: -f(c)``.
    """

    def residual(state: Any) -> Any:
        return state.ops.laplacian(state, component) - _eval(source, state.coords)

    return System(
        domain=domain,
        fields=(Field(component),),
        residuals=(residual,),
        boundary=(BoundaryCondition(component, "dirichlet", boundary),),
        pde_type=PDEType.ELLIPTIC,
        linearity=Linearity.LINEAR,
        name="poisson",
    )


def heat(
    domain: Domain,
    *,
    diffusivity: Coefficient = 1.0,
    initial: ValueLike = 0.0,
    boundary: ValueLike = 0.0,
    source: ValueLike = 0.0,
    component: str = "u",
) -> System:
    r"""Heat equation ``u_t = diffusivity * Delta u + source``.

    Residual: ``u_t - diffusivity * laplacian(u) - source``.
    """
    ta = domain.time_axis
    if ta is None:
        raise ValueError("heat requires a time axis in the domain")

    def residual(state: Any) -> Any:
        u_t = state.ops.derivative(state, component, axis=ta, order=1)
        lap = state.ops.laplacian(state, component)
        return u_t - coefficient(diffusivity) * lap - _eval(source, state.coords)

    return System(
        domain=domain,
        fields=(Field(component),),
        residuals=(residual,),
        boundary=(BoundaryCondition(component, "dirichlet", boundary),),
        initial=(InitialCondition(component, initial, order=0),),
        pde_type=PDEType.PARABOLIC,
        linearity=Linearity.LINEAR,
        name="heat",
        unknowns=collect_unknowns(diffusivity),
    )


def wave(
    domain: Domain,
    *,
    speed: Coefficient = 1.0,
    initial: ValueLike = 0.0,
    initial_velocity: ValueLike = 0.0,
    boundary: ValueLike = 0.0,
    component: str = "u",
) -> System:
    r"""Wave equation ``u_tt = speed**2 * Delta u``.

    Residual: ``u_tt - speed**2 * laplacian(u)``. Two initial conditions:
    ``u(x, 0)`` and ``u_t(x, 0)``.

    Recovering ``speed`` recovers only its *magnitude*: the residual sees
    ``speed ** 2``, so the sign is not identifiable. Use
    ``Unknown(..., transform="positive")``.
    """
    ta = domain.time_axis
    if ta is None:
        raise ValueError("wave requires a time axis in the domain")

    def residual(state: Any) -> Any:
        u_tt = state.ops.derivative(state, component, axis=ta, order=2)
        lap = state.ops.laplacian(state, component)
        return u_tt - (coefficient(speed) ** 2) * lap

    return System(
        domain=domain,
        fields=(Field(component),),
        residuals=(residual,),
        boundary=(BoundaryCondition(component, "dirichlet", boundary),),
        initial=(
            InitialCondition(component, initial, order=0),
            InitialCondition(component, initial_velocity, order=1),
        ),
        pde_type=PDEType.HYPERBOLIC,
        linearity=Linearity.LINEAR,
        name="wave",
        unknowns=collect_unknowns(speed),
    )


def burgers(
    domain: Domain,
    *,
    viscosity: Coefficient = 0.05,
    initial: ValueLike = 0.0,
    component: str = "u",
) -> System:
    r"""Viscous Burgers ``u_t + u u_x = viscosity * u_xx`` (1 spatial axis).

    Residual: ``u_t + u * u_x - viscosity * laplacian(u)``. Typically posed on a
    periodic spatial domain.
    """
    ta = domain.time_axis
    if ta is None:
        raise ValueError("burgers requires a time axis in the domain")
    if domain.n_spatial != 1:
        raise ValueError("burgers builder targets exactly one spatial axis")
    (sx,) = domain.spatial_axes

    def residual(state: Any) -> Any:
        u = state.ops.value(state, component)
        u_t = state.ops.derivative(state, component, axis=ta, order=1)
        u_x = state.ops.derivative(state, component, axis=sx, order=1)
        u_xx = state.ops.laplacian(state, component)
        return u_t + u * u_x - coefficient(viscosity) * u_xx

    return System(
        domain=domain,
        fields=(Field(component),),
        residuals=(residual,),
        initial=(InitialCondition(component, initial, order=0),),
        pde_type=PDEType.PARABOLIC,
        linearity=Linearity.NONLINEAR,
        name="burgers",
        unknowns=collect_unknowns(viscosity),
    )


def reaction_diffusion(
    domain: Domain,
    *,
    diffusivities: tuple[Coefficient, Coefficient] = (1.0, 0.5),
    reaction: Callable[[Any, Any], tuple[Any, Any]],
    initial: tuple[ValueLike, ValueLike] = (0.0, 0.0),
    components: tuple[str, str] = ("u", "v"),
) -> System:
    r"""Two-species reaction-diffusion (coupled, nonlinear).

    ``u_t = Du * Delta u + R_u(u, v)`` and ``v_t = Dv * Delta v + R_v(u, v)``.
    ``reaction(u, v)`` returns the pair ``(R_u, R_v)`` given the component
    *values* (tensors). Residuals: ``u_t - Du * lap(u) - R_u`` and likewise for
    ``v``.
    """
    ta = domain.time_axis
    if ta is None:
        raise ValueError("reaction_diffusion requires a time axis")
    cu, cv = components
    du, dv = diffusivities

    def residual_u(state: Any) -> Any:
        u = state.ops.value(state, cu)
        v = state.ops.value(state, cv)
        ru, _ = reaction(u, v)
        u_t = state.ops.derivative(state, cu, axis=ta, order=1)
        return u_t - coefficient(du) * state.ops.laplacian(state, cu) - ru

    def residual_v(state: Any) -> Any:
        u = state.ops.value(state, cu)
        v = state.ops.value(state, cv)
        _, rv = reaction(u, v)
        v_t = state.ops.derivative(state, cv, axis=ta, order=1)
        return v_t - coefficient(dv) * state.ops.laplacian(state, cv) - rv

    return System(
        domain=domain,
        fields=(Field(cu), Field(cv)),
        residuals=(residual_u, residual_v),
        initial=(
            InitialCondition(cu, initial[0], order=0),
            InitialCondition(cv, initial[1], order=0),
        ),
        pde_type=PDEType.PARABOLIC,
        linearity=Linearity.NONLINEAR,
        name="reaction_diffusion",
        unknowns=collect_unknowns(diffusivities),
    )


def advection_diffusion(
    domain: Domain,
    *,
    velocity: Coefficient | Sequence[Coefficient] = 1.0,
    diffusivities: tuple[Coefficient, Coefficient] = (0.1, 0.1),
    coupling: Coefficient = 0.0,
    initial: tuple[ValueLike, ValueLike] = (0.0, 0.0),
    boundary: tuple[ValueLike, ValueLike] = (0.0, 0.0),
    components: tuple[str, str] = ("u", "v"),
) -> System:
    r"""Two-field advection-diffusion (coupled, linear).

    ``u_t + a.grad u = Du * Delta u + k (v - u)`` and
    ``v_t + a.grad v = Dv * Delta v + k (u - v)`` with a linear exchange
    coupling ``k``. Residual per field: transport + diffusion + coupling.

    ``velocity`` is a scalar (broadcast over spatial axes) or one value per
    spatial axis; either form may mix floats and
    :class:`~omnibias.pinn.solver._core.unknowns.Unknown` entries.
    """
    ta = domain.time_axis
    if ta is None:
        raise ValueError("advection_diffusion requires a time axis")
    cu, cv = components
    du, dv = diffusivities
    vel: Sequence[Coefficient] = (
        (velocity,) * domain.n_spatial
        if isinstance(velocity, int | float | Unknown)
        else tuple(velocity)
    )

    def residual_u(state: Any) -> Any:
        u = state.ops.value(state, cu)
        v = state.ops.value(state, cv)
        u_t = state.ops.derivative(state, cu, axis=ta, order=1)
        adv = _advection(state, cu, vel)
        return (
            u_t
            + adv
            - coefficient(du) * state.ops.laplacian(state, cu)
            - coefficient(coupling) * (v - u)
        )

    def residual_v(state: Any) -> Any:
        u = state.ops.value(state, cu)
        v = state.ops.value(state, cv)
        v_t = state.ops.derivative(state, cv, axis=ta, order=1)
        adv = _advection(state, cv, vel)
        return (
            v_t
            + adv
            - coefficient(dv) * state.ops.laplacian(state, cv)
            - coefficient(coupling) * (u - v)
        )

    return System(
        domain=domain,
        fields=(Field(cu), Field(cv)),
        residuals=(residual_u, residual_v),
        boundary=(
            BoundaryCondition(cu, "dirichlet", boundary[0]),
            BoundaryCondition(cv, "dirichlet", boundary[1]),
        ),
        initial=(
            InitialCondition(cu, initial[0], order=0),
            InitialCondition(cv, initial[1], order=0),
        ),
        pde_type=PDEType.PARABOLIC,
        linearity=Linearity.LINEAR,
        name="advection_diffusion",
        unknowns=collect_unknowns(vel, diffusivities, coupling),
    )


__all__ = [
    "advection_diffusion",
    "burgers",
    "heat",
    "poisson",
    "reaction_diffusion",
    "wave",
]
