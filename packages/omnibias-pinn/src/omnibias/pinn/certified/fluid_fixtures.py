# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Generated periodic toy-flow fixtures for proof-carrying fluid dynamics.

Pure-``numpy`` *analytic* incompressible flows on the periodic torus
``[0, L)^d``.  They are the ground truth for the residual-only Navier--Stokes
certificates in :mod:`omnibias.pinn.certified.fluid`.  Two families are shipped:

* :func:`taylor_green_vortex` -- the exact 2-D decaying Taylor--Green solution
  ``u = e^{-2\nu t}(\sin x\cos y,\,-\cos x\sin y)`` with the matching pressure.
  This is the **laminar correctness baseline**: the momentum residual,
  divergence and pressure-Poisson residual are all machine zero.
* :func:`kolmogorov_flow` -- the exact steady low-mode forced shear flow
  ``u = (A\sin(k y),\,0)`` balanced by ``f = \nu A k^2(\sin(k y),\,0)``.  This is
  the entry point into the **forced / chaotic-facing** regime (the laminar base
  state of Kolmogorov turbulence); the certificate over it stays residual-only.

Each builder returns a :class:`PeriodicFlowSample` carrying the sampled fields
*and* a JSON-native ``descriptor`` that fully regenerates them.  The descriptor
is what travels inside a certificate, so an independent verifier (for example
:mod:`omnibias.symbolic.fluid`) can rebuild the field from scratch *without
importing this module* -- a genuine second source.

Nothing here approximates high-Reynolds turbulence, claims perfect weather, or
tracks chaos pointwise over long horizons; these are exact analytic model flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FLUID_FIXTURE_DESCRIPTOR_VERSION = "omnibias-periodic-flow-fixture-1"


def _periodic_axis(n: int, length: float) -> np.ndarray:
    """The ``n`` collocation nodes of one periodic axis of length ``length``."""
    return length * np.arange(n, dtype=float) / n


@dataclass(frozen=True)
class PeriodicFlowSample:
    """A sampled periodic incompressible flow plus its regeneration descriptor.

    Attributes
    ----------
    velocity, pressure, velocity_t, forcing
        Component-first sampled arrays (``velocity`` has shape ``(dim, *grid)``).
    viscosity, density
        Fluid parameters; the kinematic viscosity is ``viscosity / density``.
    lengths
        Per-axis periodic domain lengths.
    descriptor
        JSON-native specification (name + parameters) sufficient to regenerate
        every field analytically.  Carried verbatim inside certificates.
    """

    velocity: np.ndarray
    pressure: np.ndarray
    velocity_t: np.ndarray
    forcing: np.ndarray
    viscosity: float
    density: float
    lengths: tuple[float, ...]
    descriptor: dict[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.velocity.shape[0])

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.velocity.shape[1:])


def taylor_green_vortex(
    n: int,
    *,
    viscosity: float,
    density: float = 1.0,
    time: float = 0.0,
    amplitude: float = 1.0,
) -> PeriodicFlowSample:
    r"""The exact 2-D Taylor--Green vortex sampled on an ``n x n`` periodic grid.

    With kinematic viscosity ``\nu = viscosity/density`` and decay factor
    ``F = e^{-2\nu t}`` the velocity, pressure and time derivative are

    .. math::

        u &= A F\,(\sin x\cos y,\; -\cos x\sin y), \\
        p &= \tfrac14\rho A^2 F^2(\cos 2x + \cos 2y), \\
        u_t &= -2\nu\,u,

    with zero forcing.  Substituting into the periodic incompressible momentum
    balance ``\rho(u_t + (u\cdot\nabla)u) + \nabla p - \mu\Delta u`` gives an
    exact zero (the advection cancels the pressure gradient and ``\rho u_t``
    cancels the viscous term), so this is a genuine unsteady solution.
    """
    if n < 4:
        raise ValueError(f"taylor_green_vortex needs n >= 4, got {n}")
    if viscosity < 0.0:
        raise ValueError("viscosity must be non-negative")
    if density <= 0.0:
        raise ValueError("density must be positive")
    length = 2.0 * np.pi
    axis = _periodic_axis(n, length)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    nu = viscosity / density
    decay = float(np.exp(-2.0 * nu * time))
    velocity = amplitude * decay * np.stack([
        np.sin(x) * np.cos(y),
        -np.cos(x) * np.sin(y),
    ])
    pressure = 0.25 * density * (amplitude * decay) ** 2 * (np.cos(2.0 * x) + np.cos(2.0 * y))
    velocity_t = -2.0 * nu * velocity
    forcing = np.zeros_like(velocity)
    descriptor: dict[str, Any] = {
        "descriptor_version": FLUID_FIXTURE_DESCRIPTOR_VERSION,
        "name": "taylor_green_vortex",
        "dimension": 2,
        "n": int(n),
        "lengths": [length, length],
        "viscosity": float(viscosity),
        "density": float(density),
        "time": float(time),
        "amplitude": float(amplitude),
        "exact_solution": True,
        "forced": False,
    }
    return PeriodicFlowSample(
        velocity=velocity,
        pressure=pressure,
        velocity_t=velocity_t,
        forcing=forcing,
        viscosity=float(viscosity),
        density=float(density),
        lengths=(length, length),
        descriptor=descriptor,
    )


def kolmogorov_flow(
    n: int,
    *,
    viscosity: float,
    density: float = 1.0,
    wavenumber: int = 1,
    amplitude: float = 1.0,
) -> PeriodicFlowSample:
    r"""The exact steady forced Kolmogorov shear flow on an ``n x n`` torus.

    The base flow ``u = (A\sin(k y),\,0)`` with ``p = 0`` is steady because the
    advection ``(u\cdot\nabla)u`` and pressure gradient both vanish.  The viscous
    term is balanced by the monochromatic body force

    .. math::

        f = \mu A k^2\,(\sin(k y),\; 0) = -\mu\,\Delta u,

    so the momentum residual and divergence are machine zero.  This is the
    laminar Kolmogorov base state -- the starting point of the forced 2-D flow
    that becomes chaotic at high Reynolds number, **not** a turbulent solution.
    """
    if n < 4:
        raise ValueError(f"kolmogorov_flow needs n >= 4, got {n}")
    if wavenumber < 1:
        raise ValueError("wavenumber must be a positive integer")
    if 2 * wavenumber >= n:
        raise ValueError(
            f"grid n={n} cannot resolve wavenumber k={wavenumber} (need n > 2k)"
        )
    if viscosity < 0.0:
        raise ValueError("viscosity must be non-negative")
    if density <= 0.0:
        raise ValueError("density must be positive")
    length = 2.0 * np.pi
    axis = _periodic_axis(n, length)
    _x, y = np.meshgrid(axis, axis, indexing="ij")
    k = int(wavenumber)
    shear = amplitude * np.sin(k * y)
    velocity = np.stack([shear, np.zeros_like(shear)])
    pressure = np.zeros_like(shear)
    velocity_t = np.zeros_like(velocity)
    forcing = np.stack([
        viscosity * amplitude * (k * k) * np.sin(k * y),
        np.zeros_like(shear),
    ])
    descriptor: dict[str, Any] = {
        "descriptor_version": FLUID_FIXTURE_DESCRIPTOR_VERSION,
        "name": "kolmogorov_flow",
        "dimension": 2,
        "n": int(n),
        "lengths": [length, length],
        "viscosity": float(viscosity),
        "density": float(density),
        "wavenumber": int(k),
        "amplitude": float(amplitude),
        "exact_solution": True,
        "forced": True,
    }
    return PeriodicFlowSample(
        velocity=velocity,
        pressure=pressure,
        velocity_t=velocity_t,
        forcing=forcing,
        viscosity=float(viscosity),
        density=float(density),
        lengths=(length, length),
        descriptor=descriptor,
    )


def beltrami_abc_flow(
    n: int,
    *,
    viscosity: float,
    density: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    wavenumber: int = 1,
    time: float = 0.0,
) -> PeriodicFlowSample:
    r"""The exact decaying 3-D Arnold--Beltrami--Childress (ABC) flow on a torus.

    The base field on ``[0, 2\pi)^3``

    .. math::

        u_0 = \bigl(A\sin kz + C\cos ky,\;\; B\sin kx + A\cos kz,\;\;
                    C\sin ky + B\cos kx\bigr)

    is divergence free and a **Beltrami eigenfunction** of the curl,
    ``\nabla\times u_0 = k\,u_0`` (hence ``\Delta u_0 = -k^2 u_0``).  With kinematic
    viscosity ``\nu = viscosity/density`` and decay factor ``F = e^{-\nu k^2 t}``,
    the time-dependent field ``u = F\,u_0`` is an **exact unsteady Navier--Stokes
    solution**: the advection ``(u\cdot\nabla)u = \nabla(|u|^2/2)`` is a pure
    gradient (because ``u\times(\nabla\times u) = k\,u\times u = 0``), so the
    matching pressure ``p = -\tfrac12\rho|u|^2`` cancels it exactly, ``\rho u_t``
    cancels the viscous term ``-\mu\Delta u``, and the forcing is zero.

    This is the 3-D companion of :func:`taylor_green_vortex`: a genuine
    finite-time exact solution of the full 3-D incompressible system, **not** a
    turbulence or regularity claim.
    """
    if n < 4:
        raise ValueError(f"beltrami_abc_flow needs n >= 4, got {n}")
    k = int(wavenumber)
    if k < 1:
        raise ValueError("wavenumber must be a positive integer")
    if 2 * k >= n:
        raise ValueError(f"grid n={n} cannot resolve wavenumber k={k} (need n > 2k)")
    if viscosity < 0.0:
        raise ValueError("viscosity must be non-negative")
    if density <= 0.0:
        raise ValueError("density must be positive")
    length = 2.0 * np.pi
    axis = _periodic_axis(n, length)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    nu = viscosity / density
    decay = float(np.exp(-nu * (k * k) * time))
    base = np.stack([
        a * np.sin(k * z) + c * np.cos(k * y),
        b * np.sin(k * x) + a * np.cos(k * z),
        c * np.sin(k * y) + b * np.cos(k * x),
    ])
    velocity = decay * base
    pressure = -0.5 * density * np.sum(velocity * velocity, axis=0)
    velocity_t = -nu * (k * k) * velocity
    forcing = np.zeros_like(velocity)
    descriptor: dict[str, Any] = {
        "descriptor_version": FLUID_FIXTURE_DESCRIPTOR_VERSION,
        "name": "beltrami_abc_flow",
        "dimension": 3,
        "n": int(n),
        "lengths": [length, length, length],
        "viscosity": float(viscosity),
        "density": float(density),
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "wavenumber": int(k),
        "time": float(time),
        "exact_solution": True,
        "forced": False,
    }
    return PeriodicFlowSample(
        velocity=velocity,
        pressure=pressure,
        velocity_t=velocity_t,
        forcing=forcing,
        viscosity=float(viscosity),
        density=float(density),
        lengths=(length, length, length),
        descriptor=descriptor,
    )


_FIXTURE_BUILDERS = {
    "taylor_green_vortex": taylor_green_vortex,
    "kolmogorov_flow": kolmogorov_flow,
    "beltrami_abc_flow": beltrami_abc_flow,
}


def available_fixtures() -> tuple[str, ...]:
    """The registered fixture names usable with :func:`regenerate_periodic_flow`."""
    return tuple(sorted(_FIXTURE_BUILDERS))


def regenerate_periodic_flow(descriptor: dict[str, Any]) -> PeriodicFlowSample:
    """Rebuild a :class:`PeriodicFlowSample` from a fixture ``descriptor``.

    The inverse of the ``descriptor`` carried by each builder.  Used by demos and
    tests to round-trip a certificate's fixture back into sampled fields.
    """
    name = descriptor.get("name")
    if name == "taylor_green_vortex":
        return taylor_green_vortex(
            int(descriptor["n"]),
            viscosity=float(descriptor["viscosity"]),
            density=float(descriptor.get("density", 1.0)),
            time=float(descriptor.get("time", 0.0)),
            amplitude=float(descriptor.get("amplitude", 1.0)),
        )
    if name == "kolmogorov_flow":
        return kolmogorov_flow(
            int(descriptor["n"]),
            viscosity=float(descriptor["viscosity"]),
            density=float(descriptor.get("density", 1.0)),
            wavenumber=int(descriptor.get("wavenumber", 1)),
            amplitude=float(descriptor.get("amplitude", 1.0)),
        )
    if name == "beltrami_abc_flow":
        return beltrami_abc_flow(
            int(descriptor["n"]),
            viscosity=float(descriptor["viscosity"]),
            density=float(descriptor.get("density", 1.0)),
            a=float(descriptor.get("a", 1.0)),
            b=float(descriptor.get("b", 1.0)),
            c=float(descriptor.get("c", 1.0)),
            wavenumber=int(descriptor.get("wavenumber", 1)),
            time=float(descriptor.get("time", 0.0)),
        )
    raise ValueError(f"unknown fluid fixture descriptor name: {name!r}")


def save_periodic_flow_sample(sample: PeriodicFlowSample, out_dir: str) -> dict[str, str]:
    """Write a sample's arrays + descriptor to ``out_dir`` (created if needed).

    Returns the paths written.  This is a *runtime* convenience for caching
    generated data (for example under a ``--scratch-dir``); nothing here is a
    tracked artifact and no path is hardcoded.
    """
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    arrays_path = os.path.join(out_dir, "periodic_flow_sample.npz")
    descriptor_path = os.path.join(out_dir, "periodic_flow_descriptor.json")
    np.savez(
        arrays_path,
        velocity=sample.velocity,
        pressure=sample.pressure,
        velocity_t=sample.velocity_t,
        forcing=sample.forcing,
    )
    with open(descriptor_path, "w", encoding="utf-8") as handle:
        json.dump(sample.descriptor, handle, indent=2, sort_keys=True)
    return {"arrays": arrays_path, "descriptor": descriptor_path}


__all__ = [
    "FLUID_FIXTURE_DESCRIPTOR_VERSION",
    "PeriodicFlowSample",
    "available_fixtures",
    "beltrami_abc_flow",
    "kolmogorov_flow",
    "regenerate_periodic_flow",
    "save_periodic_flow_sample",
    "taylor_green_vortex",
]
