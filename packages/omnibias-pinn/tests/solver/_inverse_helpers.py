# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared problem fixtures for the inverse-problem tests.

``test_inverse.py`` owns the cheap, decidable mechanics and
``test_inverse_recovery.py`` owns the heavier accuracy gate. Both describe the
*same* problems, so the builders live here rather than in either test module --
a test module importing another test module gets imported twice (once by the
collector, once by the importer), which runs its module-level body twice and
breaks under ``--import-mode=importlib``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402

N_OBS = 48

#: The budget every *accuracy* claim is measured at, here and in the recovery gate.
BUDGET = {
    "hidden": 16,
    "collocation": pde.CollocationSpec(n_interior=10, n_boundary=6),
    "iters": 40,
    "adam_iters": 150,
    "condition_weight": 20.0,
    "data_weight": 20.0,
}

#: Enough to exercise a code path, nowhere near enough to converge. Used by the
#: tests whose claim is structural (a constraint holds, a field is populated), so
#: that this module stays a fast suite; converging each of them costs ~75s.
FAST = {**BUDGET, "hidden": 8, "iters": 8, "adam_iters": 60}

TRUE_D, TRUE_C, TRUE_NU = 0.35, 1.4, 0.08

TWO_TRUTH = (0.4, 0.15)


def sin_initial(c):
    xp = pde.array_namespace(c)
    return xp.sin(math.pi * c[:, 0])


def heat_system(coefficient):
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.2)), time_axis="t")
    return pde.heat(dom, diffusivity=coefficient, initial=sin_initial, boundary=0.0)


def wave_system(coefficient):
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)), time_axis="t")
    return pde.wave(dom, speed=coefficient, initial=sin_initial, boundary=0.0)


def burgers_system(coefficient):
    def initial(c):
        xp = pde.array_namespace(c)
        return -xp.sin(math.pi * c[:, 0])

    dom = pde.Domain(("x", "t"), ((-1.0, 1.0), (0.0, 0.4)), time_axis="t")
    return pde.burgers(dom, viscosity=coefficient, initial=initial)


def obs_points(system: pde.System, *, seed: int = 4242) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(lo, hi, N_OBS) for (lo, hi) in system.domain.bounds], axis=-1
    )


def heat_exact(coords: np.ndarray) -> np.ndarray:
    return np.exp(-TRUE_D * math.pi**2 * coords[:, 1]) * np.sin(math.pi * coords[:, 0])


def wave_exact(coords: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * coords[:, 0]) * np.cos(math.pi * TRUE_C * coords[:, 1])


def guess(truth: float) -> pde.Unknown:
    """A 3x-wrong starting point, held positive by the transform."""
    return pde.Unknown("theta", initial=3.0 * truth, transform="positive")


def recover(build, truth, values, coords, *, seed: int, **kw) -> pt.InverseSolution:
    return pt.solve_inverse(
        build(guess(truth)),
        [pde.Observations("u", coords, values)],
        seed=seed,
        **{**BUDGET, **kw},
    )


def two_unknown_system(diffusivities):
    """Two uncoupled diffusions, so each observed component sees one coefficient."""
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.2)), time_axis="t")
    return pde.advection_diffusion(
        dom,
        velocity=0.0,
        diffusivities=diffusivities,
        coupling=0.0,
        initial=(sin_initial, sin_initial),
    )


def two_unknown_observations(coords: np.ndarray) -> list[pde.Observations]:
    return [
        pde.Observations(
            name,
            coords,
            np.exp(-d * math.pi**2 * coords[:, 1]) * np.sin(math.pi * coords[:, 0]),
        )
        for name, d in zip(("u", "v"), TWO_TRUTH, strict=True)
    ]


def two_unknown_guesses() -> tuple[pde.Unknown, pde.Unknown]:
    return (
        pde.Unknown("Du", initial=1.0, transform="positive"),
        pde.Unknown("Dv", initial=1.0, transform="positive"),
    )
