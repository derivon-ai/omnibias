# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stage 2a: ``periodic_boundary`` opt-in on the six problem builders.

Default ``False`` must reproduce today's boundary tuples (kinds, counts,
components, axes). ``True`` on a periodic spatial domain appends periodic
seam conditions after any existing BCs so ``absorbed_boundary`` indices of
the original conditions stay stable.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver.torch.assemble import (  # noqa: E402
    all_rows,
    default_interior,
)


def _bc_fingerprint(system: pde.System) -> list[tuple[str, str, str | None]]:
    return [(bc.component, bc.kind, bc.axis) for bc in system.boundary]


def _rows(field, system: pde.System, spec: pde.CollocationSpec):
    return all_rows(field, system, default_interior(field, system, spec), spec)


def _reaction(u, v):  # noqa: ANN001, ANN202
    return (u * v, -u * v)


def _all_builders(*, periodic_boundary: bool) -> dict[str, pde.System]:
    """Build every canonical problem on a domain with one periodic spatial axis."""
    box = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))
    timed = pde.Domain(
        ("x", "t"), ((0.0, 2.0 * math.pi), (0.0, 0.5)), periodic=(True, False)
    )
    return {
        "poisson": pde.poisson(box, source=0.0, boundary=0.0, periodic_boundary=periodic_boundary),
        "heat": pde.heat(
            timed, diffusivity=0.1, initial=0.0, boundary=0.0, periodic_boundary=periodic_boundary
        ),
        "wave": pde.wave(
            timed, speed=1.0, initial=0.0, boundary=0.0, periodic_boundary=periodic_boundary
        ),
        "burgers": pde.burgers(
            timed, viscosity=0.05, initial=0.0, periodic_boundary=periodic_boundary
        ),
        "reaction_diffusion": pde.reaction_diffusion(
            timed, reaction=_reaction, periodic_boundary=periodic_boundary
        ),
        "advection_diffusion": pde.advection_diffusion(
            timed, velocity=1.0, periodic_boundary=periodic_boundary
        ),
    }


#: Today's boundary fingerprints (``periodic_boundary=False`` / omitted).
_TODAY = {
    "poisson": [("u", "dirichlet", None)],
    "heat": [("u", "dirichlet", None)],
    "wave": [("u", "dirichlet", None)],
    "burgers": [],
    "reaction_diffusion": [],
    "advection_diffusion": [("u", "dirichlet", None), ("v", "dirichlet", None)],
}


def test_default_matches_todays_boundary_tuples() -> None:
    """Opt-out (default) is today's builders: same BC count, kinds, axes."""
    omitted = _all_builders(periodic_boundary=False)
    # Explicit False and the keyword-omitted path must agree with the freeze.
    for name, expected in _TODAY.items():
        assert _bc_fingerprint(omitted[name]) == expected, name

    box = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))
    timed = pde.Domain(
        ("x", "t"), ((0.0, 2.0 * math.pi), (0.0, 0.5)), periodic=(True, False)
    )
    assert _bc_fingerprint(pde.poisson(box, source=0.0)) == _TODAY["poisson"]
    assert _bc_fingerprint(pde.heat(timed, diffusivity=0.1, initial=0.0)) == _TODAY["heat"]
    assert _bc_fingerprint(pde.wave(timed, speed=1.0)) == _TODAY["wave"]
    assert _bc_fingerprint(pde.burgers(timed, viscosity=0.05)) == _TODAY["burgers"]
    assert (
        _bc_fingerprint(pde.reaction_diffusion(timed, reaction=_reaction))
        == _TODAY["reaction_diffusion"]
    )
    assert (
        _bc_fingerprint(pde.advection_diffusion(timed, velocity=1.0))
        == _TODAY["advection_diffusion"]
    )


def test_default_off_path_matches_todays_residual_arrays_bit_for_bit() -> None:
    """Collocation residual vectors agree with the pre-flag builders."""
    torch.set_default_dtype(torch.float64)
    timed = pde.Domain(
        ("x", "t"), ((0.0, 2.0 * math.pi), (0.0, 0.25)), periodic=(True, False)
    )
    today = pde.burgers(timed, viscosity=0.05, initial=0.0)
    flagged = pde.burgers(timed, viscosity=0.05, initial=0.0, periodic_boundary=False)
    field = pt.build_field(today, hidden=16, seed=0)
    spec = pde.CollocationSpec(n_interior=12, n_boundary=8)
    assert torch.equal(_rows(field, today, spec), _rows(field, flagged, spec))


def test_periodic_boundary_true_appends_periodic_bcs() -> None:
    systems = _all_builders(periodic_boundary=True)
    for name, sys in systems.items():
        kinds = [bc.kind for bc in sys.boundary]
        assert "periodic" in kinds, name
        # Appended, never prepended: every original BC stays at its old index.
        today = _TODAY[name]
        assert _bc_fingerprint(sys)[: len(today)] == today, name
        for bc in sys.boundary[len(today) :]:
            assert bc.kind == "periodic"
            assert bc.axis == "x"  # the only periodic spatial axis


def test_periodic_boundary_true_emits_one_bc_per_component_per_axis() -> None:
    timed = pde.Domain(
        ("x", "t"), ((0.0, 2.0 * math.pi), (0.0, 0.5)), periodic=(True, False)
    )
    rd = pde.reaction_diffusion(timed, reaction=_reaction, periodic_boundary=True)
    assert _bc_fingerprint(rd) == [
        ("u", "periodic", "x"),
        ("v", "periodic", "x"),
    ]


def test_periodic_boundary_ignores_a_periodic_time_axis() -> None:
    """Even if ``t`` is marked periodic, seam BCs are spatial only."""
    dom = pde.Domain(
        ("x", "t"), ((0.0, 1.0), (0.0, 0.5)), periodic=(True, True), time_axis="t"
    )
    sys = pde.burgers(dom, viscosity=0.1, periodic_boundary=True)
    assert _bc_fingerprint(sys) == [("u", "periodic", "x")]


def test_periodic_boundary_true_on_nonperiodic_domain_emits_nothing() -> None:
    dom = pde.Domain(("x",), ((0.0, 1.0),))
    sys = pde.poisson(dom, source=0.0, periodic_boundary=True)
    assert _bc_fingerprint(sys) == [("u", "dirichlet", None)]


def test_absorbed_boundary_indices_stay_stable_when_periodic_is_appended() -> None:
    """Appending keeps the Dirichlet at index 0 so absorption indices hold."""
    from omnibias.pinn.solver._core.hard import plan_hard_conditions

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)), periodic=(False, True))
    base = pde.poisson(dom, source=0.0, boundary=0.0, periodic_boundary=False)
    with_seam = pde.poisson(dom, source=0.0, boundary=0.0, periodic_boundary=True)
    plan_base = plan_hard_conditions(base)
    plan_seam = plan_hard_conditions(with_seam)
    assert 0 in plan_base.absorbed_boundary
    assert 0 in plan_seam.absorbed_boundary  # Dirichlet still at index 0
    assert any(bc.kind == "periodic" for bc in with_seam.boundary)


def test_optional_periodic_boundary_replaces_manual_replace_for_burgers() -> None:
    """Tests that used ``replace(..., boundary=(periodic,))`` can opt in instead."""
    from dataclasses import replace

    timed = pde.Domain(
        ("x", "t"), ((0.0, 2.0 * math.pi), (0.0, 0.5)), periodic=(True, False)
    )
    manual = replace(
        pde.burgers(timed, viscosity=0.05, initial=0.0),
        boundary=(pde.BoundaryCondition(component="u", kind="periodic", axis="x"),),
    )
    opted = pde.burgers(timed, viscosity=0.05, initial=0.0, periodic_boundary=True)
    assert _bc_fingerprint(manual) == _bc_fingerprint(opted)
    # Residual arrays stay bit-identical under the same field.
    torch.set_default_dtype(torch.float64)
    field = pt.build_field(opted, hidden=16, seed=1)
    spec = pde.CollocationSpec(n_interior=10, n_boundary=6)
    assert torch.equal(_rows(field, manual, spec), _rows(field, opted, spec))


def test_docs_style_builders_without_the_flag_are_unchanged() -> None:
    """Call sites that never pass the flag keep today's BC surface."""
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)))
    assert len(pde.heat(dom, diffusivity=0.1, initial=0.0).boundary) == 1
    assert pde.burgers(
        pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)), periodic=(True, False)),
        viscosity=0.05,
    ).boundary == ()
