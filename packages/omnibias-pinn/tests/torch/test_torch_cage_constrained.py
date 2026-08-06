# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard Dirichlet / Neumann / Robin / initial conditions on the torch cage.

The claim under test is that the conditions hold *identically*, not that they
are fitted well. So the discriminating checks are the ones that would pass for a
merely well-trained field and fail for anything less than structural exactness:
re-checking after randomising every parameter, and checking the corners where
two constrained axes meet.
"""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.constrained import (
    HardCondition,
    derivative_at,
    dirichlet,
    neumann,
    periodic,
    robin,
)
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.cage import ConstrainedExpressionField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField

DTYPE = torch.float64
EXACT = 1e-14
DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def _base(seed: int = 0, hidden: int = 16) -> OneLayerVectorField:
    torch.manual_seed(seed)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("t", "x"), domain=DOMAIN, time_axis="t"),
        components=ComponentSpec(("u",)),
        hidden=hidden,
        base="tanh",
        dtype=DTYPE,
    )


def _cage(conditions: list[HardCondition], **kw) -> ConstrainedExpressionField:  # noqa: ANN003
    return ConstrainedExpressionField(base=_base(kw.pop("seed", 0)), conditions=conditions, **kw)


def _val(field, coords):  # noqa: ANN001
    s = field(coords)
    return s.ops.value(s, "u")


def _der(field, coords, axis, order):  # noqa: ANN001
    s = field(coords)
    return s.ops.derivative(s, "u", axis=axis, order=order)


def _gap(got, want) -> float:  # noqa: ANN001
    """Largest absolute miss, detached so a live graph never leaks into a scalar."""
    return float((got - want).detach().abs().max())


def _pts(n: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g, dtype=DTYPE)


def _face(axis: int, value: float, n: int = 64, seed: int = 3) -> torch.Tensor:
    coords = _pts(n, seed)
    coords[:, axis] = value
    return coords


def _randomise(field: ConstrainedExpressionField, seed: int = 11) -> None:
    """Move every parameter far from its initialisation.

    Any exactness that survives this is structural. Any that does not was
    fitted, which is exactly the property this cage exists to replace.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in field.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.75)


# --------------------------------------------------------------------------- #
# One condition kind at a time.
# --------------------------------------------------------------------------- #
def test_dirichlet_holds_identically_on_both_faces() -> None:
    cage = _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), lambda c: torch.sin(c[:, 0])),
            HardCondition("u", 1, dirichlet(1.0), -0.4),
        ]
    )
    for _ in range(2):
        lo, hi = _face(1, 0.0), _face(1, 1.0)
        assert _gap(_val(cage, lo), torch.sin(lo[:, 0])) < EXACT
        assert _gap(_val(cage, hi), -0.4) < EXACT
        _randomise(cage)


def test_neumann_holds_identically_with_the_outward_normal_sign() -> None:
    cage = _cage(
        [
            HardCondition("u", 1, neumann(0.0, outward=-1.0), 0.9),
            HardCondition("u", 1, neumann(1.0, outward=1.0), lambda c: torch.cos(c[:, 0])),
        ]
    )
    for _ in range(2):
        lo, hi = _face(1, 0.0), _face(1, 1.0)
        # a lo-face outward normal points in -x
        assert _gap(-_der(cage, lo, 1, 1), 0.9) < EXACT
        assert _gap(_der(cage, hi, 1, 1), torch.cos(hi[:, 0])) < EXACT
        _randomise(cage)


def test_robin_holds_identically_on_both_faces() -> None:
    cage = _cage(
        [
            HardCondition("u", 1, robin(0.0, alpha=2.0, beta=0.5, outward=-1.0), 0.3),
            HardCondition("u", 1, robin(1.0, alpha=1.5, beta=0.25, outward=1.0), -0.7),
        ]
    )
    for _ in range(2):
        lo, hi = _face(1, 0.0), _face(1, 1.0)
        r_lo = 2.0 * _val(cage, lo) - 0.5 * _der(cage, lo, 1, 1)
        r_hi = 1.5 * _val(cage, hi) + 0.25 * _der(cage, hi, 1, 1)
        assert _gap(r_lo, 0.3) < EXACT
        assert _gap(r_hi, -0.7) < EXACT
        _randomise(cage)


def test_value_and_velocity_at_the_same_initial_time_both_hold() -> None:
    """The wave-equation pair: two conditions sharing one projection point."""
    cage = _cage(
        [
            HardCondition("u", 0, dirichlet(0.0), lambda c: torch.sin(3.0 * c[:, 1])),
            HardCondition("u", 0, derivative_at(0.0, 1), 0.25),
        ]
    )
    for _ in range(2):
        t0 = _face(0, 0.0)
        assert _gap(_val(cage, t0), torch.sin(3.0 * t0[:, 1])) < EXACT
        assert _gap(_der(cage, t0, 0, 1), 0.25) < EXACT
        _randomise(cage)


# --------------------------------------------------------------------------- #
# Several kinds, several axes, and the corners where they meet.
# --------------------------------------------------------------------------- #
def _mixed_cage() -> ConstrainedExpressionField:
    """Dirichlet + Neumann on x, an initial condition on t, compatible at (0,0)."""
    return _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), lambda c: torch.sin(c[:, 0])),
            HardCondition("u", 1, neumann(1.0), lambda c: torch.cos(c[:, 0])),
            HardCondition("u", 0, dirichlet(0.0), lambda c: c[:, 1] ** 2 / 2),
        ]
    )


def test_conditions_on_two_axes_hold_simultaneously() -> None:
    cage = _mixed_cage()
    for _ in range(3):
        lo, hi, t0 = _face(1, 0.0), _face(1, 1.0), _face(0, 0.0)
        assert _gap(_val(cage, lo), torch.sin(lo[:, 0])) < EXACT
        assert _gap(_der(cage, hi, 1, 1), torch.cos(hi[:, 0])) < EXACT
        assert _gap(_val(cage, t0), t0[:, 1] ** 2 / 2) < EXACT
        _randomise(cage)


def test_the_corners_where_the_two_axes_meet_are_exact() -> None:
    """The cross terms are what a naive per-axis ansatz gets wrong."""
    cage = _mixed_cage()
    corners = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    for _ in range(2):
        s = cage(corners)
        u = s.ops.value(s, "u").detach()
        # x = 0 corner: both the Dirichlet face and the initial slice claim it
        assert abs(float(u[0]) - math.sin(0.0)) < EXACT
        assert abs(float(u[0]) - 0.0) < EXACT
        # x = 1 corner: the initial condition claims the value there
        assert abs(float(u[1]) - 0.5) < EXACT
        _randomise(cage)


def test_exactness_is_unaffected_by_which_free_function_is_used() -> None:
    conditions = [
        HardCondition("u", 1, dirichlet(0.0), 0.6),
        HardCondition("u", 1, neumann(1.0), -0.2),
    ]
    for seed in (0, 1, 2):
        cage = ConstrainedExpressionField(base=_base(seed, hidden=24), conditions=conditions)
        lo, hi = _face(1, 0.0), _face(1, 1.0)
        assert _gap(_val(cage, lo), 0.6) < EXACT
        assert _gap(_der(cage, hi, 1, 1), -0.2) < EXACT


# --------------------------------------------------------------------------- #
# The derivatives the cage claims to compute in closed form.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_closed_form_derivatives_match_autodiff(axis: int, order: int) -> None:
    cage = _mixed_cage()
    coords = _pts(9, seed=5).requires_grad_(True)
    cur = _val(cage, coords)
    for _ in range(order):
        (grad,) = torch.autograd.grad(cur.sum(), coords, create_graph=True)
        cur = grad[:, axis]
    got = _der(cage, coords, axis, order)
    torch.testing.assert_close(got, cur, rtol=1e-9, atol=1e-9)


def test_mixed_partials_match_autodiff() -> None:
    cage = _mixed_cage()
    coords = _pts(9, seed=5).requires_grad_(True)
    cur = _val(cage, coords)
    for axis in (0, 1, 1):
        (grad,) = torch.autograd.grad(cur.sum(), coords, create_graph=True)
        cur = grad[:, axis]
    s = cage(coords)
    got = s.ops.mixed_partial(s, "u", (0, 1), (1, 2))
    torch.testing.assert_close(got, cur, rtol=1e-9, atol=1e-9)


def test_gradients_reach_the_base_parameters() -> None:
    cage = _mixed_cage()
    coords = _pts(32, seed=8)
    loss = (_der(cage, coords, 0, 1) - _der(cage, coords, 1, 2)).pow(2).mean()
    loss.backward()
    total = sum(
        float(p.grad.norm()) for p in cage.parameters() if p.grad is not None
    )
    assert total > 0.0


# --------------------------------------------------------------------------- #
# The two refusals. Both are live falsifiers: if the cage ever stopped
# detecting them, these tests fail rather than silently pass.
# --------------------------------------------------------------------------- #
_INCOMPATIBLE = [
    HardCondition("u", 1, dirichlet(0.0), 0.0),
    HardCondition("u", 0, dirichlet(0.0), lambda c: 1.0 + c[:, 1] ** 2),
]


def test_incompatible_corner_data_is_refused_at_construction() -> None:
    """Dirichlet u(t,0)=0 against an initial state that is 1 at x=0."""
    with pytest.raises(ValueError, match="mutually inconsistent"):
        _cage(list(_INCOMPATIBLE))


def test_the_refusal_names_the_two_conditions_that_clash() -> None:
    with pytest.raises(ValueError, match=r"'u\(0.0\)' on axis 0 vs 'u\(0.0\)' on axis 1"):
        _cage(list(_INCOMPATIBLE))


def test_incompatible_corner_data_shows_at_its_true_size() -> None:
    """The falsifier: the gate must report order one, not a small number."""
    bad = _cage(list(_INCOMPATIBLE), check_data=False)
    assert bad.compatibility_residual(_pts(32)) == pytest.approx(1.0, abs=1e-10)
    assert bad.compatibility_residual() == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError, match="mutually inconsistent"):
        bad.check_compatibility(_pts(32))


def test_corner_data_that_clashes_only_in_its_slopes_is_caught_too() -> None:
    """The values agree exactly; only ``d/dt`` disagrees at ``(0, 0)``.

    ``u(t, 0) = sin t`` forces ``u_t(0, 0) = 1``, which a zero initial velocity
    contradicts. Nothing about the values reveals it, so a gate that only
    compared values would pass this and leave the cage quietly wrong.
    """
    conditions = [
        HardCondition("u", 1, dirichlet(0.0), lambda c: torch.sin(c[:, 0])),
        HardCondition("u", 0, dirichlet(0.0), lambda c: torch.sin(c[:, 1])),
        HardCondition("u", 0, derivative_at(0.0, 1), 0.0),
    ]
    with pytest.raises(ValueError, match="mutually inconsistent"):
        _cage(list(conditions))
    bad = _cage(conditions, check_data=False)
    assert bad.compatibility_residual(_pts(32)) == pytest.approx(1.0, abs=1e-10)


def test_slope_compatible_corner_data_passes() -> None:
    """The same set, with the velocity the boundary data actually implies."""
    good = _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), lambda c: torch.sin(c[:, 0])),
            HardCondition("u", 1, neumann(1.0), 0.0),
            HardCondition(
                "u", 0, dirichlet(0.0), lambda c: torch.sin(math.pi * c[:, 1] / 2)
            ),
            HardCondition(
                "u",
                0,
                derivative_at(0.0, 1),
                lambda c: torch.cos(math.pi * c[:, 1] / 2) ** 2,
            ),
        ]
    )
    assert good.compatibility_residual(_pts(32)) < EXACT
    good.check_compatibility(_pts(32))
    t0 = _face(0, 0.0)
    assert _gap(_der(good, t0, 0, 1), torch.cos(math.pi * t0[:, 1] / 2) ** 2) < EXACT


def test_compatible_corner_data_passes_the_same_gate() -> None:
    good = _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 0, dirichlet(0.0), lambda c: c[:, 1] ** 2),
        ]
    )
    assert good.compatibility_residual(_pts(32)) < EXACT
    good.check_compatibility(_pts(32))


def test_a_singular_condition_set_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="singular support matrix"):
        _cage(
            [
                HardCondition("u", 1, dirichlet(0.0), 0.0),
                HardCondition("u", 1, dirichlet(0.0), 1.0),
            ]
        )


def test_an_empty_condition_set_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one condition"):
        _cage([])


def test_a_component_cannot_be_both_constrained_and_passthrough() -> None:
    with pytest.raises(ValueError, match="both constrained and passthrough"):
        _cage([HardCondition("u", 1, dirichlet(0.0), 0.0)], passthrough_names=("u",))


def test_missing_domain_bounds_are_refused_with_a_usable_message() -> None:
    torch.manual_seed(0)
    base = OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("t", "x"), time_axis="t"),
        components=ComponentSpec(("u",)),
        hidden=8,
        base="tanh",
        dtype=DTYPE,
    )
    with pytest.raises(ValueError, match="need per-axis"):
        ConstrainedExpressionField(
            base=base, conditions=[HardCondition("u", 1, dirichlet(0.0), 0.0)]
        )


# --------------------------------------------------------------------------- #
# Certification and cost.
# --------------------------------------------------------------------------- #
def test_every_constrained_axis_carries_a_verifiable_certificate() -> None:
    from omnibias.core.proof.certificate import verify_certificate_digest

    certs = _mixed_cage().support_certificates()
    assert set(certs) == {"u"}
    assert len(certs["u"]) == 2  # one per constrained axis
    for cert in certs["u"]:
        assert verify_certificate_digest(cert)
        assert float.fromhex(cert["payload"]["gram_lambda_min"]["lo"]) > 0.0


def test_conditions_sharing_a_face_share_one_base_evaluation() -> None:
    """Cost is one base evaluation per *distinct* projection point, not per condition."""
    cage = _cage(
        [
            HardCondition("u", 0, dirichlet(0.0), 0.0),
            HardCondition("u", 0, derivative_at(0.0, 1), 0.0),
        ]
    )
    coords = _pts(16)
    state = cage(coords)
    state.ops.value(state, "u")
    projected = state.extra["_cage_inner_state"].extra["_constrained_cache"]
    assert len(projected) == 1, "both conditions live on t = 0, so one projection"


# --------------------------------------------------------------------------- #
# Three axes at once. The recursion is the same one used for two, so what is
# actually under test is that nothing about it was two-dimensional by accident:
# the edges (two axes pinned) and the corner (all three) are where a missing
# cross term would first show.
# --------------------------------------------------------------------------- #
CUBE = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def _cube_base(seed: int = 0, hidden: int = 12) -> OneLayerVectorField:
    torch.manual_seed(seed)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("t", "x", "y"), domain=CUBE, time_axis="t"),
        components=ComponentSpec(("u",)),
        hidden=hidden,
        base="tanh",
        dtype=DTYPE,
    )


def _cube_conditions() -> list[HardCondition]:
    """Six conditions over three axes, of four different kinds.

    Every target is zero so the data is trivially compatible at all twelve
    corner pairs; what is being tested here is the *recursion*, and a nonzero
    target would only add a compatibility puzzle to the same algebra.
    """
    return [
        HardCondition("u", 0, dirichlet(0.0), 0.0),
        HardCondition("u", 0, derivative_at(0.0, 1), 0.0),
        HardCondition("u", 1, dirichlet(0.0), 0.0),
        HardCondition("u", 1, neumann(1.0), 0.0),
        HardCondition("u", 2, dirichlet(0.0), 0.0),
        HardCondition("u", 2, robin(1.0, alpha=1.0, beta=0.5), 0.0),
    ]


def _cube_cage(seed: int = 0) -> ConstrainedExpressionField:
    return ConstrainedExpressionField(base=_cube_base(seed), conditions=_cube_conditions())


def _cube_pts(n: int = 32, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g, dtype=DTYPE)


def test_six_conditions_over_three_axes_all_hold_identically() -> None:
    cage = _cube_cage()
    for _ in range(2):
        assert cage.condition_residual(_cube_pts()) < EXACT
        _randomise(cage)


def test_the_edges_where_two_constrained_axes_meet_are_exact() -> None:
    """A missing cross term survives on the faces and dies on the edges."""
    cage = _cube_cage()
    _randomise(cage)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pts = _cube_pts(24, seed=5)
        pts[:, a] = 0.0
        pts[:, b] = 0.0
        assert _gap(_val(cage, pts), torch.zeros(24, dtype=DTYPE)) < EXACT


def test_the_corner_where_all_three_meet_is_exact() -> None:
    cage = _cube_cage()
    _randomise(cage)
    corner = torch.zeros(1, 3, dtype=DTYPE)
    assert _gap(_val(cage, corner), torch.zeros(1, dtype=DTYPE)) < EXACT


def test_a_third_axis_multiplies_the_projection_cost_rather_than_adding_to_it() -> None:
    """Stated in the docstring, so it is measured rather than assumed."""
    cage = _cube_cage()
    state = cage(_cube_pts(8))
    state.ops.value(state, "u")
    projected = state.extra["_cage_inner_state"].extra["_constrained_cache"]
    # t has one projection point, x has two, y has two: 2 * 3 * 3 - 1 pinned states
    assert len(projected) == 2 * 3 * 3 - 1


def test_the_reported_cost_is_the_cost_actually_paid() -> None:
    """A published number that nothing checks is a number that drifts."""
    cage = _cube_cage()
    state = cage(_cube_pts(8))
    state.ops.value(state, "u")
    projected = state.extra["_cage_inner_state"].extra["_constrained_cache"]
    assert cage.projection_cost == len(projected) + 1  # + the unpinned batch


def test_the_corner_gate_covers_every_axis_pair_not_just_neighbours() -> None:
    """Axes 0 and 2 clash while both agree with axis 1, which sits between them.

    ``u(t, 0, y) = t`` and ``u(t, x, 0) = t + x`` agree with each other and with
    ``u(0, x, y) = 0`` pairwise except on the one pair that skips a step. A gate
    that only compared consecutive axes would report this set as fine.
    """
    conditions = [
        HardCondition("u", 0, dirichlet(0.0), 0.0),
        HardCondition("u", 1, dirichlet(0.0), lambda c: c[:, 0]),
        HardCondition("u", 2, dirichlet(0.0), lambda c: c[:, 0] + c[:, 1]),
    ]
    with pytest.raises(ValueError, match=r"on axis 0 vs .* on axis 2"):
        ConstrainedExpressionField(base=_cube_base(), conditions=conditions)


# --------------------------------------------------------------------------- #
# Periodicity: a *relative* constraint, tying two faces together without
# pinning either.
# --------------------------------------------------------------------------- #
def test_a_periodic_seam_closes_in_value_and_slope() -> None:
    cage = _cage(
        [
            HardCondition("u", 1, periodic(0.0, 1.0, order=0), 0.0),
            HardCondition("u", 1, periodic(0.0, 1.0, order=1), 0.0),
        ]
    )
    for _ in range(2):
        lo, hi = _face(1, 0.0), _face(1, 1.0)
        assert _gap(_val(cage, hi), _val(cage, lo)) < EXACT
        assert _gap(_der(cage, hi, 1, 1), _der(cage, lo, 1, 1)) < EXACT
        _randomise(cage)


def test_periodicity_composes_with_a_condition_on_another_axis() -> None:
    cage = _cage(
        [
            HardCondition("u", 1, periodic(0.0, 1.0, order=0), 0.0),
            HardCondition("u", 1, periodic(0.0, 1.0, order=1), 0.0),
            HardCondition(
                "u", 0, dirichlet(0.0), lambda c: torch.sin(2 * math.pi * c[:, 1])
            ),
        ]
    )
    _randomise(cage)
    assert cage.condition_residual(_pts(32)) < EXACT
    t0 = _face(0, 0.0)
    assert _gap(_val(cage, t0), torch.sin(2 * math.pi * t0[:, 1])) < EXACT


def test_initial_data_that_is_not_itself_periodic_is_refused() -> None:
    """``u(0, x) = x`` cannot meet a seam that demands ``u(t, 0) = u(t, 1)``."""
    with pytest.raises(ValueError, match="mutually inconsistent"):
        _cage(
            [
                HardCondition("u", 1, periodic(0.0, 1.0, order=0), 0.0),
                HardCondition("u", 1, periodic(0.0, 1.0, order=1), 0.0),
                HardCondition("u", 0, dirichlet(0.0), lambda c: c[:, 1]),
            ]
        )


def test_a_relative_constraint_refuses_a_callable_target() -> None:
    """It would have no face to be evaluated on, so guessing one is not an option."""
    with pytest.raises(ValueError, match="must be a constant"):
        _cage(
            [
                HardCondition(
                    "u", 1, periodic(0.0, 1.0), lambda c: torch.sin(c[:, 0])
                ),
            ]
        )
