# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for the hard-condition constrained-expression cage.

Both cages import the switching coefficients from the same pure-Python module
and walk the same recursion in the same axis order, so parity here is a check
that the two implementations of that walk agree -- not a calibration.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

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
from omnibias.pinn.jax.cage import make_constrained_expression_field
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JaxField
from omnibias.pinn.torch.cage import ConstrainedExpressionField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TorchField

RTOL = 1e-12
ATOL = 1e-12
DOMAIN = ((0.0, 1.0), (0.0, 2.0))


def _shared(seed: int, hidden: int = 12, n_pts: int = 33) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W": rng.normal(scale=0.7, size=(hidden, 2)),
        "beta": rng.normal(scale=0.3, size=(hidden,)),
        "c": rng.normal(scale=0.5, size=(1, hidden)),
        "b": rng.normal(scale=0.2, size=(1,)),
        "coords": np.stack(
            [rng.uniform(*DOMAIN[0], n_pts), rng.uniform(*DOMAIN[1], n_pts)], axis=1
        ),
    }


def _cspec() -> CoordinateSpec:
    return CoordinateSpec(("t", "x"), domain=DOMAIN, time_axis="t")


def _torch_cage(shared: dict[str, np.ndarray], conditions: list[HardCondition]):
    base = TorchField(
        coordinate_spec=_cspec(),
        components=ComponentSpec(("u",)),
        hidden=shared["W"].shape[0],
        base="tanh",
        dtype=torch.float64,
    )
    with torch.no_grad():
        base.W.weight.copy_(torch.from_numpy(shared["W"]))
        base.W.bias.copy_(torch.from_numpy(shared["beta"]))
        base.c.weight.copy_(torch.from_numpy(shared["c"]))
        base.c.bias.copy_(torch.from_numpy(shared["b"]))
    return ConstrainedExpressionField(base=base, conditions=conditions)


def _jax_cage(shared: dict[str, np.ndarray], conditions: list[HardCondition]):
    from omnibias.jax.activations import get_activation

    base = JaxField(
        coordinate_spec=_cspec(),
        components=ComponentSpec(("u",)),
        spec=get_activation("tanh"),
        W=jnp.asarray(shared["W"]),
        beta=jnp.asarray(shared["beta"]),
        c=jnp.asarray(shared["c"]),
        b=jnp.asarray(shared["b"]),
        hidden=shared["W"].shape[0],
    )
    return make_constrained_expression_field(base=base, conditions=conditions)


def _condition_sets() -> dict[str, tuple[list[HardCondition], list[HardCondition]]]:
    """``(torch conditions, jax conditions)`` -- identical but for the array type."""

    def pair(build):  # noqa: ANN001
        return build(torch), build(jnp)

    def mixed(xp):  # noqa: ANN001
        return [
            HardCondition("u", 1, dirichlet(0.0), lambda c: xp.sin(c[:, 0])),
            HardCondition("u", 1, neumann(2.0), lambda c: xp.cos(c[:, 0])),
            HardCondition("u", 0, dirichlet(0.0), lambda c: c[:, 1] ** 2 / 4),
        ]

    def robin_both(xp):  # noqa: ANN001, ARG001
        return [
            HardCondition("u", 1, robin(0.0, alpha=2.0, beta=0.5, outward=-1.0), 0.3),
            HardCondition("u", 1, robin(2.0, alpha=1.5, beta=0.25, outward=1.0), -0.7),
        ]

    def wave(xp):  # noqa: ANN001
        return [
            HardCondition("u", 0, dirichlet(0.0), lambda c: xp.sin(3.0 * c[:, 1])),
            HardCondition("u", 0, derivative_at(0.0, 1), 0.25),
        ]

    def seam(xp):  # noqa: ANN001
        # Periodic in x, with initial data that is itself periodic on [0, 2].
        return [
            HardCondition("u", 1, periodic(0.0, 2.0, order=0), 0.0),
            HardCondition("u", 1, periodic(0.0, 2.0, order=1), 0.0),
            HardCondition("u", 0, dirichlet(0.0), lambda c: xp.sin(math.pi * c[:, 1])),
        ]

    return {
        "mixed": pair(mixed),
        "robin": pair(robin_both),
        "seam": pair(seam),
        "wave": pair(wave),
    }


@pytest.mark.parametrize("case", sorted(_condition_sets()))
@pytest.mark.parametrize("seed", [0, 1])
def test_value_and_derivatives_match_across_backends(case: str, seed: int) -> None:
    t_conds, j_conds = _condition_sets()[case]
    shared = _shared(seed)
    tc = _torch_cage(shared, t_conds)
    jc = _jax_cage(shared, j_conds)
    t_coords = torch.from_numpy(shared["coords"])
    j_coords = jnp.asarray(shared["coords"])
    ts, js = tc(t_coords), jc(j_coords)

    got_t = ts.ops.value(ts, "u").detach().numpy()
    got_j = np.asarray(js.ops.value(js, "u"))
    np.testing.assert_allclose(got_t, got_j, rtol=RTOL, atol=ATOL)

    for axis in (0, 1):
        for order in (1, 2, 3):
            a = ts.ops.derivative(ts, "u", axis=axis, order=order).detach().numpy()
            b = np.asarray(js.ops.derivative(js, "u", axis=axis, order=order))
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

    a = ts.ops.mixed_partial(ts, "u", (0, 1), (1, 2)).detach().numpy()
    b = np.asarray(js.ops.mixed_partial(js, "u", (0, 1), (1, 2)))
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def test_the_shared_switching_coefficients_are_literally_the_same_object_source() -> None:
    """Both cages read the geometry from one pure-Python module, so it cannot drift."""
    t_conds, j_conds = _condition_sets()["mixed"]
    shared = _shared(0)
    tc = _torch_cage(shared, t_conds)
    jc = _jax_cage(shared, j_conds)
    t_switch = [s.constraints.switching for s in tc._plans["u"]]
    j_switch = [s.constraints.switching for s in jc.steps_for("u")]
    assert t_switch == j_switch


def test_the_compatibility_residual_agrees_across_backends() -> None:
    t_conds, j_conds = _condition_sets()["mixed"]
    shared = _shared(2)
    tc = _torch_cage(shared, t_conds)
    jc = _jax_cage(shared, j_conds)
    a = tc.compatibility_residual(torch.from_numpy(shared["coords"]))
    b = jc.compatibility_residual(jnp.asarray(shared["coords"]))
    assert a == pytest.approx(b, abs=1e-14)


def test_the_self_generated_sample_is_the_same_on_both_backends() -> None:
    """The default gate draws no random numbers, so a refusal is reproducible."""
    t_conds, j_conds = _condition_sets()["mixed"]
    shared = _shared(2)
    a = _torch_cage(shared, t_conds).compatibility_residual()
    b = _jax_cage(shared, j_conds).compatibility_residual()
    assert a == pytest.approx(b, abs=1e-14)


# --------------------------------------------------------------------------- #
# Three axes: the recursion depth Stage C opened up.
# --------------------------------------------------------------------------- #
CUBE = ((0.0, 1.0), (0.0, 2.0), (0.0, 1.5))


def _cube_shared(seed: int, hidden: int = 10, n_pts: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W": rng.normal(scale=0.7, size=(hidden, 3)),
        "beta": rng.normal(scale=0.3, size=(hidden,)),
        "c": rng.normal(scale=0.5, size=(1, hidden)),
        "b": rng.normal(scale=0.2, size=(1,)),
        "coords": np.stack(
            [rng.uniform(lo, hi, n_pts) for lo, hi in CUBE], axis=1
        ),
    }


def _cube_conditions() -> list[HardCondition]:
    return [
        HardCondition("u", 0, dirichlet(0.0), 0.0),
        HardCondition("u", 0, derivative_at(0.0, 1), 0.0),
        HardCondition("u", 1, dirichlet(0.0), 0.0),
        HardCondition("u", 1, neumann(2.0), 0.0),
        HardCondition("u", 2, robin(1.5, alpha=1.0, beta=0.4), 0.0),
    ]


def test_a_three_axis_cage_matches_across_backends() -> None:
    from omnibias.jax.activations import get_activation

    shared = _cube_shared(0)
    cspec = CoordinateSpec(("t", "x", "y"), domain=CUBE, time_axis="t")
    torch_base = TorchField(
        coordinate_spec=cspec,
        components=ComponentSpec(("u",)),
        hidden=shared["W"].shape[0],
        base="tanh",
        dtype=torch.float64,
    )
    with torch.no_grad():
        torch_base.W.weight.copy_(torch.from_numpy(shared["W"]))
        torch_base.W.bias.copy_(torch.from_numpy(shared["beta"]))
        torch_base.c.weight.copy_(torch.from_numpy(shared["c"]))
        torch_base.c.bias.copy_(torch.from_numpy(shared["b"]))
    tc = ConstrainedExpressionField(base=torch_base, conditions=_cube_conditions())
    jc = make_constrained_expression_field(
        base=JaxField(
            coordinate_spec=cspec,
            components=ComponentSpec(("u",)),
            spec=get_activation("tanh"),
            W=jnp.asarray(shared["W"]),
            beta=jnp.asarray(shared["beta"]),
            c=jnp.asarray(shared["c"]),
            b=jnp.asarray(shared["b"]),
            hidden=shared["W"].shape[0],
        ),
        conditions=_cube_conditions(),
    )
    ts = tc(torch.from_numpy(shared["coords"]))
    js = jc(jnp.asarray(shared["coords"]))
    np.testing.assert_allclose(
        ts.ops.value(ts, "u").detach().numpy(),
        np.asarray(js.ops.value(js, "u")),
        rtol=RTOL,
        atol=ATOL,
    )
    for axis in (0, 1, 2):
        for order in (1, 2):
            a = ts.ops.derivative(ts, "u", axis=axis, order=order).detach().numpy()
            b = np.asarray(js.ops.derivative(js, "u", axis=axis, order=order))
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
    a = ts.ops.mixed_partial(ts, "u", (0, 1, 2), (1, 1, 1)).detach().numpy()
    b = np.asarray(js.ops.mixed_partial(js, "u", (0, 1, 2), (1, 1, 1)))
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
