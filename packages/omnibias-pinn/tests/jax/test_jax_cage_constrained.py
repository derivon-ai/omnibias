# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard Dirichlet / Neumann / Robin / initial conditions on the JAX cage.

The exactness claims themselves are pinned identically on both backends (and
compared against each other in ``tests/cross_backend``). What is tested *here*
is what only JAX can get wrong: surviving a pytree round trip, tracing under
``jit``, and differentiating through the cage with ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.constrained import (
    HardCondition,
    derivative_at,
    dirichlet,
    neumann,
    robin,
)
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.cage import (
    ConstrainedExpressionField,
    make_constrained_expression_field,
)
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField

EXACT = 1e-14
DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def _base(seed: int = 0, hidden: int = 16) -> OneLayerVectorField:
    rng = np.random.default_rng(seed)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("t", "x"), domain=DOMAIN, time_axis="t"),
        components=ComponentSpec(("u",)),
        spec=get_activation("tanh"),
        W=jnp.asarray(rng.normal(scale=0.8, size=(hidden, 2))),
        beta=jnp.asarray(rng.normal(scale=0.3, size=(hidden,))),
        c=jnp.asarray(rng.normal(scale=0.6, size=(1, hidden))),
        b=jnp.asarray(rng.normal(scale=0.2, size=(1,))),
        hidden=hidden,
    )


def _cage(conditions: list[HardCondition], seed: int = 0) -> ConstrainedExpressionField:
    return make_constrained_expression_field(base=_base(seed), conditions=conditions)


def _mixed_cage(seed: int = 0) -> ConstrainedExpressionField:
    return _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), lambda c: jnp.sin(c[:, 0])),
            HardCondition("u", 1, neumann(1.0), lambda c: jnp.cos(c[:, 0])),
            HardCondition("u", 0, dirichlet(0.0), lambda c: c[:, 1] ** 2 / 2),
        ],
        seed=seed,
    )


def _pts(n: int, seed: int = 7) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, 1.0, size=(n, 2)))


def _face(axis: int, value: float, n: int = 32, seed: int = 3) -> jnp.ndarray:
    return _pts(n, seed).at[:, axis].set(value)


def _val(field, coords):  # noqa: ANN001
    s = field(coords)
    return np.asarray(s.ops.value(s, "u"))


def _der(field, coords, axis, order):  # noqa: ANN001
    s = field(coords)
    return np.asarray(s.ops.derivative(s, "u", axis=axis, order=order))


def _perturb(field, seed: int = 11):  # noqa: ANN001
    """Move every leaf far from its initialisation, through the pytree API.

    Exactness that survives this is structural. Doing it through
    ``tree_flatten`` also proves the cage's leaves really are the base field's
    parameters and nothing else.
    """
    leaves, treedef = jax.tree_util.tree_flatten(field)
    rng = np.random.default_rng(seed)
    moved = [
        leaf + jnp.asarray(rng.normal(scale=0.75, size=leaf.shape)) for leaf in leaves
    ]
    return jax.tree_util.tree_unflatten(treedef, moved)


# --------------------------------------------------------------------------- #
# Exactness, and that it is structural rather than fitted.
# --------------------------------------------------------------------------- #
def test_every_condition_kind_holds_identically() -> None:
    cage = _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), 0.6),
            HardCondition("u", 1, robin(1.0, alpha=1.5, beta=0.25), -0.7),
        ]
    )
    for _ in range(2):
        lo, hi = _face(1, 0.0), _face(1, 1.0)
        assert np.abs(_val(cage, lo) - 0.6).max() < EXACT
        robin_value = 1.5 * _val(cage, hi) + 0.25 * _der(cage, hi, 1, 1)
        assert np.abs(robin_value + 0.7).max() < EXACT
        cage = _perturb(cage)


def test_conditions_on_two_axes_hold_simultaneously_and_at_the_corner() -> None:
    cage = _mixed_cage()
    for _ in range(3):
        lo, hi, t0 = _face(1, 0.0), _face(1, 1.0), _face(0, 0.0)
        assert np.abs(_val(cage, lo) - np.sin(np.asarray(lo[:, 0]))).max() < EXACT
        assert np.abs(_der(cage, hi, 1, 1) - np.cos(np.asarray(hi[:, 0]))).max() < EXACT
        assert np.abs(_val(cage, t0) - np.asarray(t0[:, 1]) ** 2 / 2).max() < EXACT
        corner = jnp.zeros((1, 2))
        assert abs(float(_val(cage, corner)[0])) < EXACT
        cage = _perturb(cage)


def test_a_wave_style_axis_carries_value_and_velocity_at_one_point() -> None:
    cage = _cage(
        [
            HardCondition("u", 0, dirichlet(0.0), lambda c: jnp.sin(3.0 * c[:, 1])),
            HardCondition("u", 0, derivative_at(0.0, 1), 0.25),
        ]
    )
    t0 = _face(0, 0.0)
    assert np.abs(_val(cage, t0) - np.sin(3.0 * np.asarray(t0[:, 1]))).max() < EXACT
    assert np.abs(_der(cage, t0, 0, 1) - 0.25).max() < EXACT


# --------------------------------------------------------------------------- #
# What only JAX can get wrong.
# --------------------------------------------------------------------------- #
def test_the_cage_survives_a_pytree_round_trip_unchanged() -> None:
    cage = _mixed_cage()
    leaves, treedef = jax.tree_util.tree_flatten(cage)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    coords = _pts(11)
    np.testing.assert_array_equal(_val(cage, coords), _val(rebuilt, coords))


def test_the_constraint_plan_rides_in_the_hashable_aux_data() -> None:
    """``jit`` caches on the treedef, so the plan has to be hashable, not a leaf.

    The leaves must be the base field's parameters and nothing else: a plan that
    leaked into them would be traced, and the switching coefficients are Python
    floats fixed by the condition set, not values to differentiate.
    """
    cage = _mixed_cage()
    leaves, treedef = jax.tree_util.tree_flatten(cage)
    assert hash(treedef) == hash(jax.tree_util.tree_flatten(cage)[1])
    assert treedef == jax.tree_util.tree_flatten(cage)[1]
    assert [tuple(x.shape) for x in leaves] == [
        tuple(x.shape) for x in jax.tree_util.tree_leaves(cage.base)
    ]


def test_the_cage_traces_under_jit_and_still_hits_its_conditions() -> None:
    cage = _mixed_cage()

    @jax.jit
    def value(field, coords):  # noqa: ANN001
        s = field(coords)
        return s.ops.value(s, "u")

    t0 = _face(0, 0.0)
    got = np.asarray(value(cage, t0))
    assert np.abs(got - np.asarray(t0[:, 1]) ** 2 / 2).max() < EXACT


def test_gradients_flow_to_the_base_parameters_through_the_cage() -> None:
    cage = _mixed_cage()
    coords = _pts(24, seed=8)

    def loss(field):  # noqa: ANN001
        s = field(coords)
        r = s.ops.derivative(s, "u", axis=0, order=1) - s.ops.derivative(
            s, "u", axis=1, order=2
        )
        return jnp.mean(r**2)

    grads = jax.grad(loss)(cage)
    total = sum(
        float(jnp.linalg.norm(g)) for g in jax.tree_util.tree_leaves(grads)
    )
    assert total > 0.0


def test_closed_form_derivatives_match_autodiff_through_the_cage() -> None:
    cage = _mixed_cage()
    coords = _pts(7, seed=5)

    def scalar(c, axis, order):  # noqa: ANN001
        def f(x):  # noqa: ANN001
            s = cage(x[None, :])
            return s.ops.value(s, "u")[0]

        for _ in range(order):
            f = (lambda prev=f, ax=axis: (lambda x: jax.grad(prev)(x)[ax]))()
        return jax.vmap(f)(c)

    for axis in (0, 1):
        for order in (1, 2, 3):
            want = np.asarray(scalar(coords, axis, order))
            np.testing.assert_allclose(
                _der(cage, coords, axis, order), want, rtol=1e-9, atol=1e-9
            )


# --------------------------------------------------------------------------- #
# The refusals, matching the torch twin.
# --------------------------------------------------------------------------- #
def test_incompatible_corner_data_is_detected_and_refused() -> None:
    bad = _cage(
        [
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 0, dirichlet(0.0), lambda c: 1.0 + c[:, 1] ** 2),
        ]
    )
    assert bad.compatibility_residual(_pts(32)) == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError, match="mutually inconsistent"):
        bad.check_compatibility(_pts(32))


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


def test_every_constrained_axis_carries_a_verifiable_certificate() -> None:
    from omnibias.core.proof.certificate import verify_certificate_digest

    certs = _mixed_cage().support_certificates()
    assert len(certs["u"]) == 2
    assert all(verify_certificate_digest(c) for c in certs["u"])
