# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form integral transforms of the activation dictionary (JAX).

Mirror of ``packages/omnibias-torch/tests/test_transforms.py``: coverage
against the shared table, correctness against Gauss-Legendre quadrature of the
defining integral, and differentiability. The JAX-specific additions are the
tracing checks (``jit`` / ``grad`` / ``vmap``) and the Fermi-Dirac gradient,
which works here and does not on torch.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.core.transforms import (
    TRANSFORM_NAMES,
    TransformName,
    find_exclusion,
    find_identity,
    identities,
)
from omnibias.jax.activations import get_activation, list_activations
from omnibias.jax.transforms import (
    COS_TRANSFORMS,
    COSH_TRANSFORMS,
    EXP_TRANSFORMS,
    GAUSSIAN_TRANSFORMS,
    RELU_TRANSFORMS,
    SECH_TRANSFORMS,
    SIGMOID_TRANSFORMS,
    SIN_TRANSFORMS,
    SINH_TRANSFORMS,
    TANH_TRANSFORMS,
    FourierTransform,
    LaplaceTransform,
    MellinTransform,
    TransformBlock,
    fermi_dirac_mellin,
    fourier_transform,
    has_transform,
    laplace_transform,
    mellin_transform,
    region_of_convergence,
)

jax.config.update("jax_enable_x64", True)

_TRANSFORM_FN = {
    "laplace": laplace_transform,
    "fourier": fourier_transform,
    "mellin": mellin_transform,
}

REGISTERED_PAIRS = [
    (identity.activation, identity.transform)
    for name in TRANSFORM_NAMES
    for identity in identities(name)
]

_BASE_BUNDLES = {
    id(bundle)
    for bundle in (
        COSH_TRANSFORMS,
        COS_TRANSFORMS,
        EXP_TRANSFORMS,
        GAUSSIAN_TRANSFORMS,
        RELU_TRANSFORMS,
        SECH_TRANSFORMS,
        SIGMOID_TRANSFORMS,
        SINH_TRANSFORMS,
        SIN_TRANSFORMS,
        TANH_TRANSFORMS,
    )
}


# --------------------------------------------------------------------------- #
# Numerical oracle: composite Gauss-Legendre, independent of omnibias.
# --------------------------------------------------------------------------- #
_GROWTH_RATE = {"exp": 1.0, "sinh": 1.0, "cosh": 1.0, "sech": -1.0}
_TAIL_DECADES = 40.0


def _quad(f, a: float, b: float, panels: int = 60, order: int = 24) -> float:
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return float(np.dot(weights, f(nodes)))


def _forward(name: str, z: np.ndarray) -> np.ndarray:
    return np.asarray(get_activation(name).forward(jnp.asarray(z)))


def _numeric_laplace(name: str, s: float) -> float:
    rate = s - _GROWTH_RATE.get(name, 0.0)
    if name == "gaussian":
        upper = 45.0
    else:
        assert rate > 0.0, f"{name} at s={s} is outside the region of convergence"
        upper = max(40.0, _TAIL_DECADES / rate)
    return _quad(lambda z: _forward(name, z) * np.exp(-s * z), 0.0, upper)


def _numeric_fourier(name: str, xi: float, half: float = 60.0) -> float:
    return _quad(lambda z: _forward(name, z) * np.cos(xi * z), -half, half)


def _numeric_mellin(name: str, s: float) -> float:
    """Substitute ``z = exp(u)``: removes the endpoint singularity for ``s < 1``."""
    return _quad(lambda u: _forward(name, np.exp(u)) * np.exp(s * u), -_TAIL_DECADES / s, 5.0)


_NUMERIC = {
    "laplace": _numeric_laplace,
    "fourier": _numeric_fourier,
    "mellin": _numeric_mellin,
}

_SAMPLES: dict[tuple[str, str], tuple[float, ...]] = {
    ("exp", "laplace"): (1.5, 2.0, 3.7),
    ("relu", "laplace"): (0.5, 1.0, 2.5),
    ("sin", "laplace"): (0.4, 1.0, 2.5),
    ("cos", "laplace"): (0.4, 1.0, 2.5),
    ("sinh", "laplace"): (1.3, 2.0, 4.0),
    ("cosh", "laplace"): (1.3, 2.0, 4.0),
    ("gaussian", "laplace"): (-1.0, 0.0, 0.5, 2.0, 5.0),
    ("sigmoid", "laplace"): (0.3, 1.0, 2.5, 7.0),
    ("tanh", "laplace"): (0.3, 1.0, 2.5, 7.0),
    ("sech", "laplace"): (-0.5, 0.0, 1.0, 4.0),
    ("gaussian", "fourier"): (0.0, 0.5, 1.7, 3.0),
    ("sech", "fourier"): (0.0, 0.5, 1.7, 3.0),
    ("gaussian", "mellin"): (0.5, 1.0, 2.0, 4.5),
}


# --------------------------------------------------------------------------- #
# Coverage.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_every_documented_identity_is_actually_registered(
    name: str, transform: TransformName
) -> None:
    assert has_transform(name, transform), f"{name}/{transform} is in the table but not registered"


@pytest.mark.parametrize("transform", TRANSFORM_NAMES)
def test_no_base_activation_ships_an_undocumented_kernel(transform: TransformName) -> None:
    for name in list_activations():
        bundle = get_activation(name).transforms
        if bundle is None or id(bundle) not in _BASE_BUNDLES:
            continue
        if has_transform(name, transform):
            assert find_identity(name, transform) is not None, (
                f"{name}/{transform} ships a kernel with no entry in omnibias.core.transforms"
            )


@pytest.mark.parametrize("transform", TRANSFORM_NAMES)
def test_the_gaps_that_a_user_would_try_first_carry_reasons(transform: TransformName) -> None:
    for name in ("relu", "sigmoid", "tanh", "exp", "sin", "cos", "sech", "gaussian"):
        if has_transform(name, transform):
            continue
        assert find_exclusion(name, transform) is not None, (
            f"{name}/{transform} is missing and undocumented"
        )


def test_has_transform_rejects_an_unknown_transform_name() -> None:
    with pytest.raises(ValueError, match="transform must be one of"):
        has_transform("sigmoid", "hankel")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Correctness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_each_kernel_matches_quadrature_of_its_defining_integral(
    name: str, transform: TransformName
) -> None:
    fn = _TRANSFORM_FN[transform]
    numeric = _NUMERIC[transform]
    for point in _SAMPLES[(name, transform)]:
        closed = float(fn(name, jnp.asarray(point)))
        assert closed == pytest.approx(numeric(name, point), rel=1e-8), (
            f"{name}/{transform} at {point}"
        )


def test_the_gaussian_is_its_own_fourier_transform() -> None:
    xi = jnp.linspace(-4.0, 4.0, 17)
    expected = math.sqrt(2.0 * math.pi) * get_activation("gaussian").forward(xi)
    assert jnp.allclose(fourier_transform("gaussian", xi), expected, rtol=0, atol=1e-15)


def test_sech_transforms_to_a_dilated_sech() -> None:
    xi = jnp.linspace(-4.0, 4.0, 17)
    expected = math.pi * get_activation("sech").forward(0.5 * math.pi * xi)
    assert jnp.allclose(fourier_transform("sech", xi), expected, rtol=0, atol=1e-15)


def test_the_gaussian_laplace_kernel_survives_where_the_naive_form_nans() -> None:
    s = jnp.array([40.0, 100.0, 500.0])
    naive = math.sqrt(math.pi / 2) * jnp.exp(0.5 * s * s) * jax.scipy.special.erfc(
        s / math.sqrt(2.0)
    )
    assert jnp.isnan(naive).all(), "premise: the unscaled form is unusable here"
    got = laplace_transform("gaussian", s)
    assert jnp.isfinite(got).all()
    assert jnp.allclose(got, 1.0 / s, rtol=1e-3)


def test_sech_laplace_at_the_origin_is_the_area_under_sech() -> None:
    assert float(laplace_transform("sech", jnp.asarray(0.0))) == pytest.approx(math.pi / 2)


def test_a_pole_is_reported_as_infinity_rather_than_clamped() -> None:
    assert jnp.isinf(laplace_transform("exp", jnp.asarray(1.0)))
    assert jnp.isinf(laplace_transform("sinh", jnp.asarray(1.0)))


# --------------------------------------------------------------------------- #
# Differentiability and tracing.
# --------------------------------------------------------------------------- #
def _finite_difference(fn, x: float, eps: float = 1e-6) -> float:
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_each_kernel_has_correct_finite_difference_gradients(
    name: str, transform: TransformName
) -> None:
    fn = _TRANSFORM_FN[transform]
    scalar = lambda v: float(fn(name, jnp.asarray(v)))  # noqa: E731
    grad_fn = jax.grad(lambda v: fn(name, v).sum())
    for point in _SAMPLES[(name, transform)]:
        analytic = float(grad_fn(jnp.asarray(point)))
        assert analytic == pytest.approx(_finite_difference(scalar, point), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_each_kernel_traces_under_jit_and_vmap(name: str, transform: TransformName) -> None:
    fn = _TRANSFORM_FN[transform]
    points = jnp.asarray(_SAMPLES[(name, transform)])
    eager = fn(name, points)
    assert jnp.allclose(jax.jit(lambda v: fn(name, v))(points), eager)
    assert jnp.allclose(jax.vmap(lambda v: fn(name, v))(points), eager)


def test_fermi_dirac_mellin_is_differentiable_here_unlike_on_torch() -> None:
    """jax.scipy.special.zeta defines a gradient rule; torch.special.zeta does not."""
    scalar = lambda v: float(fermi_dirac_mellin(jnp.asarray(v)))  # noqa: E731
    grad_fn = jax.grad(lambda v: fermi_dirac_mellin(v).sum())
    for point in (1.5, 2.0, 4.0):
        analytic = float(grad_fn(jnp.asarray(point)))
        assert analytic == pytest.approx(_finite_difference(scalar, point), rel=1e-5)


# --------------------------------------------------------------------------- #
# The Fermi-Dirac companion and its scope wall.
# --------------------------------------------------------------------------- #
def test_fermi_dirac_mellin_matches_the_complementary_sigmoid_integral() -> None:
    for s in (1.2, 2.0, 3.0, 5.5):
        numeric = _quad(
            lambda u, v=s: np.exp(v * u) / (1.0 + np.exp(np.exp(u))), -_TAIL_DECADES / s, 5.0
        )
        assert float(fermi_dirac_mellin(jnp.asarray(s))) == pytest.approx(numeric, rel=1e-9), s


def test_fermi_dirac_mellin_refuses_to_cross_the_verified_zeta_wall() -> None:
    with pytest.raises(ValueError, match=r"requires s > 1"):
        fermi_dirac_mellin(jnp.array([2.0, 0.5]))
    with pytest.raises(ValueError, match="external obligation"):
        fermi_dirac_mellin(jnp.asarray(1.0))


def test_the_scope_check_is_skipped_under_tracing_rather_than_crashing() -> None:
    """A concrete comparison cannot run on a tracer; jit must still work.

    Same trade-off as ``omnibias.measure.jax.integraleq``'s solvability check:
    validate in eager mode, trace freely. The arithmetic is trace-safe either
    way, so a jitted call below the wall returns the (meaningless) continued
    value rather than raising -- which is why the eager guard exists.
    """
    jitted = jax.jit(fermi_dirac_mellin)
    assert jnp.allclose(jitted(jnp.asarray(2.0)), fermi_dirac_mellin(jnp.asarray(2.0)))
    assert jnp.isfinite(jax.grad(lambda v: fermi_dirac_mellin(v).sum())(jnp.asarray(3.0)))


def test_the_sigmoid_mellin_transform_is_not_silently_the_fermi_dirac_one() -> None:
    with pytest.raises(TypeError, match="complementary"):
        mellin_transform("sigmoid", jnp.asarray(2.0))


# --------------------------------------------------------------------------- #
# Guards and regions.
# --------------------------------------------------------------------------- #
def test_a_missing_kernel_raises_with_the_recorded_reason() -> None:
    with pytest.raises(TypeError, match="distributional"):
        fourier_transform("tanh", jnp.asarray(1.0))
    with pytest.raises(TypeError, match="Dirac"):
        fourier_transform("relu", jnp.asarray(1.0))
    with pytest.raises(TypeError, match="divergent"):
        mellin_transform("exp", jnp.asarray(2.0))


def test_an_activation_with_no_transforms_at_all_still_reports_cleanly() -> None:
    with pytest.raises(TypeError, match="has none"):
        laplace_transform("softplus", jnp.asarray(1.0))


@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_region_of_convergence_matches_the_table(name: str, transform: TransformName) -> None:
    lower, region = region_of_convergence(name, transform)
    identity = find_identity(name, transform)
    assert identity is not None
    assert lower == identity.min_argument
    assert region == identity.region


# --------------------------------------------------------------------------- #
# Trainable layers (functional).
# --------------------------------------------------------------------------- #
def test_the_block_evaluates_the_transform_at_its_argument() -> None:
    block = TransformBlock("sigmoid", "laplace", features=3, init_shift=2.0, init_scale=0.5)
    params = block.init()
    x = jnp.array([[0.0], [1.0], [-1.0]])
    got = block.apply(params, x)
    assert jnp.allclose(got, laplace_transform("sigmoid", block.argument(params, x)))
    assert got.shape == (3, 3)


def test_the_block_starts_inside_the_region_of_convergence() -> None:
    for name, transform in REGISTERED_PAIRS:
        block = TransformBlock(name, transform)
        params = block.init()
        lower, _ = region_of_convergence(name, transform)
        start = float(block.argument(params, jnp.zeros(1))[0])
        if lower is not None:
            assert start > lower, (name, transform, start, lower)
        assert jnp.isfinite(block.apply(params, jnp.zeros(1))).all(), (name, transform)


def test_the_softplus_reparameterisation_cannot_be_pushed_past_the_boundary() -> None:
    block = TransformBlock("relu", "laplace", features=4)  # region s > 0
    params = {"raw_shift": jnp.array([-1e3, -50.0, 0.0, 1e3]), "scale": jnp.ones(4)}
    argument = block.argument(params, jnp.zeros(4))
    assert (argument >= 0.0).all(), argument
    assert (argument[1:] > 0.0).all(), "only a float64 underflow may reach the boundary"


def test_an_entire_transform_uses_a_plain_shift() -> None:
    block = TransformBlock("gaussian", "fourier", features=2, init_shift=-3.0)
    assert block.min_argument is None
    assert jnp.allclose(block.argument(block.init(), jnp.zeros(2)), jnp.full((2,), -3.0))


def test_an_out_of_region_init_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="outside the region of convergence"):
        TransformBlock("exp", "laplace", init_shift=0.5)


def test_a_zero_feature_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="features must be >= 1"):
        TransformBlock("sigmoid", "laplace", features=0)


def test_the_named_subclasses_pin_their_transform() -> None:
    assert LaplaceTransform("sigmoid").transform == "laplace"
    assert FourierTransform("gaussian").transform == "fourier"
    assert MellinTransform("gaussian").transform == "mellin"


def test_a_block_trains_under_jitted_gradient_descent() -> None:
    """End to end: the closed-form transform composes with jit and grad."""
    block = LaplaceTransform("sigmoid", features=1, init_shift=4.0)
    params = block.init()
    target = laplace_transform("sigmoid", jnp.asarray(1.25))
    x = jnp.zeros(1)

    @jax.jit
    def step(p: dict[str, jnp.ndarray]) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        loss, grads = jax.value_and_grad(
            lambda q: jnp.sum((block.apply(q, x) - target) ** 2)
        )(p)
        return jax.tree.map(lambda a, g: a - 5.0 * g, p, grads), loss

    for _ in range(2000):
        params, loss = step(params)
    assert float(loss) < 1e-12
    assert float(block.argument(params, x)[0]) == pytest.approx(1.25, abs=1e-3)


# --------------------------------------------------------------------------- #
# Tempering, on a real registered activation.
# --------------------------------------------------------------------------- #
def test_a_tempered_activation_carries_correctly_scaled_transforms() -> None:
    from omnibias.core.spec import tempered

    beta = 2.5
    surrogate = tempered(get_activation("sech"), beta, name="tempered_sech")
    assert surrogate.transforms is not None
    assert surrogate.transforms.laplace is not None
    s = 1.5
    numeric = _quad(lambda z: (1.0 / np.cosh(beta * z)) * np.exp(-s * z), 0.0, 60.0)
    assert float(surrogate.transforms.laplace(jnp.asarray(s))) == pytest.approx(numeric, rel=1e-9)


def test_a_tempered_surrogate_inherits_its_base_transforms() -> None:
    derived = get_activation("soft_step")
    assert derived.transforms is not None
    assert id(derived.transforms) not in _BASE_BUNDLES
    assert has_transform("soft_step", "laplace")
    assert find_identity("soft_step", "laplace") is None
    assert jnp.isfinite(laplace_transform("soft_step", jnp.asarray(2.0)))
