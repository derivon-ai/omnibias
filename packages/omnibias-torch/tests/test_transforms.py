# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form integral transforms of the activation dictionary (PyTorch).

Three things are checked here, in increasing order of how much they would hurt
if they broke:

1. **Coverage.** The registry and the pure-Python table in
   :mod:`omnibias.core.transforms` agree exactly -- every registered kernel has
   a documented identity, and every activation *without* a kernel has a
   recorded reason. This is what keeps the honesty labels honest.
2. **Correctness.** Every kernel is checked against high-order Gauss-Legendre
   quadrature of its own defining integral, so the closed form is validated
   against the integral it claims to equal rather than against itself.
   ``tests/test_transforms_parity.py`` repeats this through
   ``omnibias.measure`` and adds the torch/jax comparison.
3. **Differentiability.** ``gradcheck`` on every kernel, since the point of a
   closed-form transform is that it drops into a network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.transforms import (
    TRANSFORM_NAMES,
    TransformName,
    find_exclusion,
    find_identity,
    identities,
)
from omnibias.torch.activations import get_activation, list_activations
from omnibias.torch.transforms import (
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

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_TRANSFORM_FN = {
    "laplace": laplace_transform,
    "fourier": fourier_transform,
    "mellin": mellin_transform,
}

#: (activation, transform) pairs that ship a closed-form kernel, read straight
#: off the shared table so this file can never claim coverage the table denies.
REGISTERED_PAIRS = [
    (identity.activation, identity.transform)
    for name in TRANSFORM_NAMES
    for identity in identities(name)
]


@pytest.fixture(autouse=True)
def _double_precision() -> object:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# --------------------------------------------------------------------------- #
# A self-contained numerical oracle: composite Gauss-Legendre on a truncated
# domain. Independent of omnibias, so a kernel bug cannot hide behind a shared
# helper.
# --------------------------------------------------------------------------- #
def _gauss_legendre(a: float, b: float, panels: int = 60, order: int = 24) -> tuple[
    np.ndarray, np.ndarray
]:
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def _quad(f, a: float, b: float, panels: int = 60, order: int = 24) -> float:
    nodes, weights = _gauss_legendre(a, b, panels, order)
    return float(np.dot(weights, f(nodes)))


#: Exponential growth rate of each activation on the positive half line, used
#: to size the truncation window: the Laplace integrand decays like
#: ``exp(-(s - growth) z)``, so a fixed window silently loses accuracy for ``s``
#: near the boundary of the region of convergence.
_GROWTH_RATE = {"exp": 1.0, "sinh": 1.0, "cosh": 1.0, "sech": -1.0}

#: Enough decades that the discarded tail sits below double-precision round-off.
_TAIL_DECADES = 40.0


def _forward(name: str, z: np.ndarray) -> np.ndarray:
    return get_activation(name).forward(torch.from_numpy(z)).numpy()


def _numeric_laplace(name: str, s: float) -> float:
    rate = s - _GROWTH_RATE.get(name, 0.0)
    if name == "gaussian":
        upper = 45.0  # decays like exp(-z^2/2), so s plays no role in the tail
    else:
        assert rate > 0.0, f"{name} at s={s} is outside the region of convergence"
        upper = max(40.0, _TAIL_DECADES / rate)
    return _quad(lambda z: _forward(name, z) * np.exp(-s * z), 0.0, upper)


def _numeric_fourier(name: str, xi: float, half: float = 60.0) -> float:
    """Real part; every registered Fourier kernel is real and even."""
    return _quad(lambda z: _forward(name, z) * np.cos(xi * z), -half, half)


def _numeric_mellin(name: str, s: float) -> float:
    """Substitute ``z = exp(u)`` so the ``z^(s-1)`` endpoint singularity vanishes.

    For ``s < 1`` the raw integrand blows up at the origin and Gauss-Legendre
    panels resolve it badly; in ``u`` the integrand is smooth and decays at both
    ends, which is exactly what Gauss-Legendre is good at.
    """
    return _quad(lambda u: _forward(name, np.exp(u)) * np.exp(s * u), -_TAIL_DECADES / s, 5.0)


_NUMERIC = {
    "laplace": _numeric_laplace,
    "fourier": _numeric_fourier,
    "mellin": _numeric_mellin,
}

#: Sample points strictly inside each region of convergence. Chosen away from
#: the boundary because the *closed form* is exact there while the truncated
#: quadrature oracle is not: near a pole the integrand stops decaying inside
#: the truncation window.
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
# 1. Coverage: the registry and the shared table agree, in both directions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_every_documented_identity_is_actually_registered(
    name: str, transform: TransformName
) -> None:
    assert has_transform(name, transform), f"{name}/{transform} is in the table but not registered"


#: The kernel bundles attached to *base* activations. A spec carrying one of
#: these is a table entry; a spec carrying anything else got its transforms by
#: derivation (tempering), which is legitimate but not separately documented.
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


@pytest.mark.parametrize("transform", TRANSFORM_NAMES)
def test_no_base_activation_ships_an_undocumented_kernel(transform: TransformName) -> None:
    """A hand-written kernel must have a documented identity behind it."""
    for name in list_activations():
        bundle = get_activation(name).transforms
        if bundle is None or id(bundle) not in _BASE_BUNDLES:
            continue
        if has_transform(name, transform):
            assert find_identity(name, transform) is not None, (
                f"{name}/{transform} ships a kernel with no entry in omnibias.core.transforms"
            )


def test_a_tempered_surrogate_inherits_its_base_transforms() -> None:
    """soft_step = tempered(sigmoid), so it carries a beta-scaled Laplace kernel.

    Derived specs are the one legitimate way to hold a kernel without a table
    entry: the scaling law is exact and lives in ``make_tempered_transforms``,
    so documenting each surrogate separately would duplicate it.
    """
    derived = get_activation("soft_step")
    assert derived.transforms is not None
    assert id(derived.transforms) not in _BASE_BUNDLES
    assert has_transform("soft_step", "laplace")
    assert find_identity("soft_step", "laplace") is None
    assert torch.isfinite(laplace_transform("soft_step", torch.tensor(2.0)))


@pytest.mark.parametrize("transform", TRANSFORM_NAMES)
def test_the_gaps_that_a_user_would_try_first_carry_reasons(transform: TransformName) -> None:
    """A caller reaching for sigmoid's Fourier transform must be told why not."""
    for name in ("relu", "sigmoid", "tanh", "exp", "sin", "cos", "sech", "gaussian"):
        if has_transform(name, transform):
            continue
        assert find_exclusion(name, transform) is not None, (
            f"{name}/{transform} is missing and undocumented"
        )


def test_has_transform_rejects_an_unknown_transform_name() -> None:
    with pytest.raises(ValueError, match="transform must be one of"):
        has_transform("sigmoid", "hankel")  # type: ignore[arg-type]


def test_coverage_is_not_vacuous() -> None:
    """Guards against the table being emptied and every check passing trivially."""
    assert len(REGISTERED_PAIRS) >= 13
    assert sum(1 for _, t in REGISTERED_PAIRS if t == "laplace") >= 9


# --------------------------------------------------------------------------- #
# 2. Correctness against the defining integral.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_each_kernel_matches_quadrature_of_its_defining_integral(
    name: str, transform: TransformName
) -> None:
    fn = _TRANSFORM_FN[transform]
    numeric = _NUMERIC[transform]
    for point in _SAMPLES[(name, transform)]:
        closed = float(fn(name, torch.tensor(point)))
        assert closed == pytest.approx(numeric(name, point), rel=1e-8), (
            f"{name}/{transform} at {point}"
        )


def test_the_gaussian_is_its_own_fourier_transform() -> None:
    """The classical analytic pair, up to the sqrt(2 pi) of this convention."""
    xi = torch.linspace(-4.0, 4.0, 17)
    got = fourier_transform("gaussian", xi)
    expected = math.sqrt(2.0 * math.pi) * get_activation("gaussian").forward(xi)
    assert torch.allclose(got, expected, rtol=0, atol=1e-15)


def test_sech_transforms_to_a_dilated_sech() -> None:
    """The second self-reciprocal profile: F[sech](xi) = pi sech(pi xi / 2)."""
    xi = torch.linspace(-4.0, 4.0, 17)
    got = fourier_transform("sech", xi)
    expected = math.pi * get_activation("sech").forward(0.5 * math.pi * xi)
    assert torch.allclose(got, expected, rtol=0, atol=1e-15)


def test_the_gaussian_laplace_kernel_survives_where_the_naive_form_nans() -> None:
    """exp(s^2/2) overflows and erfc(s/sqrt2) underflows; erfcx computes the product."""
    s = torch.tensor([40.0, 100.0, 500.0])
    naive = math.sqrt(math.pi / 2) * torch.exp(0.5 * s * s) * torch.special.erfc(s / math.sqrt(2.0))
    assert torch.isnan(naive).all(), "premise: the unscaled form is unusable here"
    got = laplace_transform("gaussian", s)
    assert torch.isfinite(got).all()
    # Asymptotically L[gaussian](s) -> 1/s for large s.
    assert torch.allclose(got, 1.0 / s, rtol=1e-3)


def test_the_mellin_kernel_is_finite_across_a_wide_useful_range() -> None:
    """The log-space assembly holds up to s ~ 300; past that Gamma itself overflows."""
    got = mellin_transform("gaussian", torch.tensor([1e-3, 1.0, 50.0, 250.0]))
    assert torch.isfinite(got).all()
    assert torch.isinf(mellin_transform("gaussian", torch.tensor(1e4)))


def test_relu_laplace_is_the_ramp_transform() -> None:
    s = torch.tensor([0.5, 1.0, 3.0])
    assert torch.allclose(laplace_transform("relu", s), 1.0 / (s * s))


def test_sech_laplace_at_the_origin_is_the_area_under_sech() -> None:
    """L[sech](0) = int_0^inf sech = pi/2, and 0 is inside the region s > -1."""
    assert float(laplace_transform("sech", torch.tensor(0.0))) == pytest.approx(math.pi / 2)


# --------------------------------------------------------------------------- #
# 3. Differentiability.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_each_kernel_has_correct_finite_difference_gradients(
    name: str, transform: TransformName
) -> None:
    fn = _TRANSFORM_FN[transform]
    point = torch.tensor(_SAMPLES[(name, transform)], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: fn(name, v), (point,), eps=1e-6, atol=1e-7)


def test_fermi_dirac_mellin_is_not_differentiable_on_torch() -> None:
    """torch.special.zeta ships no derivative rule; the JAX twin does.

    Pinned as a test so the documented backend asymmetry is a fact about the
    build rather than a claim in a docstring. If torch ever adds the rule this
    test fails loudly and the docs get corrected.
    """
    s = torch.tensor([1.5, 2.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(NotImplementedError, match="zeta"):
        fermi_dirac_mellin(s).sum().backward()


# --------------------------------------------------------------------------- #
# The Fermi-Dirac companion and its scope wall.
# --------------------------------------------------------------------------- #
def test_fermi_dirac_mellin_matches_the_complementary_sigmoid_integral() -> None:
    for s in (1.2, 2.0, 3.0, 5.5):
        numeric = _quad(  # z = exp(u), as in _numeric_mellin
            lambda u, v=s: np.exp(v * u) / (1.0 + np.exp(np.exp(u))), -_TAIL_DECADES / s, 5.0
        )
        got = float(fermi_dirac_mellin(torch.tensor(s)))
        assert got == pytest.approx(numeric, rel=1e-9), s


def test_fermi_dirac_mellin_refuses_to_cross_the_verified_zeta_wall() -> None:
    """Re(s) > 1 is where omnibias.core.verified.dirichlet stops; so does this."""
    with pytest.raises(ValueError, match=r"requires s > 1"):
        fermi_dirac_mellin(torch.tensor([2.0, 0.5]))
    with pytest.raises(ValueError, match="external obligation"):
        fermi_dirac_mellin(torch.tensor(1.0))


def test_the_sigmoid_mellin_transform_is_not_silently_the_fermi_dirac_one() -> None:
    """The whole point of shipping it separately: sigmoid's own Mellin diverges."""
    with pytest.raises(TypeError, match="complementary"):
        mellin_transform("sigmoid", torch.tensor(2.0))


# --------------------------------------------------------------------------- #
# Guards and regions.
# --------------------------------------------------------------------------- #
def test_a_missing_kernel_raises_with_the_recorded_reason() -> None:
    with pytest.raises(TypeError, match="distributional"):
        fourier_transform("tanh", torch.tensor(1.0))
    with pytest.raises(TypeError, match="Dirac"):
        fourier_transform("relu", torch.tensor(1.0))
    with pytest.raises(TypeError, match="divergent"):
        mellin_transform("exp", torch.tensor(2.0))


def test_an_activation_with_no_transforms_at_all_still_reports_cleanly() -> None:
    with pytest.raises(TypeError, match="has none"):
        laplace_transform("softplus", torch.tensor(1.0))


@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_region_of_convergence_matches_the_table(name: str, transform: TransformName) -> None:
    lower, region = region_of_convergence(name, transform)
    identity = find_identity(name, transform)
    assert identity is not None
    assert lower == identity.min_argument
    assert region == identity.region


def test_a_pole_is_reported_as_infinity_rather_than_clamped() -> None:
    """Outside the region the closed form is returned as-is: no silent repair."""
    assert torch.isinf(laplace_transform("exp", torch.tensor(1.0)))
    assert torch.isinf(laplace_transform("sinh", torch.tensor(1.0)))


# --------------------------------------------------------------------------- #
# Trainable layers.
# --------------------------------------------------------------------------- #
def test_the_block_evaluates_the_transform_at_its_argument() -> None:
    block = TransformBlock("sigmoid", "laplace", features=3, init_shift=2.0, init_scale=0.5)
    x = torch.tensor([[0.0], [1.0], [-1.0]])
    got = block(x)
    expected = laplace_transform("sigmoid", block.argument(x))
    assert torch.allclose(got, expected)
    assert got.shape == (3, 3)


def test_the_block_starts_inside_the_region_of_convergence() -> None:
    for name, transform in REGISTERED_PAIRS:
        block = TransformBlock(name, transform)
        lower, _ = region_of_convergence(name, transform)
        start = float(block.argument(torch.zeros(1))[0])
        if lower is not None:
            assert start > lower, (name, transform, start, lower)
        assert torch.isfinite(block(torch.zeros(1))).all(), (name, transform)


def test_the_softplus_reparameterisation_cannot_be_pushed_past_the_boundary() -> None:
    """A hostile optimizer step must not move a learnable s into the divergent side.

    The guarantee is one-sided and closed: softplus underflows to exactly zero
    below about raw = -745 in float64, so the argument can reach the boundary
    but never cross it. An unconstrained shift would sit at -1000 here, deep in
    the half-plane where 1/s^2 means nothing.
    """
    block = TransformBlock("relu", "laplace", features=4)  # region s > 0
    with torch.no_grad():
        block.raw_shift.copy_(torch.tensor([-1e3, -50.0, 0.0, 1e3]))
    argument = block.argument(torch.zeros(4))
    assert (argument >= 0.0).all(), argument
    assert (argument[1:] > 0.0).all(), "only a float64 underflow may reach the boundary"
    assert torch.isfinite(block(torch.zeros(4))[1:]).all()


def test_an_entire_transform_uses_a_plain_shift() -> None:
    block = TransformBlock("gaussian", "fourier", features=2, init_shift=-3.0)
    assert block.min_argument is None
    assert torch.allclose(block.argument(torch.zeros(2)), torch.full((2,), -3.0))


def test_an_out_of_region_init_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="outside the region of convergence"):
        TransformBlock("exp", "laplace", init_shift=0.5)  # region s > 1


def test_a_frozen_block_exposes_buffers_not_parameters() -> None:
    block = TransformBlock("sigmoid", "laplace", learnable=False)
    assert list(block.parameters()) == []
    assert {name for name, _ in block.named_buffers()} == {"raw_shift", "scale"}


def test_block_parameters_receive_gradients() -> None:
    block = TransformBlock("tanh", "laplace", features=2)
    loss = block(torch.tensor([0.5, -0.5])).sum()
    loss.backward()
    assert block.raw_shift.grad is not None
    assert block.scale.grad is not None
    assert torch.isfinite(block.raw_shift.grad).all()
    assert torch.isfinite(block.scale.grad).all()


def test_a_zero_feature_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="features must be >= 1"):
        TransformBlock("sigmoid", "laplace", features=0)


def test_the_named_subclasses_pin_their_transform() -> None:
    assert LaplaceTransform("sigmoid").transform == "laplace"
    assert FourierTransform("gaussian").transform == "fourier"
    assert MellinTransform("gaussian").transform == "mellin"


def test_a_block_survives_a_training_loop_on_a_bounded_target() -> None:
    """End to end: the closed-form transform trains like any other layer."""
    torch.manual_seed(0)
    block = LaplaceTransform("sigmoid", features=1, init_shift=4.0)
    target = laplace_transform("sigmoid", torch.tensor(1.25))
    optimizer = torch.optim.Adam(block.parameters(), lr=0.05)
    x = torch.zeros(1)
    for _ in range(400):
        optimizer.zero_grad()
        loss = (block(x) - target).pow(2).sum()
        loss.backward()
        optimizer.step()
    assert float(loss) < 1e-10
    assert float(block.argument(x)[0]) == pytest.approx(1.25, abs=1e-3)


# --------------------------------------------------------------------------- #
# Tempering, on a real registered activation.
# --------------------------------------------------------------------------- #
def test_a_tempered_activation_carries_correctly_scaled_transforms() -> None:
    """L[b^-p g(b .)](s) = b^-(p+1) L[g](s/b), checked against direct quadrature."""
    from omnibias.core.spec import tempered

    beta = 2.5
    surrogate = tempered(get_activation("sech"), beta, name="tempered_sech")
    assert surrogate.transforms is not None
    assert surrogate.transforms.laplace is not None
    s = 1.5
    got = float(surrogate.transforms.laplace(torch.tensor(s)))
    numeric = _quad(
        lambda z: (1.0 / np.cosh(beta * z)) * np.exp(-s * z), 0.0, 60.0
    )
    assert got == pytest.approx(numeric, rel=1e-9)


def test_tempering_scales_the_mellin_transform_by_the_argument_dependent_power() -> None:
    from omnibias.core.spec import tempered

    beta = 3.0
    surrogate = tempered(get_activation("gaussian"), beta, scale="one_over_beta")
    assert surrogate.transforms is not None
    assert surrogate.transforms.mellin is not None
    s = 2.0
    got = float(surrogate.transforms.mellin(torch.tensor(s)))
    numeric = _quad(
        lambda z: np.exp(-((beta * z) ** 2) / 2.0) / beta * z ** (s - 1.0), 0.0, 60.0
    )
    assert got == pytest.approx(numeric, rel=1e-9)
