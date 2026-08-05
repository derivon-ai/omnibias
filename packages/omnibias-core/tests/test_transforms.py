# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The pure-Python integral-transform identity tables.

These tests police the *table*, not the numerics: the tensor kernels are
validated against quadrature in the backend suites. What matters here is that
the table stays internally consistent and that every gap carries a recorded
mathematical reason, since the backends and the docs both read from it.
"""

from __future__ import annotations

import pytest
from omnibias.core.spec import ActivationSpec, TransformKernels, make_tempered_transforms, tempered
from omnibias.core.transforms import (
    EXCLUDED_TRANSFORMS,
    FERMI_DIRAC_MELLIN,
    SPECTRAL_VARIABLE,
    TEMPERING_LAWS,
    TRANSFORM_DEFINITION,
    TRANSFORM_NAMES,
    find_exclusion,
    find_identity,
    identities,
    registered_activations,
)

_ALL_IDENTITIES = tuple(i for name in TRANSFORM_NAMES for i in identities(name)) + (
    FERMI_DIRAC_MELLIN,
)


def test_every_transform_name_is_fully_described() -> None:
    for name in TRANSFORM_NAMES:
        assert name in SPECTRAL_VARIABLE
        assert name in TRANSFORM_DEFINITION
        assert name in TEMPERING_LAWS


def test_identities_reject_an_unknown_transform() -> None:
    with pytest.raises(ValueError, match="transform must be one of"):
        identities("hankel")  # type: ignore[arg-type]


@pytest.mark.parametrize("identity", _ALL_IDENTITIES, ids=lambda i: f"{i.activation}-{i.transform}")
def test_each_identity_is_completely_filled_in(identity: object) -> None:
    assert identity.activation
    assert identity.expression
    assert identity.region
    assert identity.evaluated_with
    assert identity.transform in TRANSFORM_NAMES


@pytest.mark.parametrize("identity", _ALL_IDENTITIES, ids=lambda i: f"{i.activation}-{i.transform}")
def test_the_numeric_bound_agrees_with_the_prose_region(identity: object) -> None:
    """``min_argument`` is what the layers use; ``region`` is what humans read."""
    lower = identity.min_argument
    region = identity.region
    variable = SPECTRAL_VARIABLE[identity.transform]
    if lower is None:
        assert region.startswith("all real"), region
    else:
        assert region == f"{variable} > {lower:g}", (region, lower)


def test_no_activation_is_both_registered_and_excluded() -> None:
    """A pair is either a shipped identity or a recorded gap -- never both."""
    for excluded in EXCLUDED_TRANSFORMS:
        assert find_identity(excluded.activation, excluded.transform) is None, (
            f"{excluded.activation}/{excluded.transform} is both registered and excluded"
        )


def test_identities_are_unique_per_activation_and_transform() -> None:
    for name in TRANSFORM_NAMES:
        activations = registered_activations(name)
        assert len(activations) == len(set(activations)), name


def test_lookups_miss_cleanly() -> None:
    assert find_identity("softplus", "laplace") is None
    assert find_exclusion("softplus", "laplace") is None
    assert find_exclusion("tanh", "fourier") is not None


def test_the_fermi_dirac_entry_is_not_attached_to_sigmoid() -> None:
    """The Fermi-Dirac integral is the Mellin transform of a *different* function."""
    assert FERMI_DIRAC_MELLIN.activation == "sigmoid_complement"
    assert find_identity("sigmoid", "mellin") is None
    excluded = find_exclusion("sigmoid", "mellin")
    assert excluded is not None
    assert excluded.reason == "complementary"


def test_the_fermi_dirac_scope_wall_is_recorded() -> None:
    """Re(s) > 1 mirrors the verified dirichlet scope, and the table must say so."""
    assert FERMI_DIRAC_MELLIN.min_argument == 1.0
    assert "zeta" in FERMI_DIRAC_MELLIN.note
    assert "external obligation" in FERMI_DIRAC_MELLIN.note


def test_the_registered_exp_mellin_gap_names_the_sign_convention() -> None:
    """omnibias's exp is e^{+z}; the Gamma(s) pair belongs to e^{-z}."""
    excluded = find_exclusion("exp", "mellin")
    assert excluded is not None
    assert excluded.reason == "divergent"
    assert "exp(+z)" in excluded.detail


@pytest.mark.parametrize("excluded", EXCLUDED_TRANSFORMS, ids=lambda e: f"{e.activation}-{e.reason}")
def test_every_gap_explains_itself(excluded: object) -> None:
    detail = excluded.detail
    assert len(detail) > 40, "a gap needs a reason, not a label"
    assert excluded.reason in {
        "divergent",
        "distributional",
        "complementary",
        "conditional",
        "unavailable",
    }


# --------------------------------------------------------------------------- #
# Tempering laws.
# --------------------------------------------------------------------------- #
def _linear(z: float) -> float:
    return z


def test_tempering_propagates_each_kernel_by_its_exact_law() -> None:
    """f_b(z) = b^-p g(bz)  =>  L,F scale the argument; M scales by b^{-p-s}."""
    base = TransformKernels[float](
        laplace=lambda s: s + 10.0,
        fourier=lambda xi: xi + 100.0,
        mellin=lambda s: 1000.0,
    )
    beta, p = 2.0, 1
    scaled = make_tempered_transforms(base, beta, scale_power=p)
    assert scaled.laplace is not None and scaled.fourier is not None
    assert scaled.mellin is not None
    # L[f_b](s) = b^-(p+1) L[g](s/b)
    assert scaled.laplace(6.0) == pytest.approx((6.0 / 2.0 + 10.0) / 2.0**2)
    # F[f_b](xi) = b^-(p+1) F[g](xi/b)
    assert scaled.fourier(6.0) == pytest.approx((6.0 / 2.0 + 100.0) / 2.0**2)
    # M[f_b](s) = b^(-p-s) M[g](s)
    assert scaled.mellin(3.0) == pytest.approx(1000.0 * 2.0 ** (-1 - 3.0))


def test_tempering_never_invents_a_missing_kernel() -> None:
    scaled = make_tempered_transforms(TransformKernels[float](laplace=lambda s: 1.0), 3.0)
    assert scaled.laplace is not None
    assert scaled.fourier is None
    assert scaled.mellin is None


def test_tempered_spec_carries_the_scaled_transforms() -> None:
    base = ActivationSpec[float](
        name="g",
        forward=_linear,
        fastpath=lambda z, n: z,
        transforms=TransformKernels[float](laplace=lambda s: 1.0 / s),
    )
    surrogate = tempered(base, 4.0, scale="one_over_beta")
    assert surrogate.transforms is not None
    assert surrogate.transforms.laplace is not None
    # b^-(p+1) * (1 / (s/b)) = b^-2 * b/s = 1/(b s) with p = 1, b = 4.
    assert surrogate.transforms.laplace(2.0) == pytest.approx(1.0 / (4.0 * 2.0))


def test_tempering_a_spec_without_transforms_leaves_them_none() -> None:
    base = ActivationSpec[float](name="g", forward=_linear, fastpath=lambda z, n: z)
    assert tempered(base, 2.0).transforms is None
