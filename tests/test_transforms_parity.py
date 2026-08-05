# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend and cross-package validation of the closed-form transforms.

The per-backend suites already check each kernel against its own defining
integral. This file adds the two checks that need more than one package:

* **torch/jax parity.** Both backends evaluate the same identity table, so they
  must agree to float64 round-off. Where they differ it is by a last ulp in
  ``digamma`` / ``zeta`` / ``erfcx``, not by a different formula, so the
  tolerance here is tight enough to catch a genuine divergence in the
  arithmetic and no looser.
* **Independent oracles.** ``omnibias.measure``'s ``lebesgue_integral``
  re-derives every transform by quadrature on a Gauss-Legendre measure, and
  ``omnibias.fractional``'s ``lerch`` re-derives the sigmoid / tanh Laplace
  kernels from the alternating series they came from. Both are *test-only*
  imports: ``omnibias-torch`` and ``omnibias-jax`` must never depend on either
  package, which is exactly why the digamma form ships and the series does not.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from omnibias.core.transforms import TRANSFORM_NAMES, TransformName, identities

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.transforms import fermi_dirac_mellin as jax_fermi  # noqa: E402
from omnibias.jax.transforms import fourier_transform as jax_fourier  # noqa: E402
from omnibias.jax.transforms import laplace_transform as jax_laplace  # noqa: E402
from omnibias.jax.transforms import mellin_transform as jax_mellin  # noqa: E402
from omnibias.torch.transforms import fermi_dirac_mellin as torch_fermi  # noqa: E402
from omnibias.torch.transforms import fourier_transform as torch_fourier  # noqa: E402
from omnibias.torch.transforms import laplace_transform as torch_laplace  # noqa: E402
from omnibias.torch.transforms import mellin_transform as torch_mellin  # noqa: E402


@pytest.fixture(autouse=True)
def _double_precision() -> object:
    """Set float64 per test rather than once at import.

    ``packages/omnibias-torch/tests/conftest.py`` deliberately resets the default
    dtype to the process original (float32) around every test it owns, so a
    module-level ``set_default_dtype`` here would survive collection but not a
    combined run -- these parity tolerances are float64 tolerances, and under
    float32 they fail by six orders of magnitude.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


_TORCH_FN = {"laplace": torch_laplace, "fourier": torch_fourier, "mellin": torch_mellin}
_JAX_FN = {"laplace": jax_laplace, "fourier": jax_fourier, "mellin": jax_mellin}

REGISTERED_PAIRS = [
    (identity.activation, identity.transform)
    for name in TRANSFORM_NAMES
    for identity in identities(name)
]

#: Points strictly inside each region of convergence.
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
# torch <-> jax parity.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_the_two_backends_agree_to_float64_round_off(
    name: str, transform: TransformName
) -> None:
    points = _SAMPLES[(name, transform)]
    got_torch = _TORCH_FN[transform](name, torch.tensor(points)).numpy()
    got_jax = np.asarray(_JAX_FN[transform](name, jnp.asarray(points)))
    np.testing.assert_allclose(got_torch, got_jax, rtol=1e-13, atol=1e-15)


def test_the_fermi_dirac_companion_agrees_across_backends() -> None:
    points = (1.2, 2.0, 3.0, 5.5)
    got_torch = torch_fermi(torch.tensor(points)).numpy()
    got_jax = np.asarray(jax_fermi(jnp.asarray(points)))
    np.testing.assert_allclose(got_torch, got_jax, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("transform", TRANSFORM_NAMES)
def test_both_backends_refuse_the_same_pairs(transform: TransformName) -> None:
    """Coverage must be identical: a kernel on one backend only is a parity bug."""
    from omnibias.jax.activations import list_activations as jax_names
    from omnibias.jax.transforms import has_transform as jax_has
    from omnibias.torch.activations import list_activations as torch_names
    from omnibias.torch.transforms import has_transform as torch_has

    shared = set(torch_names()) & set(jax_names())
    assert len(shared) > 30, "premise: the two registries overlap almost entirely"
    for name in sorted(shared):
        assert torch_has(name, transform) == jax_has(name, transform), (name, transform)


# --------------------------------------------------------------------------- #
# omnibias.measure as the quadrature oracle.
# --------------------------------------------------------------------------- #
#: Gauss-Legendre nodes cost O(n^2) to build, so the measures are cached across
#: the sample points of a pair. 800 nodes reproduce every kernel to ~1e-13,
#: including the two demanding cases: sigmoid's Laplace transform at s = 0.3
#: needs a window 170 units wide before the tail is negligible, and sech's
#: Fourier transform is oscillatory over a window its slow exp(-|z|) decay keeps
#: wide, so it needs enough nodes per wavelength across the whole domain.
_QUADRATURE_NODES = 800


@functools.cache
def _quadrature_measure(lower: float, upper: float) -> object:
    from omnibias.measure._core.measure import lebesgue

    return lebesgue([(lower, upper)], _QUADRATURE_NODES)


def _measure_integral(integrand, lower: float, upper: float) -> float:
    """``int_lower^upper f`` on a Gauss-Legendre Lebesgue measure."""
    from omnibias.measure._core.integrate import lebesgue_integral

    measure = _quadrature_measure(lower, upper)
    return float(lebesgue_integral(lambda z: integrand(z[:, 0]), measure))


@pytest.mark.parametrize(("name", "transform"), REGISTERED_PAIRS, ids=lambda v: str(v))
def test_measure_quadrature_reproduces_every_kernel(
    name: str, transform: TransformName
) -> None:
    """The oracle the plan asks for: omnibias's own quadrature, not scipy's."""
    pytest.importorskip("omnibias.measure")
    from omnibias.torch.activations import get_activation

    spec = get_activation(name)
    forward = lambda z: spec.forward(torch.from_numpy(np.asarray(z))).numpy()  # noqa: E731

    samples = _SAMPLES[(name, transform)]
    growth = {"exp": 1.0, "sinh": 1.0, "cosh": 1.0, "sech": -1.0}.get(name, 0.0)
    # One window per pair, wide enough for the slowest-decaying sample, so the
    # cached measure is shared across the sample points.
    if name == "gaussian" and transform == "laplace":
        window = (0.0, 45.0)  # decays like exp(-z^2/2); s plays no part
    elif transform == "laplace":
        window = (0.0, max(40.0, 40.0 / (min(samples) - growth)))
    elif transform == "fourier":
        # sech decays only like 2 exp(-|z|), but 2 exp(-40) is already 8e-18;
        # a wider window just adds oscillations for the rule to resolve.
        window = (-40.0, 40.0)
    else:  # mellin, on the log-substituted integrand z = exp(u)
        window = (-40.0 / min(samples), 5.0)

    for point in samples:
        if transform == "laplace":
            integrand = lambda z, p=point: forward(z) * np.exp(-p * z)  # noqa: E731
        elif transform == "fourier":
            integrand = lambda z, p=point: forward(z) * np.cos(p * z)  # noqa: E731
        else:
            integrand = lambda u, p=point: forward(np.exp(u)) * np.exp(p * u)  # noqa: E731
        numeric = _measure_integral(integrand, *window)
        closed = float(_TORCH_FN[transform](name, torch.tensor(point)))
        assert closed == pytest.approx(numeric, rel=1e-8), f"{name}/{transform} at {point}"


def test_measure_quadrature_reproduces_the_fermi_dirac_integral() -> None:
    pytest.importorskip("omnibias.measure")
    for s in (1.2, 2.0, 3.0, 5.5):
        numeric = _measure_integral(
            lambda u, v=s: np.exp(v * u) / (1.0 + np.exp(np.exp(u))), -40.0 / s, 5.0
        )
        assert float(torch_fermi(torch.tensor(s))) == pytest.approx(numeric, rel=1e-9), s


# --------------------------------------------------------------------------- #
# omnibias.fractional's Lerch transcendent as an independent series oracle.
# --------------------------------------------------------------------------- #
def _lerch_alternating(s: float, terms: int) -> float:
    """``Phi(-1, 1, s)`` from omnibias-fractional, a *test-only* dependency."""
    lerch = pytest.importorskip("omnibias.fractional.torch.ops.special").lerch
    return float(lerch(torch.tensor(-1.0), 1.0, torch.tensor(s), terms=terms))


def test_the_sigmoid_laplace_kernel_matches_the_lerch_series_it_came_from() -> None:
    r"""``L[sigmoid](s) = Phi(-1, 1, s)``, the series the digamma form replaces.

    The tolerance is set by the series, not by the kernel: an alternating sum of
    ``1/(s+k)`` truncated at ``N`` terms carries an error of order ``1/(2(s+N))``,
    which is why omnibias evaluates the digamma form in production and keeps the
    series here as an independent check on the identity.
    """
    terms = 40000
    for s in (0.5, 1.0, 2.5, 7.0):
        closed = float(torch_laplace("sigmoid", torch.tensor(s)))
        assert closed == pytest.approx(_lerch_alternating(s, terms), abs=1.0 / (s + terms))


def test_the_tanh_laplace_kernel_matches_its_lerch_form() -> None:
    """``L[tanh](s) = Phi(-1, 1, s/2) - 1/s``."""
    terms = 40000
    for s in (0.5, 1.0, 2.5, 7.0):
        closed = float(torch_laplace("tanh", torch.tensor(s)))
        series = _lerch_alternating(s / 2.0, terms) - 1.0 / s
        assert closed == pytest.approx(series, abs=2.0 / (s + terms))


def test_the_sech_laplace_kernel_matches_its_lerch_form() -> None:
    """``L[sech](s) = Phi(-1, 1, (s+1)/2)``."""
    terms = 40000
    for s in (0.0, 1.0, 4.0):
        closed = float(torch_laplace("sech", torch.tensor(s)))
        assert closed == pytest.approx(
            _lerch_alternating((s + 1.0) / 2.0, terms), abs=2.0 / (s + terms)
        )


def test_neither_backend_imports_omnibias_fractional_or_measure() -> None:
    """The oracles are test-only; a production import would be a dependency bug."""
    import subprocess
    import sys

    probe = (
        "import sys; import omnibias.torch.transforms, omnibias.jax.transforms; "
        "leaked = [m for m in sys.modules "
        "if m.startswith(('omnibias.fractional', 'omnibias.measure'))]; "
        "print(leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout


# --------------------------------------------------------------------------- #
# Known analytic pairs, checked once more across both backends together.
# --------------------------------------------------------------------------- #
def test_both_backends_reproduce_the_gaussian_self_transform() -> None:
    xi = np.linspace(-4.0, 4.0, 17)
    expected = math.sqrt(2.0 * math.pi) * np.exp(-0.5 * xi * xi)
    np.testing.assert_allclose(torch_fourier("gaussian", torch.from_numpy(xi)).numpy(), expected)
    np.testing.assert_allclose(np.asarray(jax_fourier("gaussian", jnp.asarray(xi))), expected)


def test_both_backends_reproduce_the_sech_self_transform() -> None:
    xi = np.linspace(-4.0, 4.0, 17)
    expected = math.pi / np.cosh(0.5 * math.pi * xi)
    np.testing.assert_allclose(torch_fourier("sech", torch.from_numpy(xi)).numpy(), expected)
    np.testing.assert_allclose(np.asarray(jax_fourier("sech", jnp.asarray(xi))), expected)
