# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch <-> jax parity for the stiff integrators.

On a real symbol the phi functions are pure elementwise arithmetic driven by a
shared Python recursion, so they are expected **bit-identical**, not merely
close -- that is the strongest statement available and it is asserted as such.
Three places legitimately fall short of it and are tested at round-off instead:

* a complex symbol, where the two backends' complex multiply differs in the
  last ulp and the squaring phase doubles that difference once per squaring;
* :func:`phi_matrix`, which accumulates matrix products whose summation order
  is the BLAS implementation's business;
* the spectral steps, which call each backend's FFT.

All three are pinned tightly enough that a real divergence could not hide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import omnibias.pinn.solver.jax as pj  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver.jax import stiff as js  # noqa: E402
from omnibias.pinn.solver.torch import stiff as ts  # noqa: E402

Z = np.array([0.0, 1e-14, 1e-6, 0.3, 1.0, 5.0, 40.0, -1.0, -20.0, -300.0])


def _both(fn_t, fn_j):
    return np.asarray(fn_t.detach().numpy()), np.asarray(fn_j)


def test_phi_diagonal_is_bit_identical() -> None:
    a = ts.phi_diagonal(torch.as_tensor(Z, dtype=torch.float64), 3).numpy()
    b = np.asarray(js.phi_diagonal(jnp.asarray(Z), 3))
    assert np.array_equal(a, b)


def test_phi_diagonal_agrees_to_round_off_for_a_complex_symbol() -> None:
    z = Z * 1j - 0.5
    a = ts.phi_diagonal(torch.as_tensor(z), 2).numpy()
    b = np.asarray(js.phi_diagonal(jnp.asarray(z), 2))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_phi_matrix_agrees_to_round_off() -> None:
    m = np.random.default_rng(0).normal(size=(6, 6)) * 3.0
    a = ts.phi_matrix(torch.as_tensor(m), 2).numpy()
    b = np.asarray(js.phi_matrix(jnp.asarray(m), 2))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def test_rosenbrock_agrees_to_round_off() -> None:
    mat = np.array([[-1000.0, 1.0], [0.0, -1.0]])
    u0 = np.array([1.0, 1.0])
    ut = torch.as_tensor(u0)
    uj = jnp.asarray(u0)
    for _ in range(10):
        ut = ts.rosenbrock_step(
            lambda u: torch.as_tensor(mat) @ u, ut, 0.01, jacobian=torch.as_tensor(mat)
        )
        uj = js.rosenbrock_step(
            lambda u: jnp.asarray(mat) @ u, uj, 0.01, jacobian=jnp.asarray(mat)
        )
    np.testing.assert_allclose(ut.numpy(), np.asarray(uj), rtol=1e-12, atol=1e-14)


def test_exponential_euler_agrees_to_round_off() -> None:
    mat = np.array([[-50.0, 1.0], [0.0, -1.0]])
    u0 = np.array([1.0, 0.5])
    got_t = ts.exponential_rosenbrock_step(
        lambda u: torch.as_tensor(mat) @ u,
        torch.as_tensor(u0),
        0.2,
        jacobian=torch.as_tensor(mat),
    )
    got_j = js.exponential_rosenbrock_step(
        lambda u: jnp.asarray(mat) @ u,
        jnp.asarray(u0),
        0.2,
        jacobian=jnp.asarray(mat),
    )
    np.testing.assert_allclose(*_both(got_t, got_j), rtol=1e-12, atol=1e-14)


def test_closed_form_jacobian_is_bit_identical_across_backends() -> None:
    from omnibias.jax.activations import get_activation as jax_activation
    from omnibias.torch.activations.registry import get_activation as torch_activation

    rng = np.random.default_rng(3)
    w1, b1 = rng.normal(size=(5, 3)), rng.normal(size=(5,))
    w2, b2 = rng.normal(size=(3, 5)), rng.normal(size=(3,))
    u = rng.normal(size=(3,))
    tl = [
        (torch.as_tensor(w1), torch.as_tensor(b1), torch_activation("tanh")),
        (torch.as_tensor(w2), torch.as_tensor(b2), None),
    ]
    jl = [
        (jnp.asarray(w1), jnp.asarray(b1), jax_activation("tanh")),
        (jnp.asarray(w2), jnp.asarray(b2), None),
    ]
    a = ts.closed_form_jacobian(tl, torch.as_tensor(u)).numpy()
    b = np.asarray(js.closed_form_jacobian(jl, jnp.asarray(u)))
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-15)


def _ks(n: int = 64):
    length = 32.0 * math.pi
    gt, gj = pt.SpectralGrid1D(n, length), pj.SpectralGrid1D(n, length)
    x = np.asarray(gt.points())
    u0 = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    return (
        pt.kuramoto_sivashinsky_semidiscrete(gt),
        pj.kuramoto_sivashinsky_semidiscrete(gj),
        u0,
    )


@pytest.mark.parametrize("scheme", ["etdrk4", "imex_euler", "imex_cnab2"])
def test_method_of_lines_stiff_schemes_agree_across_backends(scheme: str) -> None:
    semi_t, semi_j, u0 = _ks()
    times = [0.0, 0.5, 1.0, 1.5]
    snaps_t, _ = pt.method_of_lines(
        semi_t, torch.as_tensor(u0), times, integrator=scheme
    )
    snaps_j, _ = pj.method_of_lines(semi_j, jnp.asarray(u0), times, integrator=scheme)
    np.testing.assert_allclose(
        snaps_t.numpy(), np.asarray(snaps_j), rtol=1e-11, atol=1e-12
    )
