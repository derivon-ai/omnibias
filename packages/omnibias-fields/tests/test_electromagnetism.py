# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 6 electromagnetism ops (3-D Maxwell, natural units).

Faraday / Ampere-Maxwell / Gauss residuals, the Poynting vector, potential
reconstructions, the Lorenz-gauge residual, and the vector d'Alembertian.
Validated by analytic recomposition, the structural Maxwell identities
(``div curl A = 0``, ``curl grad phi = 0``, Faraday auto-satisfied by
potentials), DSL parity, and torch<->jax bit-parity at ``rtol=atol=1e-12``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from _analytic import Poly, make_field

# Fields over (x, y, z, t): E, B, vector potential A, current J, scalar phi, rho.
_EM = {
    "Ex": (Poly([1, 2]), Poly([1, -1]), Poly([1, 0.5]), Poly([1, 0.3])),
    "Ey": (Poly([0, 1]), Poly([1, 1]), Poly([1, -0.5]), Poly([1, 0.2])),
    "Ez": (Poly([1, 0.5]), Poly([1, 0.2]), Poly([1, 1]), Poly([1, -0.3])),
    "Bx": (Poly([1, 1]), Poly([1, 0.5]), Poly([0, 1]), Poly([1, 0.4])),
    "By": (Poly([1, -1]), Poly([1, 1]), Poly([1, 0.2]), Poly([1, 0.1])),
    "Bz": (Poly([1, 0.3]), Poly([0, 1]), Poly([1, 0.5]), Poly([1, -0.2])),
    "Ax": (Poly([1, 0.5, 0.2]), Poly([1, 0.1]), Poly([1, 0.3]), Poly([1, 0.2])),
    "Ay": (Poly([1, 0.2]), Poly([1, 0.5, 0.1]), Poly([1, 0.4]), Poly([1, 0.1])),
    "Az": (Poly([1, 0.1]), Poly([1, 0.3]), Poly([1, 0.5, 0.2]), Poly([1, 0.3])),
    "phi": (Poly([1, 0.5, 0.3]), Poly([1, 0.2]), Poly([1, 0.4]), Poly([1, 0.1])),
    "rho": (Poly([1, 1]), Poly([1, 0.2]), Poly([1, 0.1]), Poly([1, 0.3])),
    "Jx": (Poly([0, 1]), Poly([1, 0.1]), Poly([1, 0.2]), Poly([1, 0])),
    "Jy": (Poly([1, 0.1]), Poly([0, 1]), Poly([1, 0.3]), Poly([1, 0])),
    "Jz": (Poly([1, 0.2]), Poly([1, 0.3]), Poly([0, 1]), Poly([1, 0])),
}
_GEM = {
    "E": ("Ex", "Ey", "Ez"),
    "B": ("Bx", "By", "Bz"),
    "A": ("Ax", "Ay", "Az"),
    "J": ("Jx", "Jy", "Jz"),
}
_E = ("Ex", "Ey", "Ez")
_B = ("Bx", "By", "Bz")
_A = ("Ax", "Ay", "Az")
_J = ("Jx", "Jy", "Jz")
_SP = ("x", "y", "z")
_CYC = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))


def _np(x):  # type: ignore[no-untyped-def]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _nodes(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(24, 4)).astype(np.float64)


def _coords(backend: str, arr: np.ndarray):  # type: ignore[no-untyped-def]
    if backend == "torch":
        return torch.as_tensor(arr, dtype=torch.float64)
    return jnp.asarray(arr, dtype=jnp.float64)


def _state(backend: str, nodes: np.ndarray):  # type: ignore[no-untyped-def]
    f = make_field(backend, ("x", "y", "z", "t"), _EM, groups=_GEM, time_axis="t")
    return f(_coords(backend, nodes))


# ======================================================================
# Recomposition (operator == manual composition of primitives)
# ======================================================================


def test_faraday_residual_recomposition():
    st = _state("torch", _nodes())
    got = _np(st.ops.faraday_residual(st, electric=_E, magnetic=_B))
    dB = _np(st.ops.vector_derivative(st, _B, axis="t", order=1))
    cE = _np(st.ops.curl(st, _E))
    assert np.allclose(got, dB + cE, rtol=1e-12, atol=1e-12)


def test_ampere_residual_recomposition_with_and_without_current():
    st = _state("torch", _nodes())
    dE = _np(st.ops.vector_derivative(st, _E, axis="t", order=1))
    cB = _np(st.ops.curl(st, _B))
    got = _np(st.ops.ampere_residual(st, electric=_E, magnetic=_B, current=_J))
    Jv = _np(st.ops.stack_components(st, _J))
    assert np.allclose(got, dE - cB + Jv, rtol=1e-12, atol=1e-12)
    got0 = _np(st.ops.ampere_residual(st, electric=_E, magnetic=_B))
    assert np.allclose(got0, dE - cB, rtol=1e-12, atol=1e-12)


def test_gauss_residual_modes():
    st = _state("torch", _nodes())
    divE = _np(st.ops.divergence(st, _E))
    rho = _np(st.ops.value(st, "rho"))
    assert np.allclose(
        _np(st.ops.gauss_residual(st, electric=_E, charge="rho")), divE - rho,
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        _np(st.ops.gauss_residual(st, electric=_E, charge=2.0)), divE - 2.0,
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(_np(st.ops.gauss_residual(st, electric=_E)), divE, rtol=1e-12, atol=1e-12)


def test_gauss_magnetic_residual_is_divergence():
    st = _state("torch", _nodes())
    assert np.allclose(
        _np(st.ops.gauss_magnetic_residual(st, magnetic=_B)),
        _np(st.ops.divergence(st, _B)),
        rtol=1e-12, atol=1e-12,
    )


def test_poynting_vector_value_and_orthogonality():
    st = _state("torch", _nodes())
    got = _np(st.ops.poynting_vector(st, electric=_E, magnetic=_B))
    e = _np(st.ops.stack_components(st, _E))
    b = _np(st.ops.stack_components(st, _B))
    exp = np.stack(
        [
            e[:, 1] * b[:, 2] - e[:, 2] * b[:, 1],
            e[:, 2] * b[:, 0] - e[:, 0] * b[:, 2],
            e[:, 0] * b[:, 1] - e[:, 1] * b[:, 0],
        ],
        axis=-1,
    )
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)
    # S = E x B is orthogonal to both E and B.
    assert np.allclose((got * e).sum(-1), 0.0, atol=1e-9)
    assert np.allclose((got * b).sum(-1), 0.0, atol=1e-9)


def test_poynting_requires_three_components():
    st = _state("torch", _nodes())
    with pytest.raises(ValueError, match="3-component"):
        st.ops.poynting_vector(st, electric=("Ex", "Ey"), magnetic=_B)


def test_magnetic_field_from_potential_is_curl():
    st = _state("torch", _nodes())
    assert np.allclose(
        _np(st.ops.magnetic_field_from_potential(st, potential=_A)),
        _np(st.ops.curl(st, _A)),
        rtol=1e-12, atol=1e-12,
    )


def test_electric_field_from_potentials_recomposition():
    st = _state("torch", _nodes())
    gphi = _np(st.ops.gradient(st, "phi", axes=_SP))
    dA = _np(st.ops.vector_derivative(st, _A, axis="t", order=1))
    got = _np(
        st.ops.electric_field_from_potentials(
            st, scalar_potential="phi", vector_potential=_A
        )
    )
    assert np.allclose(got, -gphi - dA, rtol=1e-12, atol=1e-12)
    # Electrostatic limit: no vector potential -> E = -grad phi.
    got_es = _np(st.ops.electric_field_from_potentials(st, scalar_potential="phi"))
    assert np.allclose(got_es, -gphi, rtol=1e-12, atol=1e-12)


def test_lorenz_gauge_residual_recomposition():
    st = _state("torch", _nodes())
    dphi = _np(st.ops.derivative(st, "phi", axis="t", order=1))
    divA = _np(st.ops.divergence(st, _A))
    got = _np(
        st.ops.lorenz_gauge_residual(st, scalar_potential="phi", vector_potential=_A)
    )
    assert np.allclose(got, dphi + divA, rtol=1e-12, atol=1e-12)


def test_vector_dalembertian_is_componentwise_box():
    st = _state("torch", _nodes())
    got = _np(st.ops.vector_dalembertian(st, _A, c=1.0))
    exp = np.stack([_np(st.ops.dalembertian(st, n, c=1.0)) for n in _A], axis=-1)
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


# ======================================================================
# Structural Maxwell identities (the differential-geometry backbone)
# ======================================================================


def _d2(st, name, a, b):  # type: ignore[no-untyped-def]
    return _np(st.ops.mixed_partial(st, name, (a, b), (1, 1)))


def test_div_curl_A_is_zero():
    """``div(curl A) = 0`` -> ``B = curl A`` makes the magnetic Gauss law automatic."""
    st = _state("torch", _nodes())
    div_b = np.zeros(24)
    for i, j, k in _CYC:
        # B_i = d_j A_k - d_k A_j; accumulate d_i B_i.
        div_b = div_b + _d2(st, _A[_SP.index(k)], i, j) - _d2(st, _A[_SP.index(j)], i, k)
    assert np.allclose(div_b, 0.0, atol=1e-12)


def test_curl_grad_phi_is_zero():
    """``curl(grad phi) = 0`` -> the electrostatic part of Faraday vanishes."""
    st = _state("torch", _nodes())
    for _i, j, k in _CYC:
        comp = _d2(st, "phi", j, k) - _d2(st, "phi", k, j)
        assert np.allclose(comp, 0.0, atol=1e-12)


def test_faraday_satisfied_by_potentials():
    """With ``B = curl A`` and ``E = -grad phi - d_t A``, Faraday holds identically."""
    st = _state("torch", _nodes())
    for _i, j, k in _CYC:
        ak, aj = _A[_SP.index(k)], _A[_SP.index(j)]
        # d_t B_i = d_t (d_j A_k - d_k A_j)
        dt_bi = _d2(st, ak, "t", j) - _d2(st, aj, "t", k)
        # (curl E)_i = d_j E_k - d_k E_j with E_m = -d_m phi - d_t A_m
        curl_e_i = (
            -_d2(st, "phi", j, k) - _d2(st, _A[_SP.index(k)], j, "t")
            + _d2(st, "phi", k, j) + _d2(st, _A[_SP.index(j)], k, "t")
        )
        assert np.allclose(dt_bi + curl_e_i, 0.0, atol=1e-12)


# ======================================================================
# DSL views
# ======================================================================


def test_dsl_views_match_dispatch():
    st = _state("torch", _nodes())
    assert np.array_equal(
        _np(st.A.dalembertian(c=1.0)),
        _np(st.ops.vector_dalembertian(st, _A, c=1.0)),
    )
    assert np.array_equal(
        _np(st.A.curl),
        _np(st.ops.magnetic_field_from_potential(st, potential=_A)),
    )


# ======================================================================
# torch <-> jax bit-parity (rtol=atol=1e-12)
# ======================================================================

_PARITY_CASES = [
    lambda o, s: o.faraday_residual(s, electric=_E, magnetic=_B),
    lambda o, s: o.ampere_residual(s, electric=_E, magnetic=_B, current=_J),
    lambda o, s: o.gauss_residual(s, electric=_E, charge="rho"),
    lambda o, s: o.gauss_magnetic_residual(s, magnetic=_B),
    lambda o, s: o.poynting_vector(s, electric=_E, magnetic=_B),
    lambda o, s: o.magnetic_field_from_potential(s, potential=_A),
    lambda o, s: o.electric_field_from_potentials(
        s, scalar_potential="phi", vector_potential=_A
    ),
    lambda o, s: o.lorenz_gauge_residual(s, scalar_potential="phi", vector_potential=_A),
    lambda o, s: o.vector_dalembertian(s, _A, c=1.0),
]


@pytest.mark.parametrize("opfn", _PARITY_CASES)
def test_cross_backend_bit_parity(opfn):
    nodes = _nodes(seed=5)
    ts = _state("torch", nodes)
    js = _state("jax", nodes)
    assert np.allclose(_np(opfn(ts.ops, ts)), _np(opfn(js.ops, js)), rtol=1e-12, atol=1e-12)
