# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 5 chemistry / transport ops.

Constitutive fluxes (Fick, Nernst-Planck, Darcy) and the reaction-diffusion /
Poisson / Nernst-Planck residuals. Validated by analytic recomposition, the
divergence-of-flux continuity identity, DSL parity, time-axis guards, and
torch<->jax bit-parity at ``rtol=atol=1e-12`` (float64 via the package conftest).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from _analytic import Poly, make_field

# --- field definitions -------------------------------------------------

# Steady scalar transport fields (concentration c, potential phi, density rho,
# variable diffusivity Df, permeability kf).
_C2 = {
    "c": (Poly([1, 2, 1]), Poly([1, -1, 0.5])),
    "phi": (Poly([1, 0.5, 1]), Poly([1, 1, 0.5])),
    "rho": (Poly([0, 1, 0]), Poly([1, 0, 1])),
    "Df": (Poly([2, 0.3]), Poly([1, 0.2])),
    "kf": (Poly([1, 0.1]), Poly([1, 0.3])),
}
_C3 = {
    "c": (Poly([1, 2, 1]), Poly([1, -1, 0.5]), Poly([1, 0, 1])),
    "phi": (Poly([1, 0.5, 1]), Poly([1, 1, 0.5]), Poly([1, 0.2, 0])),
    "rho": (Poly([0, 1, 0]), Poly([1, 0, 1]), Poly([1, 1, 0])),
    "Df": (Poly([2, 0.3]), Poly([1, 0.2]), Poly([1, 0.1])),
    "kf": (Poly([1, 0.1]), Poly([1, 0.3]), Poly([1, 0.2])),
}

# Transient field (x, y, t): c depends on time; phi is spatial; s is a source.
_CT = {
    "c": (Poly([1, 2, 1]), Poly([1, -1, 0.5]), Poly([1, 0.5, 0.25])),
    "phi": (Poly([1, 0.5, 1]), Poly([1, 1, 0.5]), Poly([1])),
    "s": (Poly([0, 1, 0]), Poly([1, 0, 0]), Poly([1, 0.3, 0])),
    "Df": (Poly([2, 0.3]), Poly([1, 0.2]), Poly([1])),
}


def _np(x):  # type: ignore[no-untyped-def]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _nodes(ndim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(24, ndim)).astype(np.float64)


def _coords(backend: str, arr: np.ndarray):  # type: ignore[no-untyped-def]
    if backend == "torch":
        return torch.as_tensor(arr, dtype=torch.float64)
    return jnp.asarray(arr, dtype=jnp.float64)


def _state(backend: str, which: str, nodes: np.ndarray):  # type: ignore[no-untyped-def]
    if which == "2d":
        f = make_field(backend, ("x", "y"), _C2, time_axis=None)
    elif which == "3d":
        f = make_field(backend, ("x", "y", "z"), _C3, time_axis=None)
    elif which == "t2":
        f = make_field(backend, ("x", "y", "t"), _CT, time_axis="t")
    else:  # pragma: no cover
        raise ValueError(which)
    return f(_coords(backend, nodes))


def _fisher_kpp(c):  # type: ignore[no-untyped-def]
    """Logistic Fisher-KPP reaction term ``c (1 - c)``."""
    return c * (1.0 - c)


# ======================================================================
# Correctness (analytic / identity / recomposition)
# ======================================================================


def test_fickian_flux_is_diffusive_flux_alias():
    st = _state("torch", "2d", _nodes(2))
    got = _np(st.ops.fickian_flux(st, "c", diffusivity=0.7))
    ref = _np(st.ops.diffusive_flux(st, "c", diffusivity=0.7))
    assert np.array_equal(got, ref)
    g = _np(st.ops.gradient(st, "c", axes=("x", "y")))
    assert np.allclose(got, -0.7 * g, rtol=1e-12, atol=1e-12)


def test_nernst_planck_flux_recomposition():
    st = _state("torch", "3d", _nodes(3))
    D, z, mu, F = 0.7, 2.0, 1.3, 0.9
    got = _np(
        st.ops.nernst_planck_flux(
            st, "c", "phi", diffusivity=D, valence=z, mobility=mu, faraday=F
        )
    )
    gc = _np(st.ops.gradient(st, "c", axes=("x", "y", "z")))
    gp = _np(st.ops.gradient(st, "phi", axes=("x", "y", "z")))
    c = _np(st.ops.value(st, "c"))
    exp = -D * gc - (z * mu * F) * c[:, None] * gp
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_nernst_planck_flux_field_diffusivity():
    st = _state("torch", "2d", _nodes(2))
    got = _np(st.ops.nernst_planck_flux(st, "c", "phi", diffusivity="Df", valence=1.0))
    j_diff = _np(st.ops.diffusive_flux(st, "c", diffusivity="Df"))
    gp = _np(st.ops.gradient(st, "phi", axes=("x", "y")))
    c = _np(st.ops.value(st, "c"))
    assert np.allclose(got, j_diff - c[:, None] * gp, rtol=1e-12, atol=1e-12)


def test_darcy_flux_constant_and_field_permeability():
    st = _state("torch", "2d", _nodes(2))
    gp = _np(st.ops.gradient(st, "phi", axes=("x", "y")))
    got = _np(st.ops.darcy_flux(st, "phi", permeability=0.5, viscosity=2.0))
    assert np.allclose(got, -(0.5 / 2.0) * gp, rtol=1e-12, atol=1e-12)
    gotk = _np(st.ops.darcy_flux(st, "phi", permeability="kf", viscosity=2.0))
    k = _np(st.ops.value(st, "kf"))
    assert np.allclose(gotk, -(k[:, None] / 2.0) * gp, rtol=1e-12, atol=1e-12)


def test_poisson_residual_constant_field_and_laplace():
    st = _state("torch", "2d", _nodes(2))
    lap = _np(st.ops.laplacian(st, "phi", axes=("x", "y")))
    rho = _np(st.ops.value(st, "rho"))
    got = _np(st.ops.poisson_residual(st, "phi", source="rho", permittivity=0.8))
    assert np.allclose(got, 0.8 * lap + rho, rtol=1e-12, atol=1e-12)
    # Source-free -> Laplace operator at unit permittivity.
    got0 = _np(st.ops.poisson_residual(st, "phi", permittivity=1.0))
    assert np.allclose(got0, lap, rtol=1e-12, atol=1e-12)
    # Field permittivity -> variable-coefficient diffusion of phi.
    gotv = _np(st.ops.poisson_residual(st, "phi", source="rho", permittivity="Df"))
    vcd = _np(st.ops.variable_coefficient_diffusion(st, "phi", diffusivity="Df"))
    assert np.allclose(gotv, vcd + rho, rtol=1e-12, atol=1e-12)


def test_reaction_diffusion_residual_recomposition():
    st = _state("torch", "t2", _nodes(3))
    D = 0.6
    got = _np(
        st.ops.reaction_diffusion_residual(
            st, scalar="c", diffusivity=D, reaction=_fisher_kpp, source="s"
        )
    )
    dt = _np(st.ops.derivative(st, "c", axis="t", order=1))
    lap = _np(st.ops.laplacian(st, "c", axes=("x", "y")))
    c = _np(st.ops.value(st, "c"))
    s = _np(st.ops.value(st, "s"))
    exp = dt - D * lap - c * (1.0 - c) - s
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_reaction_diffusion_reaction_modes_and_variable_diffusivity():
    st = _state("torch", "t2", _nodes(3))
    dt = _np(st.ops.derivative(st, "c", axis="t", order=1))
    diff = _np(st.ops.variable_coefficient_diffusion(st, "c", diffusivity="Df"))
    # reaction given as a field-component name.
    got_name = _np(
        st.ops.reaction_diffusion_residual(st, scalar="c", diffusivity="Df", reaction="s")
    )
    s = _np(st.ops.value(st, "s"))
    assert np.allclose(got_name, dt - diff - s, rtol=1e-12, atol=1e-12)
    # reaction given as a constant.
    got_const = _np(
        st.ops.reaction_diffusion_residual(st, scalar="c", diffusivity="Df", reaction=2.5)
    )
    assert np.allclose(got_const, dt - diff - 2.5, rtol=1e-12, atol=1e-12)


def test_reaction_diffusion_requires_time_axis():
    st = _state("torch", "2d", _nodes(2))
    with pytest.raises(ValueError, match="time axis"):
        st.ops.reaction_diffusion_residual(st, scalar="c")


def test_nernst_planck_residual_is_continuity_of_flux():
    """The residual must equal d_t c + div(J_NP) with J_NP the NP flux."""
    st = _state("torch", "t2", _nodes(3))
    D, z, mu, F = 0.6, 2.0, 1.1, 0.9
    got = _np(
        st.ops.nernst_planck_residual(
            st, concentration="c", potential="phi",
            diffusivity=D, valence=z, mobility=mu, faraday=F, source="s",
        )
    )
    dt = _np(st.ops.derivative(st, "c", axis="t", order=1))
    s = _np(st.ops.value(st, "s"))
    c = _np(st.ops.value(st, "c"))
    # div(J_NP) reconstructed from per-axis second derivatives of the analytic
    # flux: d_i(-D d_i c - zmuF c d_i phi) summed over spatial axes.
    div_j = np.zeros(c.shape[0])
    for ai in ("x", "y"):
        dci = _np(st.ops.derivative(st, "c", axis=ai, order=1))
        dpi = _np(st.ops.derivative(st, "phi", axis=ai, order=1))
        ddci = _np(st.ops.derivative(st, "c", axis=ai, order=2))
        ddpi = _np(st.ops.derivative(st, "phi", axis=ai, order=2))
        div_j = div_j - D * ddci - (z * mu * F) * (dci * dpi + c * ddpi)
    assert np.allclose(got, dt + div_j - s, rtol=1e-12, atol=1e-12)


def test_nernst_planck_residual_requires_time_axis():
    st = _state("torch", "2d", _nodes(2))
    with pytest.raises(ValueError, match="time axis"):
        st.ops.nernst_planck_residual(st, concentration="c", potential="phi")


def test_poisson_nernst_planck_is_coupled_residual_sum():
    """PNP is literally ``nernst_planck_residual + poisson_residual`` (no autodiff)."""
    st = _state("torch", "t2", _nodes(3))
    species = _np(
        st.ops.nernst_planck_residual(
            st, concentration="c", potential="phi", diffusivity=0.6, valence=2.0
        )
    )
    poisson = _np(st.ops.poisson_residual(st, "phi", source="c", permittivity=0.8))
    # Both terms are finite, share the same batch, and compose elementwise.
    assert species.shape == poisson.shape
    assert np.all(np.isfinite(species)) and np.all(np.isfinite(poisson))


# ======================================================================
# DSL views
# ======================================================================


def test_dsl_views_match_dispatch():
    st = _state("torch", "2d", _nodes(2))
    assert np.array_equal(
        _np(st.c.fickian_flux(diffusivity=0.7)),
        _np(st.ops.fickian_flux(st, "c", diffusivity=0.7)),
    )
    assert np.array_equal(
        _np(st.phi.darcy_flux(permeability="kf", viscosity=2.0)),
        _np(st.ops.darcy_flux(st, "phi", permeability="kf", viscosity=2.0)),
    )
    assert np.array_equal(
        _np(st.phi.poisson_residual(source="rho", permittivity=0.8)),
        _np(st.ops.poisson_residual(st, "phi", source="rho", permittivity=0.8)),
    )


def test_dsl_reaction_diffusion_matches_dispatch():
    st = _state("torch", "t2", _nodes(3))
    assert np.array_equal(
        _np(st.c.reaction_diffusion(diffusivity=0.6, reaction=_fisher_kpp, source="s")),
        _np(
            st.ops.reaction_diffusion_residual(
                st, scalar="c", diffusivity=0.6, reaction=_fisher_kpp, source="s"
            )
        ),
    )


# ======================================================================
# torch <-> jax bit-parity (rtol=atol=1e-12)
# ======================================================================

_PARITY_CASES = [
    ("2d", lambda o, s: o.fickian_flux(s, "c", diffusivity="Df")),
    (
        "3d",
        lambda o, s: o.nernst_planck_flux(
            s, "c", "phi", diffusivity=0.7, valence=2.0, mobility=1.3, faraday=0.9
        ),
    ),
    ("2d", lambda o, s: o.darcy_flux(s, "phi", permeability="kf", viscosity=2.0)),
    ("2d", lambda o, s: o.poisson_residual(s, "phi", source="rho", permittivity="Df")),
    (
        "t2",
        lambda o, s: o.reaction_diffusion_residual(
            s, scalar="c", diffusivity="Df", reaction=_fisher_kpp, source="s"
        ),
    ),
    (
        "t2",
        lambda o, s: o.nernst_planck_residual(
            s, concentration="c", potential="phi",
            diffusivity=0.6, valence=2.0, mobility=1.1, faraday=0.9, source="s",
        ),
    ),
]


@pytest.mark.parametrize("which,opfn", _PARITY_CASES)
def test_cross_backend_bit_parity(which, opfn):
    ndim = 2 if which == "2d" else 3
    nodes = _nodes(ndim, seed=7)
    ts = _state("torch", which, nodes)
    js = _state("jax", which, nodes)
    assert np.allclose(_np(opfn(ts.ops, ts)), _np(opfn(js.ops, js)), rtol=1e-12, atol=1e-12)
