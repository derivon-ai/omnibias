# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch <-> JAX lattice parity.

The *deterministic* kernels in :mod:`omnibias.geometry.gauge.lattice._core.kernels` are
bit-identical twins on a fixed link config (``rtol=1e-9, atol=1e-11`` in x64).
The full stochastic Monte-Carlo only agrees *statistically* (the RNG streams
differ between ``torch.Generator`` and ``jax.random``), so that is asserted with
a statistical margin, not to ULP.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.geometry.gauge.lattice._core import kernels  # noqa: E402


def _fixed_links(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((4, *shape, 4))
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


@pytest.fixture
def configs() -> tuple[np.ndarray, torch.Tensor, jax.Array]:
    arr = _fixed_links((4, 4, 4, 4), 1234)
    return arr, torch.tensor(arr, dtype=torch.float64), jnp.asarray(arr)


def _close(t_val: object, j_val: object, *, rtol: float = 1e-9, atol: float = 1e-11) -> bool:
    return bool(np.allclose(np.asarray(t_val), np.asarray(j_val), rtol=rtol, atol=atol))


def test_quat_and_staple_parity(configs) -> None:
    _, tl, jl = configs
    assert _close(kernels.staple_sum(torch, tl, 0), kernels.staple_sum(jnp, jl, 0))
    assert _close(
        kernels.quat_mul(torch, tl[0], tl[1]), kernels.quat_mul(jnp, jl[0], jl[1])
    )
    # SU(2) matrix map (complex): np.allclose handles complex arrays directly.
    assert _close(
        kernels.quat_to_matrix(torch, tl[0]), kernels.quat_to_matrix(jnp, jl[0])
    )


def test_loop_observable_parity(configs) -> None:
    _, tl, jl = configs
    for mu in range(4):
        for nu in range(mu + 1, 4):
            assert _close(
                kernels.plaquette_trace(torch, tl, mu, nu),
                kernels.plaquette_trace(jnp, jl, mu, nu),
            )
    assert _close(
        kernels.wilson_loop_trace(torch, tl, 0, 2, 2),
        kernels.wilson_loop_trace(jnp, jl, 0, 2, 2),
    )
    assert _close(
        kernels.polyakov_loop_field(torch, tl), kernels.polyakov_loop_field(jnp, jl)
    )


def test_smear_and_glueball_parity(configs) -> None:
    _, tl, jl = configs
    ts = kernels.ape_smear_spatial_links(torch, tl, n_steps=3, alpha=0.5)
    js = kernels.ape_smear_spatial_links(jnp, jl, n_steps=3, alpha=0.5)
    assert _close(ts, js)
    assert _close(
        kernels.glueball_operator_timeslice(torch, tl, n_smear=4),
        kernels.glueball_operator_timeslice(jnp, jl, n_smear=4),
    )


def test_correlator_and_gevp_parity() -> None:
    rng = np.random.default_rng(7)
    o_samples = rng.standard_normal((60, 2, 8))
    t_corr, t_err = kernels.connected_correlator_matrix_ensemble(
        torch, torch.tensor(o_samples, dtype=torch.float64)
    )
    j_corr, j_err = kernels.connected_correlator_matrix_ensemble(jnp, jnp.asarray(o_samples))
    assert _close(t_corr, j_corr, rtol=1e-8, atol=1e-10)
    assert _close(t_err, j_err, rtol=1e-8, atol=1e-10)

    c0 = np.asarray(t_corr[:, :, 0])
    c1 = np.asarray(t_corr[:, :, 1])
    lam_t = kernels.gevp_ground_lambda(torch, torch.tensor(c0), torch.tensor(c1))
    lam_j = kernels.gevp_ground_lambda(jnp, jnp.asarray(c0), jnp.asarray(c1))
    assert lam_t == pytest.approx(lam_j, rel=1e-8)


def test_orbit_distance_parity(configs) -> None:
    arr, tl, jl = configs
    other = _fixed_links((4, 4, 4, 4), 9999)
    d_t = kernels.gauge_orbit_distance(torch, tl, torch.tensor(other, dtype=torch.float64))
    d_j = kernels.gauge_orbit_distance(jnp, jl, jnp.asarray(other))
    assert d_t == pytest.approx(d_j, rel=1e-8)


@pytest.mark.slow
def test_full_mc_statistical_agreement() -> None:
    """Full MC agrees statistically (different RNG streams, not bit-identical)."""
    from omnibias.geometry.gauge.lattice.jax import run_lattice_mc as jax_mc
    from omnibias.geometry.gauge.lattice.montecarlo import run_lattice_mc as torch_mc

    kw = dict(lattice_shape=(4, 4, 4, 4), beta=2.0, n_therm=30, n_meas=30, n_sep=1)
    t_out = torch_mc(seed=1, **kw)
    j_out = jax_mc(seed=1, **kw)
    tol = 0.05 + t_out["avg_plaquette_err"] + j_out["avg_plaquette_err"]
    assert abs(t_out["avg_plaquette"] - j_out["avg_plaquette"]) < tol
    assert 0.0 < t_out["avg_plaquette"] < 1.0
    assert 0.0 < j_out["avg_plaquette"] < 1.0
