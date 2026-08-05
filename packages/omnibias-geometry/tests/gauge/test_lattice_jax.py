# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fast JAX-only smoke and correctness tests for the SU(2) lattice backend."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
# Lattice MC needs float64 for GEVP / correlator stability.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402, I001

from omnibias.geometry.gauge.lattice.jax import (  # noqa: E402
    average_plaquette,
    identity_links,
    quat_to_matrix,
    random_links,
    run_lattice_mc,
    sweep,
)

_REQUIRED_MC_KEYS = {
    "gauge_group",
    "lattice_shape",
    "beta",
    "n_therm",
    "n_meas",
    "n_sep",
    "device",
    "avg_plaquette",
    "avg_plaquette_err",
    "avg_polyakov",
    "avg_polyakov_err",
    "glueball_correlator",
    "glueball_correlator_err",
    "smearing",
    "wilson_loops",
    "creutz_ratios",
    "string_tension",
    "gevp",
    "gevp_plateau",
    "evidence_note",
    "seed",
    "elapsed_s",
}


def _assert_links_unit_quaternions(links: jnp.ndarray) -> None:
    norm = jnp.linalg.norm(links, axis=-1)
    assert jnp.allclose(norm, jnp.ones_like(norm), atol=1e-10)


def _assert_links_su2(links: jnp.ndarray) -> None:
    _assert_links_unit_quaternions(links)
    mats = quat_to_matrix(links)
    ud = jnp.conj(mats).swapaxes(-2, -1)
    prod = mats @ ud
    eye = jnp.eye(2, dtype=jnp.complex128)
    assert jnp.allclose(prod, eye, atol=1e-10, rtol=0.0)
    det = mats[..., 0, 0] * mats[..., 1, 1] - mats[..., 0, 1] * mats[..., 1, 0]
    assert jnp.allclose(det.real, jnp.ones_like(det.real), atol=1e-10, rtol=0.0)


def test_random_links_are_unit_quaternions() -> None:
    shape = (4, 4, 4, 4)
    key = jax.random.PRNGKey(11)
    links = random_links(shape, key)
    _assert_links_unit_quaternions(links)


def test_sweep_preserves_su2_constraints() -> None:
    shape = (4, 4, 4, 4)
    key = jax.random.PRNGKey(0)
    key, init_key, sweep_key = jax.random.split(key, 3)
    links = random_links(shape, init_key)
    _assert_links_su2(links)
    links = sweep(links, beta=2.0, key=sweep_key)
    _assert_links_su2(links)


def test_average_plaquette_identity_and_random() -> None:
    cold = identity_links((4, 4, 4, 4))
    assert average_plaquette(cold) == 1.0

    key = jax.random.PRNGKey(5)
    links = random_links((4, 4, 4, 4), key)
    p = average_plaquette(links)
    assert 0.0 < p < 1.0


@pytest.mark.slow
def test_run_lattice_mc_deterministic() -> None:
    kwargs = dict(
        lattice_shape=(4, 4, 4, 4),
        beta=2.0,
        n_therm=8,
        n_meas=5,
        n_sep=1,
        seed=42,
    )
    out_a = run_lattice_mc(**kwargs)
    out_b = run_lattice_mc(**kwargs)
    assert out_a["avg_plaquette"] == out_b["avg_plaquette"]

    out_other = run_lattice_mc(**{**kwargs, "seed": 99})
    assert out_a["avg_plaquette"] != out_other["avg_plaquette"]


@pytest.mark.slow
def test_run_lattice_mc_strong_coupling_sanity() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=1.0,
        n_therm=30,
        n_meas=20,
        n_sep=1,
        seed=43,
    )
    p = out["avg_plaquette"]
    err = out["avg_plaquette_err"]
    assert 0.0 < p < 1.0
    assert abs(p - 0.25) < 0.15 + err


@pytest.mark.slow
def test_run_lattice_mc_returns_expected_keys() -> None:
    out = run_lattice_mc(
        lattice_shape=(4, 4, 4, 4),
        beta=2.0,
        n_therm=5,
        n_meas=3,
        n_sep=1,
        seed=0,
    )
    assert set(out.keys()) == _REQUIRED_MC_KEYS
    assert out["device"] == jax.default_backend()
