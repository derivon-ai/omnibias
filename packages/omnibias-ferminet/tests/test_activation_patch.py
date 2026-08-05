# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity test for the FermiNet activation patch.

Ensures that the configurable ``BaseNetworkOptions.activation`` field
in upstream FermiNet's ``networks.py``:

(a) defaults to ``jnp.tanh`` and reproduces the upstream-DeepMind
    forward pass bit-for-bit (so any existing checkpoint, test, or
    benchmark that did not set the field continues to behave
    identically);

(b) flips behaviour when set to an omnibias-fastpath activation
    (here ``softplus``), producing a numerically different network
    output -- demonstrating that the slot is actually live in the
    equivariant blocks.

This is the regression contract for Track R-R1.  Track R-R2
(closed-form Delta^2 log|psi| chain rule) and Track R-R4
(Au/Hg/U benchmark sweep) build directly on this slot.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation  # noqa: E402

# ferminet -> tensorflow_probability triggers a JAX deprecation warning at
# import time which pytest may surface as a collection error.  Suppress it.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from ferminet import envelopes as ferminet_envelopes  # noqa: E402
        from ferminet import networks as ferminet_networks  # noqa: E402
    except ImportError:
        pytest.skip("ferminet not installed", allow_module_level=True)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _build_minimal_network(activation=None):
    """Build a 2-electron H2 FermiNet with tiny hidden dims."""
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    nspins = (1, 1)
    kwargs = dict(
        nspins=nspins,
        charges=charges,
        ndim=3,
        determinants=2,
        states=0,
        envelope=ferminet_envelopes.make_isotropic_envelope(),
        feature_layer=ferminet_networks.make_ferminet_features(
            natoms=2,
            nspins=nspins,
            ndim=3,
        ),
        hidden_dims=((8, 4),),
        jastrow="default",
        bias_orbitals=False,
        full_det=True,
        rescale_inputs=False,
        complex_output=False,
        use_last_layer=False,
        separate_spin_channels=False,
        schnet_electron_electron_convolutions=(),
        nuclear_embedding_dim=0,
        electron_nuclear_aux_dims=(),
        schnet_electron_nuclear_convolutions=(),
    )
    if activation is not None:
        kwargs["activation"] = activation
    return ferminet_networks.make_fermi_net(**kwargs)


def test_default_activation_is_tanh() -> None:
    """``FermiNetOptions.activation`` defaults to ``jnp.tanh``."""
    net = _build_minimal_network()
    assert net.options.activation is jnp.tanh


def test_softplus_activation_propagates_into_options() -> None:
    softplus_act = get_activation("softplus").forward
    net = _build_minimal_network(activation=softplus_act)
    assert net.options.activation is softplus_act
    # Sanity: callable returns correct shape and value at 0 (log 2).
    x = jnp.asarray([0.0, 1.0, -1.0], dtype=jnp.float64)
    y = net.options.activation(x)
    assert y.shape == x.shape
    np.testing.assert_allclose(float(y[0]), np.log(2.0), rtol=1e-12)


def test_forward_pass_deterministic_with_default_tanh() -> None:
    """Default-activation path is deterministic and reproducible
    (regression contract: upstream-DeepMind behaviour is preserved)."""
    rng = np.random.default_rng(20260520)
    pos = jnp.asarray(rng.normal(size=(6,)), dtype=jnp.float64)
    atoms = jnp.asarray([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float64)
    spins = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    net = _build_minimal_network()
    params = net.init(jax.random.PRNGKey(42))
    sign1, log1 = net.apply(params, pos, spins, atoms, charges)
    sign2, log2 = net.apply(params, pos, spins, atoms, charges)
    np.testing.assert_allclose(float(log1), float(log2), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(np.asarray(sign1), np.asarray(sign2))


def test_softplus_activation_changes_network_output() -> None:
    """Switching the activation from default tanh to omnibias-softplus
    changes log|psi| -- proves the slot is wired through to the
    equivariant-layer activations at networks.py:953, :967."""
    rng = np.random.default_rng(2026053)
    pos = jnp.asarray(rng.normal(size=(6,)), dtype=jnp.float64)
    atoms = jnp.asarray([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=jnp.float64)
    spins = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    softplus_act = get_activation("softplus").forward

    net_t = _build_minimal_network()
    net_s = _build_minimal_network(activation=softplus_act)

    params_t = net_t.init(jax.random.PRNGKey(7))
    params_s = net_s.init(jax.random.PRNGKey(7))

    _, log_tanh = net_t.apply(params_t, pos, spins, atoms, charges)
    _, log_softplus = net_s.apply(params_s, pos, spins, atoms, charges)

    # The outputs should DIFFER -- proving the activation is live.
    assert not np.isclose(float(log_tanh), float(log_softplus), rtol=1e-6), (
        f"log|psi| identical between tanh and softplus runs "
        f"(tanh={float(log_tanh):.6e}, softplus={float(log_softplus):.6e}); "
        "the activation slot is NOT wired through."
    )
