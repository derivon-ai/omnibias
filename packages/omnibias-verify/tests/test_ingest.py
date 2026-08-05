# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""torch / jax weight ingestion + cross-backend parity (bit-identical Networks)."""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.verify import Network, reachable_box

# Shared float64 weights for a 2 -> 3 -> 2 tanh MLP.
_W0 = [[0.8, -0.5], [0.3, 0.9], [-0.6, 0.2]]
_B0 = [0.1, -0.2, 0.05]
_W1 = [[0.5, -0.4, 0.7], [0.2, 0.6, -0.3]]
_B1 = [0.0, 0.1]


def _torch_network() -> tuple[Network, object]:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    seq = nn.Sequential(
        nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)
    ).double()
    with torch.no_grad():
        seq[0].weight.copy_(torch.tensor(_W0, dtype=torch.float64))
        seq[0].bias.copy_(torch.tensor(_B0, dtype=torch.float64))
        seq[2].weight.copy_(torch.tensor(_W1, dtype=torch.float64))
        seq[2].bias.copy_(torch.tensor(_B1, dtype=torch.float64))
    from omnibias.verify.torch import network_from_sequential

    return network_from_sequential(seq), seq


def _jax_network() -> Network:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.verify.jax import network_from_params

    params = [
        (jnp.asarray(_W0), jnp.asarray(_B0)),
        (jnp.asarray(_W1), jnp.asarray(_B1)),
    ]
    return network_from_params(params, activation="tanh")


def _grid(box: Sequence[Interval], n: int) -> itertools.product:  # type: ignore[type-arg]
    axes = [[iv.lo + (iv.hi - iv.lo) * k / (n - 1) for k in range(n)] for iv in box]
    return itertools.product(*axes)


def test_torch_ingest_roundtrip() -> None:
    net, _ = _torch_network()
    assert len(net) == 3
    assert net.layers[0].weight == tuple(tuple(r) for r in _W0)
    assert net.layers[0].bias == tuple(_B0)


def test_jax_ingest_roundtrip() -> None:
    net = _jax_network()
    assert len(net) == 3
    assert net.layers[2].weight == tuple(tuple(r) for r in _W1)


def test_cross_backend_networks_are_bit_identical() -> None:
    net_t, _ = _torch_network()
    net_j = _jax_network()
    assert net_t == net_j  # frozen dataclasses -> value equality on float64 weights


def test_cross_backend_certificates_match() -> None:
    net_t, _ = _torch_network()
    net_j = _jax_network()
    box = [Interval(-0.5, 0.5), Interval(-0.5, 0.5)]
    reach_t = reachable_box(net_t, box, order=2, max_boxes=32)
    reach_j = reachable_box(net_j, box, order=2, max_boxes=32)
    for it, ij in zip(reach_t, reach_j, strict=True):
        assert it.lo == ij.lo and it.hi == ij.hi


def test_verifier_contains_torch_forward() -> None:
    torch = pytest.importorskip("torch")
    net, seq = _torch_network()
    box = [Interval(-0.6, 0.6), Interval(-0.6, 0.6)]
    reach = reachable_box(net, box, order=3, max_boxes=64)
    for pt in _grid(box, 9):
        with torch.no_grad():
            out = seq(torch.tensor(pt, dtype=torch.float64)).tolist()
        for val, iv in zip(out, reach, strict=True):
            assert iv.lo - 1e-9 <= val <= iv.hi + 1e-9


def test_verifier_contains_torch_forward_grid_and_random() -> None:
    """Founding delta->0 soundness rule: reachable_box contains the ingested
    torch forward output at a dense grid AND a random sample in the box."""
    torch = pytest.importorskip("torch")
    net, seq = _torch_network()
    box = [Interval(-0.6, 0.6), Interval(-0.6, 0.6)]
    reach = reachable_box(net, box, order=3, max_boxes=64)
    rng = random.Random(5)
    pts = [tuple(pt) for pt in _grid(box, 9)]
    pts.extend((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)) for _ in range(60))
    for pt in pts:
        with torch.no_grad():
            out = seq(torch.tensor(list(pt), dtype=torch.float64)).tolist()
        for val, iv in zip(out, reach, strict=True):
            assert iv.lo - 1e-9 <= val <= iv.hi + 1e-9


def test_unsupported_torch_layer_raises() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from omnibias.verify.torch import network_from_sequential

    with pytest.raises(TypeError):
        network_from_sequential(nn.Sequential(nn.Conv2d(1, 1, 3)))


def test_unsupported_jax_activation_raises() -> None:
    pytest.importorskip("jax")
    from omnibias.verify.jax import network_from_params

    with pytest.raises(ValueError):
        network_from_params([([[1.0]], [0.0])], activation="swish")
