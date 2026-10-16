# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sparse FaceNet parity, static topology semantics, JIT and CUDA placement."""
from __future__ import annotations

import numpy as np
import pytest
from omnibias.graph.arrangement._core import ArrangementGraph, _message_topology, node_features
from omnibias.partition.arrangement import Arrangement

_EDGES = [
    (),
    ((0, 1, 0), (1, 2, 1)),
    ((0, 1, 0), (1, 0, 1), (2, 2, 0), (2, 2, 1)),
    tuple((i, j, (i + j) % 2) for i in range(4) for j in range(i, 4)),
    ((-1, 0, 1), (1, 2, 0)),
]


def _graph(edges):
    reps = np.array([[-1.0, -0.7], [-0.8, 0.9], [1.0, 0.6], [0.7, -0.6]])
    arr = Arrangement(np.eye(2), np.zeros(2))
    graph = ArrangementGraph(tuple(map(tuple, np.sign(reps).astype(int))), edges, reps)
    return graph, arr


def _parameters():
    rng = np.random.default_rng(3141)
    return tuple(rng.normal(size=shape) * 0.2 for shape in ((3, 4), (3,), (3, 4), (3,), (1, 3), (1,)))


def _dense_reference(graph, features, weights, ops, constant, rounds):
    """Historical dense adjacency assignment and scalar neighbor traversal."""
    lift_w, lift_b, msg_w, msg_b, out_w, out_b = weights
    h = ops.tanh(features @ lift_w.T + lift_b)
    n = features.shape[0]
    adjacency = np.zeros((n, n))
    crossed = np.zeros((n, n))
    for u, v, k in graph.edges:
        adjacency[u, v] = adjacency[v, u] = 1.0
        crossed[u, v] = crossed[v, u] = float(k)
    for _ in range(rounds):
        messages = []
        for i in range(n):
            total = ops.zeros_like(h[i])
            degree = 0
            for j in range(n):
                if adjacency[i, j] <= 0:
                    continue
                cat = ops.concatenate([h[j], constant([crossed[i, j]])])
                total = total + ops.tanh(cat @ msg_w.T + msg_b)
                degree += 1
            messages.append(total / max(degree, 1))
        h = h + ops.stack(messages)
    return (h @ out_w.T + out_b).reshape(-1)


def test_topology_duplicate_overwrite_self_edge_and_neighbor_order():
    graph, _ = _graph(_EDGES[2])
    receivers, senders, labels, degrees = _message_topology(graph, 4)
    assert receivers == (0, 1, 2)
    assert senders == (1, 0, 2)
    assert labels == (1.0, 1.0, 1.0)
    assert degrees == (1, 1, 1, 1)


@pytest.mark.parametrize("edge,error", [((0.5, 1, 0), TypeError), ((4, 0, 0), IndexError),
                                        ((-5, 0, 0), IndexError)])
def test_topology_rejects_nonintegral_or_out_of_range_endpoints(edge, error):
    graph, _ = _graph((edge,))
    with pytest.raises(error):
        _message_topology(graph, 4)


@pytest.mark.parametrize("edges", _EDGES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jax_jit_and_complete_parameter_jacobians_match_dense(edges, dtype):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.graph.arrangement.jax import facenet_apply

    graph, arr = _graph(edges)
    weights = tuple(jnp.asarray(w, dtype=getattr(jnp, dtype)) for w in _parameters())
    features = jnp.asarray(node_features(arr, graph, beta=2.0), dtype=weights[0].dtype)
    def actual(p):
        return facenet_apply(graph, arr, lift_w=p[0], lift_b=p[1], msg_w=p[2],
                             msg_b=p[3], out_w=p[4], out_b=p[5], rounds=2, beta=2.0)
    def reference(p):
        return _dense_reference(graph, features, p, jnp,
                                lambda x: jnp.asarray(x, dtype=p[0].dtype), 2)
    atol = 3e-6 if dtype == "float32" else 2e-13
    compiled = jax.jit(actual)
    for shift in (0.0, 0.07):
        changed = tuple(w + shift for w in weights)
        np.testing.assert_allclose(compiled(changed), reference(changed), rtol=atol, atol=atol)
    got = jax.jit(jax.jacrev(actual))(weights)
    expected = jax.jacrev(reference)(weights)
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=atol, atol=atol)


@pytest.mark.parametrize("edges", _EDGES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_values_and_complete_parameter_jacobians_match_dense(edges, dtype):
    torch = pytest.importorskip("torch")
    from omnibias.graph.arrangement.torch import FaceNet

    graph, arr = _graph(edges)
    net = FaceNet(2, 3, 2, beta=2.0, dtype=getattr(torch, dtype))
    weights = tuple(torch.tensor(w, dtype=getattr(torch, dtype), requires_grad=True) for w in _parameters())
    names = tuple(name for name, _ in net.named_parameters())
    features = torch.tensor(node_features(arr, graph, beta=2.0), dtype=weights[0].dtype)
    def actual(*p):
        return torch.func.functional_call(net, dict(zip(names, p, strict=True)), (graph, arr))
    def reference(*p):
        return _dense_reference(graph, features, p, torch,
                                lambda x: torch.tensor(x, dtype=p[0].dtype), 2)
    atol = 3e-6 if dtype == "float32" else 2e-13
    for shift in (0.0, 0.07):
        changed = tuple(w + shift for w in weights)
        np.testing.assert_allclose(actual(*changed).detach(), reference(*changed).detach(), rtol=atol, atol=atol)
    got = torch.autograd.functional.jacobian(actual, weights)
    expected = torch.autograd.functional.jacobian(reference, weights)
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=atol, atol=atol)
    if not edges:
        grads = torch.autograd.grad(actual(*weights).sum(), weights, allow_unused=True)
        assert grads[2] is None and grads[3] is None


@pytest.mark.parametrize("edges", [_EDGES[0], _EDGES[2], _EDGES[3]])
def test_torch_cuda_matches_independent_cpu_dense_values_and_gradients(edges):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a usable CUDA runtime is unavailable")
    from omnibias.graph.arrangement.torch import FaceNet

    graph, arr = _graph(edges)
    net = FaceNet(2, 3, 2, beta=2.0, dtype=torch.float64).cuda()
    weights = tuple(torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in _parameters())
    with torch.no_grad():
        for parameter, value in zip(net.parameters(), weights, strict=True):
            parameter.copy_(value)
    features = torch.tensor(node_features(arr, graph, beta=2.0), dtype=torch.float64)
    reference = _dense_reference(graph, features, weights, torch,
                                 lambda x: torch.tensor(x, dtype=torch.float64), 2)
    actual = net(graph, arr)
    assert actual.is_cuda
    np.testing.assert_allclose(actual.detach().cpu(), reference.detach(), rtol=2e-12, atol=2e-12)
    cotangent = torch.tensor([0.25, -1.0, 0.7, 0.1], dtype=torch.float64)
    got = torch.autograd.grad((actual * cotangent.cuda()).sum(), tuple(net.parameters()), allow_unused=True)
    expected = torch.autograd.grad((reference * cotangent).sum(), weights, allow_unused=True)
    for a, b in zip(got, expected, strict=True):
        if b is None:
            assert a is None
        else:
            assert a.is_cuda
            np.testing.assert_allclose(a.cpu(), b, rtol=2e-12, atol=2e-12)
