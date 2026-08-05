# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the non-elementwise jet primitives (torch).

``compose_jet_mv`` only reaches *elementwise* maps. These four primitives extend
the reachable class to rational and coupled ones -- ``1/u``, ``exp``, softmax over
an axis, and the attention block built from them -- so each is checked against the
oracle that matters:

* :func:`jet_reciprocal` / :func:`jet_exp` against nested ``torch.func.jacfwd``.
* :func:`jet_softmax` against AD *and* against the structural invariant that only
  the constant coefficient of a partition of unity is non-zero (every derivative
  of ``sum_j p_j = 1`` must vanish identically).
* :func:`jet_attention` against AD, and its value against
  :func:`omnibias.hopfield.torch.ops.attention` -- the point of the exercise being
  that omnibias-hopfield differentiates the same block with respect to the
  *scores*, while this gives the coordinate derivatives a PDE residual needs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.multi_index import multi_indices  # noqa: E402
from omnibias.torch.jet_mv import (  # noqa: E402
    identity_jet,
    jet_attention,
    jet_exp,
    jet_multiply,
    jet_partials,
    jet_reciprocal,
    jet_softmax,
    mlp_jet_mv,
)
from torch.func import jacfwd  # noqa: E402

DIM = 3
ORDER = 3


@pytest.fixture(autouse=True)
def _default_float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@pytest.fixture
def memory():
    rng = np.random.default_rng(7)
    return {
        "W": torch.as_tensor(rng.normal(scale=0.7, size=(4, DIM))),
        "b": torch.as_tensor(rng.normal(scale=0.3, size=(4,))),
        "K": torch.as_tensor(rng.normal(scale=0.8, size=(5, 4))),
        "V": torch.as_tensor(rng.normal(scale=1.1, size=(5, 2))),
        "x0": torch.as_tensor(rng.normal(size=(DIM,))),
    }


def _ad_partials(f, x0, order: int = ORDER):
    """``{alpha: D^alpha f(x0)}`` by nested forward-mode AD."""
    out = {}
    derivs = [f]
    for _ in range(order):
        derivs.append(jacfwd(derivs[-1]))
    for alpha in multi_indices(DIM, order):
        k = sum(alpha)
        if k == 0:
            out[alpha] = f(x0)
            continue
        axes = [i for i, a in enumerate(alpha) for _ in range(a)]
        val = derivs[k](x0)
        # Trailing axes are the successive differentiation directions, applied
        # outermost-last; index them in reverse.
        for ax in reversed(axes):
            val = val[..., ax]
        out[alpha] = val
    return out


# ----------------------------- reciprocal / exp -----------------------------


def test_reciprocal_matches_autodiff(memory) -> None:
    W, b, x0 = memory["W"], memory["b"], memory["x0"]

    def f(x):
        return 1.0 / (2.5 + torch.tanh(W @ x + b))

    jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    shifted = torch.cat([(jet[0] + 2.5).unsqueeze(0), jet[1:]], dim=0)
    got = jet_partials(jet_reciprocal(shifted, DIM, ORDER), DIM, ORDER)
    want = _ad_partials(f, x0)
    for alpha, value in want.items():
        assert torch.allclose(got[alpha], value, rtol=1e-11, atol=1e-11), alpha


def test_reciprocal_times_original_is_one(memory) -> None:
    """The defining identity, at *every* order: ``u * (1/u) = 1``."""
    W, b, x0 = memory["W"], memory["b"], memory["x0"]
    jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    shifted = torch.cat([(jet[0] + 2.5).unsqueeze(0), jet[1:]], dim=0)
    product = jet_multiply(shifted, jet_reciprocal(shifted, DIM, ORDER), DIM, ORDER)
    assert torch.allclose(product[0], torch.ones_like(product[0]), atol=1e-14)
    assert float(product[1:].abs().max()) < 1e-13


def test_exp_matches_autodiff(memory) -> None:
    W, b, x0 = memory["W"], memory["b"], memory["x0"]

    def f(x):
        return torch.exp(torch.tanh(W @ x + b))

    jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    got = jet_partials(jet_exp(jet, DIM, ORDER), DIM, ORDER)
    want = _ad_partials(f, x0)
    for alpha, value in want.items():
        assert torch.allclose(got[alpha], value, rtol=1e-11, atol=1e-11), alpha


# --------------------------------- softmax ----------------------------------


def test_softmax_matches_autodiff(memory) -> None:
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]

    def f(x):
        return torch.softmax(K @ torch.tanh(W @ x + b), dim=-1)

    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K, None, None)], ORDER)
    got = jet_partials(jet_softmax(scores, DIM, ORDER), DIM, ORDER)
    want = _ad_partials(f, x0)
    for alpha, value in want.items():
        assert torch.allclose(got[alpha], value, rtol=1e-11, atol=1e-11), alpha


def test_softmax_is_a_partition_of_unity_at_every_order(memory) -> None:
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K, None, None)], ORDER)
    p = jet_softmax(scores, DIM, ORDER)
    rows = p.sum(dim=-1)
    assert abs(float(rows[0]) - 1.0) < 1e-14
    assert float(rows[1:].abs().max()) < 1e-14


def test_softmax_is_shift_invariant(memory) -> None:
    """A constant added to every score changes nothing -- including the jet."""
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K, None, None)], ORDER)
    bumped = torch.cat([(scores[0] + 40.0).unsqueeze(0), scores[1:]], dim=0)
    assert torch.allclose(
        jet_softmax(scores, DIM, ORDER), jet_softmax(bumped, DIM, ORDER), atol=1e-14
    )


def test_softmax_survives_scores_that_overflow_a_naive_exp(memory) -> None:
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    scores = mlp_jet_mv(x0, [(W, b, "tanh"), (K * 400.0, None, None)], ORDER)
    p = jet_softmax(scores, DIM, ORDER)
    assert bool(torch.isfinite(p).all())
    assert abs(float(p[0].sum()) - 1.0) < 1e-14


def test_softmax_rejects_a_scalar_jet(memory) -> None:
    x0 = memory["x0"]
    with pytest.raises(ValueError, match="trailing softmax axis"):
        jet_softmax(identity_jet(x0, ORDER)[:, 0], DIM, ORDER)


# -------------------------------- attention ---------------------------------


def test_attention_matches_autodiff(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    beta = 1.7

    def f(x):
        q = torch.tanh(W @ x + b)
        return torch.softmax(beta * (K @ q), dim=-1) @ V

    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    got = jet_partials(
        jet_attention(q_jet, K, V, DIM, ORDER, beta=beta), DIM, ORDER
    )
    want = _ad_partials(f, x0)
    for alpha, value in want.items():
        assert torch.allclose(got[alpha], value, rtol=1e-11, atol=1e-11), alpha


def test_attention_value_matches_omnibias_hopfield(memory) -> None:
    """The block *is* hopfield attention; this module only adds ``d/dx``."""
    hopfield = pytest.importorskip("omnibias.hopfield.torch.ops")
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    beta = 0.8
    q = torch.tanh(W @ x0 + b)
    reference = hopfield.attention(q.unsqueeze(0), K, V, beta=beta).squeeze(0)
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    got = jet_attention(q_jet, K, V, DIM, ORDER, beta=beta)[0]
    assert torch.allclose(got, reference, atol=1e-14)


def test_attention_output_lies_in_the_convex_hull_of_the_values(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    out = jet_attention(q_jet, K, V, DIM, ORDER)[0]
    assert bool((out >= V.min(dim=0).values - 1e-12).all())
    assert bool((out <= V.max(dim=0).values + 1e-12).all())


def test_attention_temperature_may_be_a_trainable_tensor(memory) -> None:
    W, b, K, V, x0 = (memory[k] for k in ("W", "b", "K", "V", "x0"))
    beta = torch.tensor(1.3, requires_grad=True)
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    out = jet_attention(q_jet, K, V, DIM, ORDER, beta=beta)
    out.sum().backward()
    assert beta.grad is not None
    assert float(beta.grad.abs()) > 0.0


def test_attention_sharpens_towards_one_slot_as_beta_grows(memory) -> None:
    """Temperature collapse (the feasibility sense): the mixture hardens."""
    W, b, K, x0 = memory["W"], memory["b"], memory["K"], memory["x0"]
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    peaks = []
    for beta in (0.5, 5.0, 50.0):
        scores = jet_attention(
            q_jet, K, torch.eye(K.shape[0], dtype=K.dtype), DIM, ORDER, beta=beta
        )
        peaks.append(float(scores[0].max()))
    assert peaks[0] < peaks[1] < peaks[2]
    assert peaks[-1] > 0.99


@pytest.mark.parametrize(
    ("keys", "values", "match"),
    [
        (torch.zeros(3), torch.zeros(3, 2), "must be 2-D"),
        (torch.zeros(3, 4), torch.zeros(2, 2), "share the memory axis"),
        (torch.zeros(3, 9), torch.zeros(3, 2), "key width"),
    ],
)
def test_attention_validates_shapes(memory, keys, values, match) -> None:
    W, b, x0 = memory["W"], memory["b"], memory["x0"]
    q_jet = mlp_jet_mv(x0, [(W, b, "tanh")], ORDER)
    with pytest.raises(ValueError, match=match):
        jet_attention(q_jet, keys.double(), values.double(), DIM, ORDER)


def test_row_count_is_validated(memory) -> None:
    x0 = memory["x0"]
    jet = identity_jet(x0, ORDER)
    with pytest.raises(ValueError, match="requires"):
        jet_reciprocal(jet, DIM, ORDER + 1)
    with pytest.raises(ValueError, match="requires"):
        jet_softmax(jet, DIM, ORDER + 1)
