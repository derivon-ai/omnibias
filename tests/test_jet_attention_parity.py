# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for the non-elementwise jet primitives.

``jet_reciprocal`` / ``jet_exp`` / ``jet_softmax`` / ``jet_attention`` are written
as the same sequence of operations on both backends -- the same tower, the same
max-shift, the same Cauchy-product table -- so with identical float64 inputs the
whole coefficient array must agree to ``1e-13``. The reciprocal tower in
particular is built by repeated multiplication rather than ``pow`` precisely so
the two backends execute the identical arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.jet_mv import jet_attention as jax_attention  # noqa: E402
from omnibias.jax.jet_mv import jet_exp as jax_exp  # noqa: E402
from omnibias.jax.jet_mv import jet_reciprocal as jax_reciprocal  # noqa: E402
from omnibias.jax.jet_mv import jet_softmax as jax_softmax  # noqa: E402
from omnibias.jax.jet_mv import mlp_jet_mv as jax_jet  # noqa: E402
from omnibias.torch.jet_mv import jet_attention as torch_attention  # noqa: E402
from omnibias.torch.jet_mv import jet_exp as torch_exp  # noqa: E402
from omnibias.torch.jet_mv import jet_reciprocal as torch_reciprocal  # noqa: E402
from omnibias.torch.jet_mv import jet_softmax as torch_softmax  # noqa: E402
from omnibias.torch.jet_mv import mlp_jet_mv as torch_jet  # noqa: E402

DIM = 3
TOL = 1e-13


@pytest.fixture(params=[1, 2, 3, 4], ids=lambda n: f"order{n}")
def order(request) -> int:
    return request.param


@pytest.fixture
def shared():
    rng = np.random.default_rng(19)
    return {
        "W": rng.normal(scale=0.7, size=(4, DIM)),
        "b": rng.normal(scale=0.3, size=(4,)),
        "K": rng.normal(scale=0.9, size=(6, 4)),
        "V": rng.normal(scale=1.2, size=(6, 2)),
        "x0": rng.normal(size=(DIM,)),
    }


def _jets(shared, order: int):
    t = torch_jet(
        torch.as_tensor(shared["x0"]),
        [(torch.as_tensor(shared["W"]), torch.as_tensor(shared["b"]), "tanh")],
        order,
    )
    j = jax_jet(
        jnp.asarray(shared["x0"]),
        [(jnp.asarray(shared["W"]), jnp.asarray(shared["b"]), "tanh")],
        order,
    )
    return t, j


def _agree(t, j) -> bool:
    return np.allclose(t.detach().numpy(), np.asarray(j), rtol=TOL, atol=TOL)


def test_reciprocal_parity(shared, order: int) -> None:
    t, j = _jets(shared, order)
    t_shift = torch.cat([(t[0] + 2.5).unsqueeze(0), t[1:]], dim=0)
    j_shift = jnp.concatenate([(j[0] + 2.5)[None], j[1:]], axis=0)
    assert _agree(
        torch_reciprocal(t_shift, DIM, order), jax_reciprocal(j_shift, DIM, order)
    )


def test_exp_parity(shared, order: int) -> None:
    t, j = _jets(shared, order)
    assert _agree(torch_exp(t, DIM, order), jax_exp(j, DIM, order))


def test_softmax_parity(shared, order: int) -> None:
    t, j = _jets(shared, order)
    t_s = torch.tensordot(t, torch.as_tensor(shared["K"]), dims=([-1], [-1]))
    j_s = jnp.tensordot(j, jnp.asarray(shared["K"]), axes=([-1], [-1]))
    assert _agree(torch_softmax(t_s, DIM, order), jax_softmax(j_s, DIM, order))


def test_attention_parity(shared, order: int) -> None:
    t, j = _jets(shared, order)
    got_t = torch_attention(
        t,
        torch.as_tensor(shared["K"]),
        torch.as_tensor(shared["V"]),
        DIM,
        order,
        beta=1.4,
    )
    got_j = jax_attention(
        j, jnp.asarray(shared["K"]), jnp.asarray(shared["V"]), DIM, order, beta=1.4
    )
    assert _agree(got_t, got_j)


def test_attention_stays_finite_when_scores_saturate(shared, order: int) -> None:
    """A naive ``exp`` would overflow at ``beta = 300``; the max-shift must not.

    Only the value is compared across backends here. The higher coefficients are
    genuinely ill-conditioned at extreme temperature -- they carry ``beta^k``
    factors against softmax weights of order ``1e-60``, so the two backends'
    summation orders disagree in the surviving digits. That is arithmetic
    conditioning, not a parity break: the moderate-temperature test above pins
    every coefficient to ``1e-13``.
    """
    t, j = _jets(shared, order)
    got_t = torch_attention(
        t,
        torch.as_tensor(shared["K"]),
        torch.as_tensor(shared["V"]),
        DIM,
        order,
        beta=300.0,
    )
    got_j = jax_attention(
        j, jnp.asarray(shared["K"]), jnp.asarray(shared["V"]), DIM, order, beta=300.0
    )
    assert np.isfinite(got_t.detach().numpy()).all()
    assert np.isfinite(np.asarray(got_j)).all()
    assert _agree(got_t[0], got_j[0])
