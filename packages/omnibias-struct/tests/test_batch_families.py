# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Batched twins for the align / MAS / plan / attention / parse / Eisner layers.

Each ``*_batched`` layer must be bit-identical to looping the per-example layer (``vmap ==
loop``) and identical across the PyTorch / JAX backends. Shapes are static per call, so the
whole batch is one traced ``vmap`` (or, for the DAG-assembled aligner, one batched reduction).
"""

from __future__ import annotations

import numpy as np
import pytest

ATOL = 1e-12


def _mdp():  # noqa: ANN202 - test helper
    from omnibias.struct._core.plan import AcyclicMDP

    return AcyclicMDP(num_states=4, actions=((0, 1), (0, 2), (1, 3), (2, 3)), start=0)


def _grammar():  # noqa: ANN202 - test helper
    from omnibias.struct._core.parse import BinaryGrammar

    return BinaryGrammar(num_nonterminals=2, rules=((0, 0, 1), (0, 1, 1), (1, 0, 0)), start=0)


# --------------------------------------------------------------------------
# torch: vmap == loop
# --------------------------------------------------------------------------


def test_soft_align_batched_matches_loop_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import soft_align, soft_align_batched

    rng = np.random.default_rng(0)
    a, b = np.array([0, 1, 2, 1]), np.array([0, 2, 1])
    sub = torch.tensor(rng.standard_normal((5, 3, 3)))
    gap = torch.tensor(rng.standard_normal(5))  # per-example
    batched = soft_align_batched(a, b, sub, gap, 3.0)
    loop = torch.stack([soft_align(a, b, sub[i], gap[i], 3.0) for i in range(5)])
    assert batched.shape == (5,)
    assert torch.max(torch.abs(batched - loop)).item() < ATOL
    # shared scalar gap also works
    shared = soft_align_batched(a, b, sub, torch.tensor(-1.0), 3.0)
    loop2 = torch.stack([soft_align(a, b, sub[i], torch.tensor(-1.0), 3.0) for i in range(5)])
    assert torch.max(torch.abs(shared - loop2)).item() < ATOL


def test_soft_mas_and_value_iteration_batched_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import (
        soft_mas,
        soft_mas_batched,
        soft_value_iteration,
        soft_value_iteration_batched,
    )

    rng = np.random.default_rng(1)
    score = torch.tensor(rng.standard_normal((4, 3, 4)))
    b_mas = soft_mas_batched(score, 2.0)
    loop_mas = torch.stack([soft_mas(score[i], 2.0) for i in range(4)])
    assert b_mas.shape == (4,)
    assert torch.max(torch.abs(b_mas - loop_mas)).item() < ATOL

    mdp = _mdp()
    rewards = torch.tensor(rng.standard_normal((4, len(mdp.actions))))
    b_vi = soft_value_iteration_batched(rewards, mdp, 1.5)
    loop_vi = torch.stack([soft_value_iteration(rewards[i], mdp, 1.5) for i in range(4)])
    assert torch.max(torch.abs(b_vi - loop_vi)).item() < ATOL


def test_structured_attention_batched_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import structured_attention, structured_attention_batched

    rng = np.random.default_rng(2)
    scores = torch.tensor(rng.standard_normal((4, 5, 3)))
    trans = torch.tensor(rng.standard_normal((3, 3)))
    batched = structured_attention_batched(scores, trans, 2.0)
    loop = torch.stack([structured_attention(scores[i], trans, 2.0) for i in range(4)])
    assert batched.shape == (4, 5, 3)
    assert torch.max(torch.abs(batched - loop)).item() < ATOL
    # per-example transitions
    trans_b = torch.tensor(rng.standard_normal((4, 3, 3)))
    batched2 = structured_attention_batched(scores, trans_b, 2.0)
    loop2 = torch.stack([structured_attention(scores[i], trans_b[i], 2.0) for i in range(4)])
    assert torch.max(torch.abs(batched2 - loop2)).item() < ATOL


def test_soft_inside_and_eisner_batched_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import (
        soft_eisner,
        soft_eisner_batched,
        soft_inside,
        soft_inside_batched,
    )

    rng = np.random.default_rng(3)
    grammar = _grammar()
    emit = torch.tensor(rng.standard_normal((4, 3, 2)))
    rule = torch.tensor(rng.standard_normal((4, grammar.num_rules)))
    b_in = soft_inside_batched(grammar, emit, rule, 2.0)
    loop_in = torch.stack([soft_inside(grammar, emit[i], rule[i], 2.0) for i in range(4)])
    assert b_in.shape == (4,)
    assert torch.max(torch.abs(b_in - loop_in)).item() < ATOL

    arc = torch.tensor(rng.standard_normal((4, 4, 4)))
    b_ei = soft_eisner_batched(arc, 2.0)
    loop_ei = torch.stack([soft_eisner(arc[i], 2.0) for i in range(4)])
    assert torch.max(torch.abs(b_ei - loop_ei)).item() < ATOL


# --------------------------------------------------------------------------
# jax: vmap == loop
# --------------------------------------------------------------------------


def test_families_batched_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import (
        soft_align,
        soft_align_batched,
        soft_eisner,
        soft_eisner_batched,
        soft_mas,
        soft_mas_batched,
    )

    rng = np.random.default_rng(4)
    a, b = np.array([0, 1, 2]), np.array([0, 2, 1])
    sub = jnp.asarray(rng.standard_normal((4, 3, 3)))
    gap = jnp.asarray(-0.7)
    b_al = np.asarray(soft_align_batched(a, b, sub, gap, 3.0))
    loop_al = np.asarray([float(soft_align(a, b, sub[i], gap, 3.0)) for i in range(4)])
    assert np.max(np.abs(b_al - loop_al)) < ATOL

    score = jnp.asarray(rng.standard_normal((4, 3, 4)))
    b_mas = np.asarray(soft_mas_batched(score, 2.0))
    loop_mas = np.asarray([float(soft_mas(score[i], 2.0)) for i in range(4)])
    assert np.max(np.abs(b_mas - loop_mas)) < ATOL

    arc = jnp.asarray(rng.standard_normal((4, 4, 4)))
    b_ei = np.asarray(soft_eisner_batched(arc, 2.0))
    loop_ei = np.asarray([float(soft_eisner(arc[i], 2.0)) for i in range(4)])
    assert np.max(np.abs(b_ei - loop_ei)) < ATOL


# --------------------------------------------------------------------------
# torch <-> jax parity of the batched twins
# --------------------------------------------------------------------------


def test_families_batched_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax import soft_eisner_batched as jax_ei
    from omnibias.struct.jax import soft_value_iteration_batched as jax_vi
    from omnibias.struct.torch import soft_eisner_batched as torch_ei
    from omnibias.struct.torch import soft_value_iteration_batched as torch_vi

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(5)
    arc = rng.standard_normal((4, 4, 4))
    et = torch_ei(torch.tensor(arc), 2.0).numpy()
    ej = np.asarray(jax_ei(jnp.asarray(arc), 2.0))
    assert np.max(np.abs(et - ej)) < ATOL

    mdp = _mdp()
    rewards = rng.standard_normal((4, len(mdp.actions)))
    vt = torch_vi(torch.tensor(rewards), mdp, 1.5).numpy()
    vj = np.asarray(jax_vi(jnp.asarray(rewards), mdp, 1.5))
    assert np.max(np.abs(vt - vj)) < ATOL
