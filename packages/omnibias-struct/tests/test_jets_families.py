# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact higher-order jets through the CKY / Eisner recursions on the semiring driver.

The inside recursion of a hypergraph DP is a ``+`` / ``lse_beta`` fold, so the exact
associative :func:`lse2_jet` propagates a directional Taylor jet through it with no autodiff.
We pin the closed-form ``(value, directional-grad, directional-curvature)`` against backend
autodiff of :func:`soft_inside` / :func:`soft_eisner`, cross-check the third-order coefficient
against finite differences, and require torch <-> jax parity to ``1e-9``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _grammar():  # noqa: ANN202 - test helper
    from omnibias.struct._core.parse import BinaryGrammar

    return BinaryGrammar(num_nonterminals=2, rules=((0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0)), start=0)


# --- CKY inside jet ------------------------------------------------------


def test_cky_jet_matches_autodiff_value_grad_hessian_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import soft_inside
    from omnibias.struct.torch.jets import cky_lse_jet

    grammar = _grammar()
    rng = np.random.default_rng(0)
    length = 4
    emit = torch.tensor(rng.standard_normal((length, grammar.num_nonterminals)))
    rule = torch.tensor(rng.standard_normal(grammar.num_rules))
    de = torch.tensor(rng.standard_normal(emit.shape))
    dr = torch.tensor(rng.standard_normal(rule.shape))
    beta = 3.0
    jet = cky_lse_jet(grammar, emit, rule, de, dr, beta, order=2)

    e2 = emit.clone().requires_grad_(True)
    r2 = rule.clone().requires_grad_(True)
    value = soft_inside(grammar, e2, r2, beta)
    ge, gr = torch.autograd.grad(value, (e2, r2), create_graph=True)
    grad_d = (ge * de).sum() + (gr * dr).sum()
    hde, hdr = torch.autograd.grad(grad_d, (e2, r2), retain_graph=True)
    d_h_d = (hde * de).sum() + (hdr * dr).sum()

    assert abs(float(jet[0]) - float(value.detach())) < 1e-9
    assert abs(float(jet[1]) - float(grad_d.detach())) < 1e-9
    assert abs(2.0 * float(jet[2]) - float(d_h_d.detach())) < 1e-9


def test_cky_jet_matches_finite_differences_order3_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import soft_inside
    from omnibias.struct.torch.jets import cky_lse_jet

    grammar = _grammar()
    rng = np.random.default_rng(1)
    length = 3
    emit = torch.tensor(rng.standard_normal((length, grammar.num_nonterminals)))
    rule = torch.tensor(rng.standard_normal(grammar.num_rules))
    de = torch.tensor(rng.standard_normal(emit.shape))
    dr = torch.tensor(rng.standard_normal(rule.shape))
    beta = 2.0
    jet = cky_lse_jet(grammar, emit, rule, de, dr, beta, order=3)

    def value_at(t: float) -> float:
        return float(soft_inside(grammar, emit + t * de, rule + t * dr, beta))

    h = 1e-3
    fd1 = (value_at(h) - value_at(-h)) / (2 * h)
    fd2 = (value_at(h) - 2 * value_at(0.0) + value_at(-h)) / h**2
    assert abs(float(jet[1]) - fd1) < 1e-6
    assert abs(2.0 * float(jet[2]) - fd2) < 1e-4


# --- Eisner projective-parse jet ----------------------------------------


def test_eisner_jet_matches_autodiff_torch() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.struct.torch import soft_eisner
    from omnibias.struct.torch.jets import eisner_lse_jet

    rng = np.random.default_rng(2)
    n = 3
    arc = torch.tensor(rng.standard_normal((n + 1, n + 1)))
    d = torch.tensor(rng.standard_normal(arc.shape))
    beta = 3.0
    jet = eisner_lse_jet(arc, d, beta, order=2)

    a2 = arc.clone().requires_grad_(True)
    value = soft_eisner(a2, beta)
    (grad,) = torch.autograd.grad(value, a2, create_graph=True)
    grad_d = (grad * d).sum()
    (hd,) = torch.autograd.grad(grad_d, a2, retain_graph=True)
    d_h_d = (hd * d).sum()

    assert abs(float(jet[0]) - float(value.detach())) < 1e-9
    assert abs(float(jet[1]) - float(grad_d.detach())) < 1e-9
    assert abs(2.0 * float(jet[2]) - float(d_h_d.detach())) < 1e-9


def test_eisner_jet_value_matches_soft_eisner_jax() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import soft_eisner
    from omnibias.struct.jax.jets import eisner_lse_jet

    rng = np.random.default_rng(3)
    n = 3
    arc = jnp.asarray(rng.standard_normal((n + 1, n + 1)))
    d = jnp.asarray(rng.standard_normal((n + 1, n + 1)))
    beta = 2.5
    jet = eisner_lse_jet(arc, d, beta, order=2)
    value = soft_eisner(arc, beta)
    grad = jax.grad(lambda a: soft_eisner(a, beta))(arc)
    grad_d = float(jnp.sum(grad * d))
    assert abs(float(jet[0]) - float(value)) < 1e-9
    assert abs(float(jet[1]) - grad_d) < 1e-9


# --- torch <-> jax parity ------------------------------------------------


def test_cky_and_eisner_jets_torch_jax_parity() -> None:
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.struct.jax.jets import cky_lse_jet as jax_cky
    from omnibias.struct.jax.jets import eisner_lse_jet as jax_eis
    from omnibias.struct.torch.jets import cky_lse_jet as torch_cky
    from omnibias.struct.torch.jets import eisner_lse_jet as torch_eis

    torch.set_default_dtype(torch.float64)
    grammar = _grammar()
    rng = np.random.default_rng(4)
    length = 4
    emit = rng.standard_normal((length, grammar.num_nonterminals))
    rule = rng.standard_normal(grammar.num_rules)
    de = rng.standard_normal(emit.shape)
    dr = rng.standard_normal(rule.shape)
    ct = torch_cky(
        grammar, torch.tensor(emit), torch.tensor(rule), torch.tensor(de), torch.tensor(dr), 4.0, order=3
    ).numpy()
    cj = np.asarray(
        jax_cky(
            grammar, jnp.asarray(emit), jnp.asarray(rule), jnp.asarray(de), jnp.asarray(dr), 4.0, order=3
        )
    )
    assert np.max(np.abs(ct - cj)) < 1e-9

    n = 3
    arc = rng.standard_normal((n + 1, n + 1))
    d = rng.standard_normal((n + 1, n + 1))
    et = torch_eis(torch.tensor(arc), torch.tensor(d), 3.0, order=2).numpy()
    ej = np.asarray(jax_eis(jnp.asarray(arc), jnp.asarray(d), 3.0, order=2))
    assert np.max(np.abs(et - ej)) < 1e-9
