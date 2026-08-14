# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Plugin gate: tab layers compose with any encoder, autograd through both sides."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from omnibias.tab import SoftTreeConfig
from omnibias.tab.torch.arrangement import ArrangementBoosted, ArrangementClassifier
from omnibias.tab.torch.model import SoftTreeEnsemble
from omnibias.tab.torch.plugin import TabHead, as_head
from torch import nn


def _adam_steps(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *, steps: int = 8) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.reshape(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()


class _EncoderHead(nn.Module):
    def __init__(self, head: nn.Module, *, in_features: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh())
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


def _assert_both_sides_moved(
    model: _EncoderHead, X: torch.Tensor, y: torch.Tensor, head_param: torch.Tensor
) -> None:
    enc_w = model.enc[0].weight
    before_enc = enc_w.detach().clone()
    before_head = head_param.detach().clone()
    _adam_steps(model, X, y)
    assert not torch.allclose(enc_w, before_enc)
    assert not torch.allclose(head_param, before_head)


def _three_heads(hidden: int = 16) -> list[tuple[str, nn.Module]]:
    cfg = SoftTreeConfig(
        n_features=hidden, n_trees=3, depth=2, task="binary", beta_final=4.0, seed=0
    )
    return [
        ("softtree", SoftTreeEnsemble(cfg)),
        ("arrangement", ArrangementClassifier(hidden, 2, beta=3.0)),
        (
            "boosted",
            ArrangementBoosted(
                [ArrangementClassifier(hidden, 2, beta=3.0) for _ in range(2)],
                learning_rate=0.3,
                base=0.0,
            ),
        ),
    ]


def test_softtree_encoder_joint_adam() -> None:
    torch.manual_seed(0)
    cfg = SoftTreeConfig(
        n_features=16, n_trees=4, depth=2, task="binary", beta_final=8.0, seed=0
    )
    head = SoftTreeEnsemble(cfg)
    model = _EncoderHead(head).to(dtype=torch.float64)
    X = torch.randn(32, 8, dtype=torch.float64)
    y = (X[:, 0] > 0).to(dtype=torch.float64)
    _assert_both_sides_moved(model, X, y, model.head.W)


def test_arrangement_encoder_joint_adam() -> None:
    torch.manual_seed(1)
    head = ArrangementClassifier(16, 2, beta=4.0)
    model = _EncoderHead(head).to(dtype=torch.float64)
    X = torch.randn(32, 8, dtype=torch.float64)
    y = (X[:, 0] > 0).to(dtype=torch.float64)
    _assert_both_sides_moved(model, X, y, model.head.W)


def test_boosted_arrangement_encoder_joint_adam() -> None:
    torch.manual_seed(2)
    members = [ArrangementClassifier(16, 2, beta=4.0) for _ in range(2)]
    head = ArrangementBoosted(members, learning_rate=0.3, base=0.0)
    model = _EncoderHead(head).to(dtype=torch.float64)
    X = torch.randn(32, 8, dtype=torch.float64)
    y = (X[:, 0] > 0).to(dtype=torch.float64)
    _assert_both_sides_moved(model, X, y, model.head.members[0].W)


def test_boosted_forward_keeps_autograd_graph() -> None:
    torch.manual_seed(3)
    X = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    head = ArrangementBoosted(
        [ArrangementClassifier(5, 2, beta=2.0)], learning_rate=0.4, base=-0.1
    )
    out = head(X)
    assert out.grad_fn is not None
    out.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_as_head_matches_host_tensor() -> None:
    torch.manual_seed(4)
    z = torch.randn(8, 7, dtype=torch.float32)
    kinds_k: list[tuple[str, int, dict[str, object]]] = [
        ("softtree", 1, {"n_trees": 2, "depth": 1, "seed": 4}),
        ("arrangement", 1, {}),
        ("boosted", 1, {}),
        ("arrangement", 3, {"task": "multiclass", "n_outputs": 3}),
    ]
    for kind, k, extra in kinds_k:
        head = as_head(z, kind, **extra)
        assert isinstance(head, TabHead)
        out = head(z)
        assert out.device == z.device
        assert out.dtype == z.dtype
        assert out.shape[:-1] == z.shape[:-1]
        assert out.shape[-1] == k


def test_as_head_encoder_joint_adam_moves_w() -> None:
    torch.manual_seed(9)
    probe = torch.zeros(1, 16, dtype=torch.float64)
    head = as_head(probe, "arrangement", n_hyperplanes=2, beta=2.0)
    assert isinstance(head, TabHead)
    model = _EncoderHead(head).to(dtype=torch.float64)
    X = torch.randn(32, 8, dtype=torch.float64)
    y = (X[:, 0] > 0).to(dtype=torch.float64)
    _assert_both_sides_moved(model, X, y, model.head.W)


def test_beta_lr_in_state_dict_follow_dtype() -> None:
    head = ArrangementClassifier(4, 2, beta=2.5)
    boosted = ArrangementBoosted([head], learning_rate=0.2, base=0.1)
    keys = boosted.state_dict()
    assert "_learning_rate" in keys
    assert "_base" in keys
    assert "_beta" in head.state_dict()
    boosted32 = boosted.to(dtype=torch.float32)
    assert boosted32.state_dict()["_learning_rate"].dtype == torch.float32
    assert boosted32.members[0]._beta.dtype == torch.float32


@pytest.mark.parametrize(
    "kind",
    ["softtree", "arrangement"],
)
def test_learnable_beta(kind: str) -> None:
    torch.manual_seed(8)
    if kind == "softtree":
        cfg = SoftTreeConfig(n_features=4, n_trees=2, depth=1, task="binary", seed=8)
        frozen: nn.Module = SoftTreeEnsemble(cfg, learnable_beta=False)
        learn: nn.Module = SoftTreeEnsemble(cfg, learnable_beta=True)
    else:
        frozen = ArrangementClassifier(4, 2, beta=2.5, learnable_beta=False)
        learn = ArrangementClassifier(4, 2, beta=2.5, learnable_beta=True)
        with torch.no_grad():
            learn.cell_logits.normal_()

    assert not isinstance(frozen._beta, nn.Parameter)
    assert "_beta" in dict(frozen.named_buffers())
    assert isinstance(learn._beta, nn.Parameter)
    assert "_beta" in dict(learn.named_parameters())
    assert "_beta" not in dict(learn.named_buffers())

    X = torch.randn(6, 4, dtype=torch.float64)
    loss = learn(X).sum()
    loss.backward()
    assert learn._beta.grad is not None
    assert float(learn._beta.grad.abs().max()) > 0.0

    learn32 = learn.to(dtype=torch.float32)
    assert learn32._beta.dtype == torch.float32
    assert "_beta" in learn32.state_dict()
    assert learn32.state_dict()["_beta"].dtype == torch.float32


def test_float32_cpu_plugin() -> None:
    torch.manual_seed(4)
    head = ArrangementClassifier(16, 2, beta=3.0)
    model = _EncoderHead(head).to(dtype=torch.float32)
    X = torch.randn(16, 8, dtype=torch.float32)
    y = (X[:, 0] > 0).to(dtype=torch.float32)
    out = model(X)
    assert out.dtype == torch.float32
    assert out.device == X.device
    _assert_both_sides_moved(model, X, y, model.head.W)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("kind,head", _three_heads())
def test_cuda_plugin_all_heads(kind: str, head: nn.Module) -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    model = _EncoderHead(head, hidden=16).to(device=device, dtype=torch.float32)
    X = torch.randn(16, 8, dtype=torch.float32, device=device)
    y = (X[:, 0] > 0).to(dtype=torch.float32)
    out = model(X)
    assert out.device.type == "cuda"
    assert out.dtype == torch.float32
    hp = model.head.W if kind != "boosted" else model.head.members[0].W
    _assert_both_sides_moved(model, X, y, hp)


def _head_weight(kind: str, model: _EncoderHead) -> torch.Tensor:
    if kind == "boosted":
        return model.head.members[0].W
    return model.head.W


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("kind,head", _three_heads())
def test_amp_cuda_finite(kind: str, head: nn.Module) -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    model = _EncoderHead(head, hidden=16).to(device=device, dtype=torch.float32)
    X = torch.randn(16, 8, dtype=torch.float32, device=device)
    y = (X[:, 0] > 0).to(dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(X)
        if logits.shape[-1] == 1:
            logits = logits.reshape(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    hp = _head_weight(kind, model)
    assert hp.grad is not None
    assert torch.isfinite(hp.grad).all()
    opt.step()
    assert torch.isfinite(hp).all()


def _cpu_amp_dtype() -> torch.dtype:
    if hasattr(torch, "bfloat16"):
        try:
            torch.zeros(1, dtype=torch.bfloat16)
            return torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float16


@pytest.mark.parametrize("kind,head", _three_heads())
def test_amp_cpu_finite(kind: str, head: nn.Module) -> None:
    torch.manual_seed(6)
    model = _EncoderHead(head, hidden=16).to(dtype=torch.float32)
    X = torch.randn(16, 8, dtype=torch.float32)
    y = (X[:, 0] > 0).to(dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    opt.zero_grad(set_to_none=True)
    amp_dtype = _cpu_amp_dtype()
    with torch.autocast(device_type="cpu", dtype=amp_dtype):
        logits = model(X)
        if logits.shape[-1] == 1:
            logits = logits.reshape(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    hp = _head_weight(kind, model)
    assert hp.grad is not None
    assert torch.isfinite(hp.grad).all()
    opt.step()
    assert torch.isfinite(hp).all()


def _compile_available() -> bool:
    if not hasattr(torch, "compile"):
        return False
    try:
        import importlib

        importlib.import_module("torch._inductor")
    except Exception:
        return False
    return True


def _on_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("1", "true", "yes")


@pytest.mark.skipif(
    not _compile_available() and not _on_ci(),
    reason="torch.compile / inductor missing",
)
@pytest.mark.parametrize("kind", ["softtree", "arrangement", "boosted"])
def test_torch_compile_all_heads(kind: str) -> None:
    if not _compile_available():
        pytest.fail("torch.compile / inductor must be importable on CI")
    torch.manual_seed(7)
    X = torch.randn(8, 5, dtype=torch.float64)
    if kind == "softtree":
        module: nn.Module = SoftTreeEnsemble(
            SoftTreeConfig(n_features=5, n_trees=2, depth=2, task="binary", seed=7)
        )
    elif kind == "arrangement":
        module = ArrangementClassifier(5, 2, beta=2.0)
    else:
        module = ArrangementBoosted(
            [ArrangementClassifier(5, 2, beta=2.0) for _ in range(2)],
            learning_rate=0.3,
            base=0.0,
        )
    eager = module(X)
    compiled = torch.compile(module, fullgraph=False)
    out = compiled(X)
    assert torch.allclose(out, eager, atol=1e-8, rtol=1e-8)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(out).all()


def test_leading_dims_arrangement_and_softtree() -> None:
    torch.manual_seed(6)
    X = torch.randn(2, 5, 7, dtype=torch.float64)
    arr = ArrangementClassifier(7, 2, beta=2.5)
    flat_a = arr(X.reshape(10, 7)).reshape(2, 5, 1)
    assert arr(X).shape == (2, 5, 1)
    assert torch.allclose(arr(X), flat_a, atol=1e-12)

    cfg = SoftTreeConfig(
        n_features=7, n_trees=3, depth=2, task="binary", beta_final=4.0, seed=6
    )
    tree = SoftTreeEnsemble(cfg)
    flat_t = tree(X.reshape(10, 7)).reshape(2, 5, 1)
    assert tree(X).shape == (2, 5, 1)
    assert torch.allclose(tree(X), flat_t, atol=1e-12)

    boosted = ArrangementBoosted([arr], learning_rate=0.5, base=0.2)
    flat_b = boosted(X.reshape(10, 7)).reshape(2, 5, 1)
    assert boosted(X).shape == (2, 5, 1)
    assert torch.allclose(boosted(X), flat_b, atol=1e-12)


def test_jax_grad_through_encoder_and_arrangement() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.tab.jax.arrangement import arrangement_forward

    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 6))
    enc_W = rng.normal(size=(6, 8)) * 0.3
    enc_b = rng.normal(size=(8,)) * 0.1
    W = rng.normal(size=(2, 8)) * 0.3
    t = rng.normal(size=(2,)) * 0.1
    cell = rng.normal(size=(4,))
    beta = 3.0

    def loss(eW, eb, tW, tt, logits, xv):
        z = jnp.tanh(xv @ eW + eb)
        return arrangement_forward(tW, tt, logits, z, beta).mean()

    g_eW, g_tW = jax.grad(loss, argnums=(0, 2))(
        jnp.asarray(enc_W),
        jnp.asarray(enc_b),
        jnp.asarray(W),
        jnp.asarray(t),
        jnp.asarray(cell),
        jnp.asarray(X),
    )
    assert float(jnp.max(jnp.abs(g_eW))) > 0.0
    assert float(jnp.max(jnp.abs(g_tW))) > 0.0
