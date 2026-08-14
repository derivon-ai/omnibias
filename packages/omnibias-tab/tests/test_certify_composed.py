# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""certify_composed: IBP encoder then tab/arrangement enclosure; grid+random soundness."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from torch import nn


def _box_and_samples(d: int, seed: int, per_axis: int = 4, n_rand: int = 80):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-1.2, -0.4, size=d)
    hi = rng.uniform(0.4, 1.2, size=d)
    box = np.stack([lo, hi])
    axes = [np.linspace(lo[f], hi[f], per_axis) for f in range(d)]
    grid = np.array(list(itertools.product(*axes)))
    rand = rng.uniform(lo, hi, size=(n_rand, d))
    return box, np.vstack([grid, rand])


def test_certify_composed_linear_tanh_softtree_encloses() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import SoftTreeConfig, certify_composed
    from omnibias.tab.torch.model import SoftTreeEnsemble

    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 4)).to(
        dtype=torch.float64
    )
    cfg = SoftTreeConfig(n_features=4, n_trees=3, depth=1, task="binary", beta_final=3.0, seed=1)
    head = SoftTreeEnsemble(cfg)
    box, samples = _box_and_samples(3, seed=4, per_axis=3, n_rand=60)
    cert = certify_composed(encoder, head, box, beta=3.0, use_verify=False)
    assert cert.method in {"ibp+tab", "ibp_fused", "verify_fused"}
    Xt = torch.as_tensor(samples, dtype=torch.float64)
    with torch.no_grad():
        true = head(encoder(Xt)).detach().cpu().numpy()
    lo, hi = cert.output_bounds[0]
    vals = true[:, 0] if true.ndim == 2 else true
    assert lo - 1e-8 <= float(vals.min())
    assert float(vals.max()) <= hi + 1e-8


def test_certify_composed_arrangement_encloses() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import certify_composed
    from omnibias.tab.torch.arrangement import ArrangementClassifier

    torch.manual_seed(1)
    encoder = nn.Sequential(nn.Linear(2, 3), nn.Tanh()).to(dtype=torch.float64)
    head = ArrangementClassifier(3, 2, beta=4.0)
    box, samples = _box_and_samples(2, seed=6, per_axis=4, n_rand=50)
    cert = certify_composed(encoder, head, box, beta=4.0)
    assert cert.method == "ibp+arrangement"
    Xt = torch.as_tensor(samples, dtype=torch.float64)
    with torch.no_grad():
        true = head(encoder(Xt)).detach().cpu().numpy()
    lo, hi = cert.output_bounds[0]
    vals = true[:, 0] if true.ndim == 2 else true
    assert lo - 1e-8 <= float(vals.min())
    assert float(vals.max()) <= hi + 1e-8


def test_certify_composed_modulelist_flattens_to_ibp() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import SoftTreeConfig, certify_composed
    from omnibias.tab.torch.model import SoftTreeEnsemble

    class _ListEnc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 4)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(2)
    encoder = _ListEnc().to(dtype=torch.float64)
    cfg = SoftTreeConfig(n_features=4, n_trees=3, depth=1, task="binary", beta_final=3.0, seed=2)
    head = SoftTreeEnsemble(cfg)
    box, samples = _box_and_samples(3, seed=5, per_axis=3, n_rand=60)
    cert = certify_composed(encoder, head, box, beta=3.0, use_verify=False)
    assert cert.method in {"ibp+tab", "ibp_fused", "verify_fused"}
    Xt = torch.as_tensor(samples, dtype=torch.float64)
    with torch.no_grad():
        true = head(encoder(Xt)).detach().cpu().numpy()
    lo, hi = cert.output_bounds[0]
    vals = true[:, 0] if true.ndim == 2 else true
    assert lo - 1e-8 <= float(vals.min())
    assert float(vals.max()) <= hi + 1e-8


def test_certify_composed_sampled_latent_custom_module() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import SoftTreeConfig, certify_composed
    from omnibias.tab.torch.model import SoftTreeEnsemble

    class _ScaleEnc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc(x)) * 1.7

    torch.manual_seed(3)
    encoder = _ScaleEnc().to(dtype=torch.float64)
    cfg = SoftTreeConfig(n_features=4, n_trees=2, depth=1, task="binary", beta_final=3.0, seed=1)
    head = SoftTreeEnsemble(cfg)
    box, samples = _box_and_samples(3, seed=7, per_axis=3, n_rand=40)
    cert = certify_composed(encoder, head, box, beta=3.0, use_verify=False)
    assert cert.method == "sampled_latent"
    Xt = torch.as_tensor(samples, dtype=torch.float64)
    with torch.no_grad():
        true = head(encoder(Xt)).detach().cpu().numpy()
    lo, hi = cert.output_bounds[0]
    vals = true[:, 0] if true.ndim == 2 else true
    assert lo - 1e-8 <= float(vals.min())
    assert float(vals.max()) <= hi + 1e-8


def test_certify_composed_ibp_tab_nested_softtree() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import SoftTreeConfig, certify_composed
    from omnibias.tab.torch.model import SoftTreeEnsemble

    torch.manual_seed(3)
    cfg_enc = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="binary", seed=0)
    nested = SoftTreeEnsemble(cfg_enc)
    encoder = nn.Sequential(nn.Linear(3, 3), nested).to(dtype=torch.float64)
    cfg_head = SoftTreeConfig(
        n_features=1, n_trees=2, depth=1, task="binary", beta_final=3.0, seed=1
    )
    head = SoftTreeEnsemble(cfg_head)
    box, samples = _box_and_samples(3, seed=7, per_axis=3, n_rand=40)
    cert = certify_composed(encoder, head, box, beta=3.0, use_verify=False)
    assert cert.method.startswith("ibp+tab")
    assert cert.method != "sampled_latent"
    Xt = torch.as_tensor(samples, dtype=torch.float64)
    with torch.no_grad():
        true = head(encoder(Xt)).detach().cpu().numpy()
    lo, hi = cert.output_bounds[0]
    vals = true[:, 0] if true.ndim == 2 else true
    assert lo - 1e-8 <= float(vals.min())
    assert float(vals.max()) <= hi + 1e-8


def test_certify_composed_tab_tab_softtree_encoder() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import SoftTreeConfig, certify_composed
    from omnibias.tab.torch.model import SoftTreeEnsemble

    torch.manual_seed(4)
    cfg_enc = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="binary", seed=0)
    encoder = SoftTreeEnsemble(cfg_enc)
    cfg_head = SoftTreeConfig(
        n_features=1, n_trees=2, depth=1, task="binary", beta_final=3.0, seed=1
    )
    head = SoftTreeEnsemble(cfg_head)
    box, samples = _box_and_samples(3, seed=8, per_axis=3, n_rand=40)
    cert = certify_composed(encoder, head, box, beta=3.0, use_verify=False)
    assert cert.method == "tab+tab"
    Xt = torch.as_tensor(samples, dtype=torch.float64)
    with torch.no_grad():
        true = head(encoder(Xt)).detach().cpu().numpy()
    lo, hi = cert.output_bounds[0]
    vals = true[:, 0] if true.ndim == 2 else true
    assert lo - 1e-8 <= float(vals.min())
    assert float(vals.max()) <= hi + 1e-8


def test_certify_composed_rejects_non_module() -> None:
    pytest.importorskip("omnibias.verify")
    from omnibias.tab import SoftTreeConfig, certify_composed
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="binary", seed=0)
    head = SoftTreeEnsemble(cfg)
    box = np.stack([-np.ones(3), np.ones(3)])
    with pytest.raises(TypeError, match="nn.Module"):
        certify_composed(object(), head, box)
