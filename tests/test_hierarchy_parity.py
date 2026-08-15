# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hierarchical scan torch/jax parity (theory 02-07 G5)."""

from __future__ import annotations

import pytest
import torch
from omnibias.core.hierarchy import build_pack_tree
from omnibias.torch.hierarchy import hierarchical_scan


def test_hierarchy_eta_zero_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.jax.hierarchy import hierarchical_scan as hier_jax

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    offsets = tuple(float(i) * 0.2 - 0.8 for i in range(8))
    weights = tuple(0.1 * ((-1.0) ** i) for i in range(8))
    orders = tuple(1 for _ in range(8))
    tree = build_pack_tree(offsets, leaf_size=2)
    z = torch.tensor([-0.3, 0.1, 0.8], dtype=torch.float64)
    y_t = hierarchical_scan(z, tree, offsets, weights, orders, p=4, eta=0.0)
    y_j = hier_jax(jnp.asarray(z.numpy()), tree, offsets, weights, orders, p=4, eta=0.0)
    import numpy as np

    assert np.asarray(y_t.detach().cpu()) == pytest.approx(np.asarray(y_j), rel=0, abs=0)
