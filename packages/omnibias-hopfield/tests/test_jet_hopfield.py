# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hopfield twins retrieve the G1 germ via contact scores."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.core.jet_hopfield import JetHopfieldConfig, jet_hopfield_retrieve as core_ret
from omnibias.hopfield.jax.ops.jet_hopfield import jet_hopfield_retrieve as jax_ret
from omnibias.hopfield.torch.ops.jet_hopfield import jet_hopfield_retrieve as torch_ret


def test_g1_matches_core() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = JetHopfieldConfig(lam=1.0, beta=10.0)
    q = (1.0, 0.01)
    mem = ((1.0, 0.0), (0.0, 1.0))
    core = core_ret(q, mem, config=cfg)
    t = torch_ret(torch.tensor(q), torch.tensor(mem), config=cfg)
    j = jax_ret(jnp.asarray(q), jnp.asarray(mem), config=cfg)
    assert abs(float(t[0]) - core[0]) < 1e-12
    assert abs(float(j[0]) - core[0]) < 1e-12
    assert np.allclose(np.asarray(t.detach()), np.asarray(j), rtol=1e-12, atol=1e-12)
