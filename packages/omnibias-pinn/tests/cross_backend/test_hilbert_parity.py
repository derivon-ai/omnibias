# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for the periodic spectral Hilbert transform."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from omnibias.pinn.jax.hilbert import hilbert_transform as hilbert_jax  # noqa: E402
from omnibias.pinn.torch.hilbert import hilbert_transform as hilbert_torch  # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("n", [33, 64, 128, 257])
def test_hilbert_jax_torch_parity(n: int) -> None:
    y = -np.pi + 2.0 * np.pi * np.arange(n) / n
    rng = np.random.default_rng(n)
    f = sum(
        rng.normal() * np.cos(k * y) + rng.normal() * np.sin(k * y)
        for k in range(1, min(8, n // 2))
    )
    out_jax = np.asarray(hilbert_jax(jnp.asarray(f)))
    out_torch = hilbert_torch(torch.tensor(f)).numpy()
    np.testing.assert_allclose(out_jax, out_torch, rtol=1e-12, atol=1e-12)
