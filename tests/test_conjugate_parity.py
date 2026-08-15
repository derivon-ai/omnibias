# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cauchy-Hardy dictionary torch/jax parity (theory 01-12 G6)."""

from __future__ import annotations

import pytest
import torch
from omnibias.core.conjugate import HardyAtom, HardyDictionary


def test_conjugate_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.jax.conjugate import hardy_atoms as hardy_jax
    from omnibias.jax.conjugate import hilbert_coeffs as hilbert_jax
    from omnibias.torch.conjugate import hardy_atoms, hilbert_coeffs

    jax.config.update("jax_enable_x64", True)
    d = HardyDictionary(
        (
            HardyAtom(1.0, 0.5, 0, "even"),
            HardyAtom(1.0, 0.5, 0, "odd"),
            HardyAtom(1.0, 0.5, 2, "even"),
            HardyAtom(1.0, 0.5, 2, "odd"),
        )
    )
    y_t = torch.tensor([0.3, 1.0, -0.4], dtype=torch.float64)
    y_j = jnp.asarray([0.3, 1.0, -0.4], dtype=jnp.float64)
    import numpy as np

    a_t = hardy_atoms(y_t, d)
    a_j = hardy_jax(y_j, d)
    np.testing.assert_array_equal(a_t.detach().cpu().numpy(), np.asarray(a_j))
    c_t = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float64)
    c_j = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float64)
    h_t = hilbert_coeffs(c_t, d)
    h_j = hilbert_jax(c_j, d)
    np.testing.assert_array_equal(h_t.detach().cpu().numpy(), np.asarray(h_j))
