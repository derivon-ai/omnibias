# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for the numerical whole-line Hilbert (not periodic FFT)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from omnibias.pinn.jax.equations.ccf_compactified import (  # noqa: E402
    alpha_from_lambda,
)
from omnibias.pinn.jax.equations.ccf_compactified import (
    hardy_odd as hardy_odd_jax,
)
from omnibias.pinn.jax.hilbert_line import (  # noqa: E402
    hilbert_wholeline_hp as hilbert_hp_jax,
)
from omnibias.pinn.torch.equations.ccf_compactified import (  # noqa: E402
    hardy_odd as hardy_odd_torch,
)
from omnibias.pinn.torch.hilbert_line import (  # noqa: E402
    hilbert_wholeline_hp as hilbert_hp_torch,
)

torch.set_default_dtype(torch.float64)

LAM = 0.6057
ALPHA = float(alpha_from_lambda(LAM))


def test_wholeline_hp_torch_jax_parity_on_hardy_q() -> None:
    """Planted Q_{1.3,α}: torch and JAX hp agree far below the 1e-8 diagnostic."""
    y_np = np.linspace(-40.0, 40.0, 257, dtype=np.float64)
    a = 1.3
    y_t = torch.as_tensor(y_np, dtype=torch.float64)
    y_j = jnp.asarray(y_np)

    def omega_fn_t(t: torch.Tensor) -> torch.Tensor:
        return hardy_odd_torch(t, a, ALPHA)

    def omega_fn_j(t: jnp.ndarray) -> jnp.ndarray:
        return hardy_odd_jax(t, a, ALPHA)

    h_t = hilbert_hp_torch(
        y_t, omega_fn_t(y_t), omega_fn_t, decay_power=ALPHA, y_trunc=40.0
    )
    h_j = hilbert_hp_jax(
        y_j, omega_fn_j(y_j), omega_fn_j, decay_power=ALPHA, y_trunc=40.0
    )
    np.testing.assert_allclose(
        h_t.detach().cpu().numpy(), np.asarray(h_j), rtol=1e-12, atol=1e-14
    )
