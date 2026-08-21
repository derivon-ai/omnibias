# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch/JAX parity for Kantorovich-accepted Newton (theory 08-04)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from omnibias.core.verified.kantorovich import polynomial_sqrt2_maps
from omnibias.jax.optim_kantorovich import (
    kantorovich_accept_step as jax_accept,
)
from omnibias.torch.optim_kantorovich import (
    kantorovich_accept_step as torch_accept,
)

jax.config.update("jax_enable_x64", True)


def test_accept_reject_parity() -> None:
    torch.set_default_dtype(torch.float64)
    func, jac, lip = polynomial_sqrt2_maps()
    for trial, expect in ((1.5, True), (3.0, False)):
        t_dec = torch_accept(
            func,
            jac,
            torch.tensor([[1.0 / 3.0]]),
            torch.tensor([trial]),
            lipschitz_df=lip,
            r_max=0.2,
        )
        j_dec = jax_accept(
            func,
            jac,
            jnp.array([[1.0 / 3.0]]),
            jnp.array([trial]),
            lipschitz_df=lip,
            r_max=0.2,
        )
        assert t_dec.accepted is expect
        assert t_dec.accepted == j_dec.accepted
        assert t_dec.reason == j_dec.reason
        if expect:
            assert t_dec.certificate is not None
            assert j_dec.certificate is not None
            assert t_dec.certificate.radius == j_dec.certificate.radius
            assert t_dec.certificate.y0 == j_dec.certificate.y0
