# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Optional PINN residual packer. Not imported by default symbolic."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from omnibias.symbolic.conditions import observation_tanh  # noqa: E402


def test_fit_residual_and_pack_two_steps() -> None:
    pytest.importorskip("omnibias.pinn")
    from omnibias.pinn.jax.discovery.observation import fit_residual_and_pack

    packed = fit_residual_and_pack(observation_tanh(), steps=2)
    residual = packed.extra_map().get("residual")
    assert residual
    assert "," in residual
