# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""EqualityLocusLayer G1/G3/G5/G6 (theory 02-12). Not a general PDE solver."""

from __future__ import annotations

import pytest
import torch
from omnibias.core.locus import EqualitySystem, UnitTerm, residual
from omnibias.core.tanh_method import classical_pdes, published_ansatz
from omnibias.fields.locus.torch import AnsatzSolutionField, EqualityLocusLayer


def _matched() -> EqualitySystem:
    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(1, 1.0, (0.0, 1.0), 0.0),
        )
    )


def test_g1_forward_residual() -> None:
    torch.set_default_dtype(torch.float64)
    sys = _matched()
    layer = EqualityLocusLayer(sys, dtype=torch.float64)
    x0 = torch.tensor([0.4, -0.1], dtype=torch.float64)
    out = layer(x0)
    assert bool(out.converged.detach())
    nrm = abs(residual(sys, tuple(float(v) for v in out.point.detach().tolist()))[0])
    assert nrm <= 1e-12
    assert out.branch.numel() >= 1
    assert float(out.condition.detach()) >= 1.0


def test_g3_degeneracy_refusal() -> None:
    torch.set_default_dtype(torch.float64)
    sys = EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
        )
    )
    layer = EqualityLocusLayer(sys, require_transversal=True, dtype=torch.float64)
    out = layer(torch.tensor([0.2, 0.3], dtype=torch.float64))
    assert bool(out.converged.detach()) is False
    assert float(out.condition.detach()) > 1e6


def test_g5_ansatz_rejects_wrong() -> None:
    pde = classical_pdes()["burgers"]
    wrong = published_ansatz("kdv")
    with pytest.raises(ValueError, match="symbolic verification"):
        AnsatzSolutionField(pde, wrong)
    good = AnsatzSolutionField(pde, published_ansatz("burgers"))
    cert = good.certificate()
    assert cert["verified"] is True
    assert cert["level3_general_solver"] is False


def test_g6_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.fields.locus.jax import equality_locus_apply

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    sys = _matched()
    x0 = torch.tensor([0.4, -0.1], dtype=torch.float64)
    out_t = EqualityLocusLayer(sys, dtype=torch.float64)(x0)
    out_j = equality_locus_apply(sys, jnp.asarray([0.4, -0.1], dtype=jnp.float64))
    assert out_t.point.detach().cpu().numpy() == pytest.approx(out_j.point.tolist(), rel=0, abs=0)
