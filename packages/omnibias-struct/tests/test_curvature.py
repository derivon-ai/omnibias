# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Second-order-through-DP bridge: exact HVP == closed-form jet curvature; Hessian is PSD.

The soft Viterbi value is a log-sum-exp of emission-linear path scores, so it is convex
and its Hessian is positive semidefinite. The generic ``omnibias-curvature`` HVP contracted
with a direction must equal ``2 * chain_lse_jet[2]`` (the ``delta -> 0`` closed-form
directional curvature), and the top eigenvalue must match a dense-Hessian eigensolve.
"""

from __future__ import annotations

import numpy as np
import pytest
from _struct_helpers import random_chain

ATOL = 1e-9


def test_hvp_matches_closed_form_jet_curvature() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import chain_directional_curvature, chain_lse_jet

    trellis = random_chain(0)
    e = torch.tensor(trellis.emissions)
    trn = torch.tensor(trellis.transitions)
    st = torch.tensor(trellis.start)
    d = torch.tensor(np.random.default_rng(1).standard_normal(e.shape))
    beta = 3.0
    curv_hvp = float(chain_directional_curvature(e, trn, d, beta, start=st))
    curv_jet = 2.0 * float(chain_lse_jet(e, trn, d, beta, order=2, start=st)[2])
    assert abs(curv_hvp - curv_jet) < ATOL


def test_value_and_hvp_value_matches_soft_viterbi() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import chain_value_and_hvp, soft_viterbi

    trellis = random_chain(1)
    e = torch.tensor(trellis.emissions)
    trn = torch.tensor(trellis.transitions)
    st = torch.tensor(trellis.start)
    v = torch.tensor(np.random.default_rng(0).standard_normal(e.shape))
    value, hv = chain_value_and_hvp(e, trn, v, 4.0, start=st)
    assert abs(float(value) - float(soft_viterbi(e, trn, 4.0, start=st))) < 1e-12
    assert hv.shape == e.shape


def test_soft_viterbi_hessian_is_psd() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import chain_hessian

    trellis = random_chain(2)
    h = chain_hessian(
        torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), 5.0,
        start=torch.tensor(trellis.start),
    ).numpy()
    assert np.allclose(h, h.T, atol=1e-10)  # symmetric
    eigs = np.linalg.eigvalsh(h)
    assert eigs.min() > -1e-9  # convex -> PSD


def test_sharpness_matches_dense_eigensolve_and_grows_with_beta() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import chain_hessian, chain_sharpness

    trellis = random_chain(3)
    e = torch.tensor(trellis.emissions)
    trn = torch.tensor(trellis.transitions)
    st = torch.tensor(trellis.start)
    top = float(chain_sharpness(e, trn, 4.0, start=st))
    ref = float(np.linalg.eigvalsh(chain_hessian(e, trn, 4.0, start=st).numpy())[-1])
    assert abs(top - ref) < 1e-9
    # sharper relaxation at higher beta
    assert float(chain_sharpness(e, trn, 8.0, start=st)) > float(chain_sharpness(e, trn, 1.0, start=st))
