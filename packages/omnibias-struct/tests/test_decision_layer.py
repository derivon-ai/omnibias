# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The embeddable certified decision layer (omnibias.struct.decision).

forward = relaxed softmax decision (differentiable); the beta -> inf hard decision = argmax;
.certificate() = the closed-form SelectionCertificate (value gap log(N)/beta, mode-mass
concentration, L^inf argmax-stability). A tiny predict-then-optimize task trains through the
layer and reduces decision regret; the certificate is sound and hardens as beta grows.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omnibias.struct.decision import (  # noqa: E402
    best_index,
    certified_decision,
    decision_regret,
)
from omnibias.struct.decision.torch import (  # noqa: E402
    DecisionLayer,
    expected_reward,
)
from omnibias.struct.decision.torch import decision_regret as decision_regret_torch  # noqa: E402


def test_soft_decision_is_a_distribution_and_hardens() -> None:
    layer = DecisionLayer(beta=1.0)
    scores = torch.tensor([0.2, 1.5, 0.9, -0.3], dtype=torch.float64)
    p = layer.forward(scores)
    assert torch.allclose(p.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.all(p >= 0.0)
    # as beta -> inf the relaxed decision approaches the one-hot at the argmax (index 1)
    sharp = DecisionLayer(beta=200.0).forward(scores)
    assert torch.argmax(sharp).item() == 1
    assert sharp[1].item() > 0.999
    assert layer.hard(scores).item() == 1


def test_certificate_is_sound_and_hardens() -> None:
    scores = np.array([0.2, 1.5, 0.9, -0.3])
    n = scores.size
    c5 = certified_decision(scores, beta=5.0, eps=0.1)
    c50 = certified_decision(scores, beta=50.0, eps=0.1)

    assert c5.is_sound and c50.is_sound
    # value gap is exactly log(N)/beta and shrinks as beta grows
    assert math.isclose(c5.gap_bound, math.log(n) / 5.0, rel_tol=1e-12)
    assert c50.gap_bound < c5.gap_bound
    # mode mass concentrates toward 1
    assert c50.p_max > c5.p_max
    assert c50.p_max > 0.999
    # decoded mode is the true argmax; robust radius is margin/2
    assert c5.argmax == 1
    assert math.isclose(c5.robust_radius, c5.margin / 2.0, rel_tol=1e-12)
    # margin here is 1.5 - 0.9 = 0.6 > 2*eps = 0.2, so the decode is eps-stable
    assert c5.argmax_stable is True


def test_layer_certificate_matches_functional() -> None:
    layer = DecisionLayer(beta=8.0, eps=0.05)
    scores = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64)
    soft, cert = layer.certified(scores)
    assert torch.allclose(soft, layer.forward(scores))
    ref = certified_decision(scores.numpy(), beta=8.0, eps=0.05)
    assert cert.argmax == ref.argmax
    assert math.isclose(cert.gap_bound, ref.gap_bound, rel_tol=1e-12)
    assert math.isclose(cert.p_max, ref.p_max, rel_tol=1e-9)


def test_batched_certificates_and_regret() -> None:
    layer = DecisionLayer(beta=6.0)
    scores = torch.tensor([[0.1, 0.9, 0.2], [1.2, 0.0, -0.4]], dtype=torch.float64)
    certs = layer.certificates(scores)
    assert len(certs) == 2
    assert [c.argmax for c in certs] == [1, 0]

    rewards = np.array([[0.0, 1.0, 0.5], [2.0, 0.1, 0.0]])
    # perfect predictions -> zero regret
    reg = decision_regret(scores.numpy(), rewards)
    assert np.allclose(reg, 0.0)
    # a wrong prediction on row 0 incurs regret 1.0
    bad = np.array([[3.0, 0.9, 0.2], [1.2, 0.0, -0.4]])
    assert np.isclose(decision_regret(bad, rewards).mean(), 0.5)
    assert best_index(rewards, axis=1).tolist() == [1, 0]


def test_layer_trains_and_reduces_regret() -> None:
    # predict-then-optimize: pick the max-reward option; reward_j is linear in features.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, d, num_opt = 256, 4, 5
    Wstar = rng.standard_normal((d, num_opt))
    X = rng.standard_normal((n, d))
    rewards = X @ Wstar  # (n, num_opt) true option rewards
    Xt = torch.tensor(X, dtype=torch.float64)
    Rt = torch.tensor(rewards, dtype=torch.float64)

    model = torch.nn.Linear(d, num_opt, dtype=torch.float64)
    layer = DecisionLayer(beta=8.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)

    def mean_regret() -> float:
        with torch.no_grad():
            return float(decision_regret_torch(model(Xt), Rt).mean().item())

    init_regret = mean_regret()
    for _ in range(150):
        opt.zero_grad()
        p = layer(model(Xt))  # relaxed decision (differentiable)
        loss = -expected_reward(p, Rt).mean()  # maximize expected realized reward
        loss.backward()
        opt.step()
    final_regret = mean_regret()

    # training through the differentiable decision drives the hard regret down
    assert final_regret < 0.5 * init_regret
