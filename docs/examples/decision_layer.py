# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified differentiable decision layer: predict-then-optimize with a decision gap.

`omnibias.struct.decision.DecisionLayer` embeds omnibias-struct's certified argmax-over-N as a
network layer: forward = relaxed decision ``softmax(beta * scores)`` (differentiable), the
``beta -> inf`` hard decision = ``argmax``, and ``.certificate()`` = the closed-form
`SelectionCertificate` (value gap ``log(N)/beta``, mode-mass concentration, ``L^inf``
argmax-stability radius).

This CPU smoke:

1. trains a tiny predict-then-optimize model *through the decision layer* on decision regret
   (pick the max-reward option) and shows the hard regret collapse;
2. certifies one decision and shows the gap is sound and hardens as ``beta -> inf``;
3. certifies the leaf-routing decision of a soft tree via `omnibias.tab.decision` (a certified
   "which leaf does this input commit to").

Terminology: the ``beta -> inf`` annealing is the feasibility / temperature sense of
"collapse" (a Gibbs law collapsing onto its mode), distinct from the founding ``delta -> 0``
bias collapse.
"""

from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _predict_then_optimize() -> None:
    import torch
    from omnibias.struct.decision.torch import DecisionLayer, decision_regret, expected_reward

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, d, num_opt = 512, 4, 6
    Wstar = rng.standard_normal((d, num_opt))
    X = rng.standard_normal((n, d))
    rewards = X @ Wstar
    Xt = torch.tensor(X, dtype=torch.float64)
    Rt = torch.tensor(rewards, dtype=torch.float64)

    model = torch.nn.Linear(d, num_opt, dtype=torch.float64)
    layer = DecisionLayer(beta=8.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)

    def mean_regret() -> float:
        with torch.no_grad():
            return float(decision_regret(model(Xt), Rt).mean().item())

    init = mean_regret()
    for _ in range(200):
        opt.zero_grad()
        p = layer(model(Xt))
        loss = -expected_reward(p, Rt).mean()
        loss.backward()
        opt.step()
    final = mean_regret()

    print("=== 1. predict-then-optimize (train through the decision layer) ===")
    print(f"    mean decision regret: {init:.4f} (init) -> {final:.4f} (trained)")
    assert final < 0.5 * init
    print("    OK: differentiating the relaxed decision drives the hard regret down.")


def _certify_a_decision() -> None:
    from omnibias.struct.decision import certified_decision

    scores = np.array([0.2, 1.5, 0.9, -0.3])
    n = scores.size
    print("\n=== 2. certify one decision (value gap log(N)/beta, mode mass, stability) ===")
    for beta in (2.0, 10.0, 50.0):
        c = certified_decision(scores, beta=beta, eps=0.1)
        print(
            f"    beta={beta:5.1f}: argmax={c.argmax}  gap<= {c.gap_bound:.4f}"
            f"  p_max={c.p_max:.4f} (>= {c.p_max_lower:.4f})"
            f"  robust_radius={c.robust_radius:.3f}  sound={c.is_sound}"
        )
    c50 = certified_decision(scores, beta=50.0, eps=0.1)
    assert c50.is_sound
    assert math.isclose(c50.gap_bound, math.log(n) / 50.0, rel_tol=1e-12)
    assert c50.argmax_stable is True
    print("    OK: sound at every beta; gap -> 0 and mode mass -> 1 as beta -> inf.")


def _certify_tab_leaf_routing() -> None:
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab._core.params import init_params
    from omnibias.tab.decision import certified_leaf_decision

    cfg = SoftTreeConfig(
        n_features=3, n_trees=1, depth=2, n_outputs=1, task="regression", beta_final=16.0
    )
    params = init_params(cfg, 0)
    x = np.array([0.7, -0.9, 0.4])
    print("\n=== 3. certify a soft tree's leaf-routing decision (omnibias.tab.decision) ===")
    for beta in (4.0, 16.0, 64.0):
        routing, cert = certified_leaf_decision(params, x, beta=beta)
        print(
            f"    beta={beta:5.1f}: routes to leaf {cert.argmax}"
            f"  p_max={cert.p_max:.4f}  gap<= {cert.gap_bound:.4f}  sound={cert.is_sound}"
        )
    _, c = certified_leaf_decision(params, x, beta=64.0)
    assert c.is_sound
    print("    OK: a certified, robust 'which leaf' decision -- no change to tab's forward.")


def main() -> None:
    _predict_then_optimize()
    _certify_a_decision()
    _certify_tab_leaf_routing()
    print("\nAll decision-layer checks passed.")


if __name__ == "__main__":
    main()
