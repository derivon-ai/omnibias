# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Synthetic compressed-sensing demo for :class:`CvxLasso`.

Generates a sparse signal ``x_true``, a random Gaussian sensing matrix
``A``, a noisy observation ``y = A x_true + noise``, and trains a
``CvxLasso`` deep-unrolled solver to recover ``x_true`` from ``y``.

Run::

    python examples/cvxlayer_lasso.py

The script reports the per-iteration LASSO objective trace and the
final reconstruction error against the ground truth, before and after
training the unrolled solver's parameters.
"""

from __future__ import annotations

from time import perf_counter

import torch
from omnibias.torch.architectures import CvxLasso


def make_problem(
    n_features: int = 128,
    n_obs: int = 64,
    sparsity: int = 8,
    noise_std: float = 0.05,
    n_problems: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_obs, n_features, generator=g) / n_obs**0.5
    x_true = torch.zeros(n_problems, n_features)
    for i in range(n_problems):
        idx = torch.randperm(n_features, generator=g)[:sparsity]
        x_true[i, idx] = torch.randn(sparsity, generator=g)
    y = x_true @ A.T + noise_std * torch.randn(n_problems, n_obs, generator=g)
    return A, x_true, y


def main(
    n_features: int = 128,
    n_obs: int = 64,
    sparsity: int = 8,
    noise_std: float = 0.05,
    T: int = 20,
    tau: float = 0.05,
    init_step: float = 0.5,
    iters: int = 200,
    lr: float = 5e-3,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    A_true, x_true, y = make_problem(
        n_features=n_features,
        n_obs=n_obs,
        sparsity=sparsity,
        noise_std=noise_std,
        seed=seed,
    )

    model = CvxLasso(n_features=n_features, n_obs=n_obs, T=T, tau=tau, init_step=init_step)
    # Initialise A in the model to the true sensing matrix so we can isolate the
    # benefit of training the step sizes / threshold.
    with torch.no_grad():
        model.A.copy_(A_true)

    print(f"CvxLasso: n_features={n_features}, n_obs={n_obs}, T={T}, tau={tau}")
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Untrained baseline.
    with torch.no_grad():
        x_hat0 = model(y)
        rec_err0 = (x_hat0 - x_true).pow(2).mean().sqrt().item()
        loss0 = model.loss(x_hat0, y).mean().item()
    print(f"  untrained: reconstruction RMSE={rec_err0:.4f}  LASSO objective={loss0:.4f}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = perf_counter()
    for it in range(1, iters + 1):
        optim.zero_grad()
        x_hat = model(y)
        # Train to minimise the supervised reconstruction error (LISTA-style supervision).
        loss = (x_hat - x_true).pow(2).mean()
        loss.backward()
        optim.step()
        if it == 1 or it % 50 == 0:
            with torch.no_grad():
                rec_err = (x_hat - x_true).pow(2).mean().sqrt().item()
                lasso_obj = model.loss(x_hat, y).mean().item()
            print(
                f"  iter {it:4d}  train_loss={loss.item():.4f}  "
                f"recon_RMSE={rec_err:.4f}  LASSO_obj={lasso_obj:.4f}"
            )

    elapsed = perf_counter() - t0
    print(f"trained in {elapsed:.2f}s")

    with torch.no_grad():
        x_hat = model(y)
        rec_err = (x_hat - x_true).pow(2).mean().sqrt().item()
        loss_final = model.loss(x_hat, y).mean().item()
    print(f"  trained: reconstruction RMSE={rec_err:.4f}  LASSO objective={loss_final:.4f}")
    print(f"  improvement: RMSE {rec_err0:.4f} -> {rec_err:.4f}")


if __name__ == "__main__":
    main()
