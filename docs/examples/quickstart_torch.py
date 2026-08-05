# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias PyTorch quickstart.

Run:

    pip install omnibias-torch
    python docs/examples/quickstart_torch.py

Shows the three core PyTorch primitives (OMBU, OperatorBlock, cmbLinear)
and a single training step.
"""

from __future__ import annotations

import torch
from omnibias.torch import OMBU, OperatorBlock, cmbLinear


def main() -> None:
    torch.manual_seed(0)

    # 1. OMBU: a trainable K-bias scalar operator, drop-in for an activation.
    ombu = OMBU(num_channels=4, K=2, base="tanh")
    x = torch.randn(8, 4)
    print("OMBU output shape:", tuple(ombu(x).shape))

    # 2. OperatorBlock: a typed scalar operator. The Laplacian block
    #    evaluates the closed-form second derivative of the base activation.
    laplacian_block = OperatorBlock(channels=4, op="laplacian", base="gaussian")
    print("Laplacian block output shape:", tuple(laplacian_block(x).shape))

    # 3. cmbLinear: nn.Linear with an inline OperatorBlock.
    model = torch.nn.Sequential(
        cmbLinear(in_features=4, out_features=16, op="identity", base="tanh"),
        cmbLinear(in_features=16, out_features=1, op="identity", base="tanh"),
    )

    # One training step on a toy regression target.
    target = x.sum(dim=1, keepdim=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    opt.zero_grad()
    loss = loss_fn(model(x), target)
    loss.backward()
    opt.step()
    print(f"loss after one step: {loss.item():.6f}")


if __name__ == "__main__":
    main()
