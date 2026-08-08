# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CI smoke: causal time-marching driver on a tiny manufactured ODE residual.

This file is a *wiring* smoke for ``march_solve`` + causality / trivial-solution
diagnostics. It is **not** the measured acceptance result.

Regime (do not conflate with this smoke):

* **Heat** with a hard IC/BC cage -- whole-interval can already win; marching
  is not a free win.
* **Krishnapriyan reaction** ``u_t = rho u(1-u)`` at ``rho = 12`` -- whole-
  interval fails causality; gated marching must beat it.

Acceptance artifact: ``benchmarks/causal_marching.py`` (smoke / ``--full``).
Recipe: ``docs/cookbook/pinn-causal-marching.md``.
Matrix: ``docs/benchmarks/pinn_four_gap_matrix.md``.

Always supply the IC as ``ic_fn`` evaluated on the marcher's own slice points.
A linspace-ordered ``ic_values`` vector contaminates the seam metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.train import causality_index, trivial_solution_guard
from omnibias.pinn.train.torch import march_solve


class _Field(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=torch.float64),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)[:, 0]


def main() -> None:
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1e-8
    )
    field = _Field()

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        u = fld(coords)
        target = torch.sin(torch.pi * coords[:, 0]) * torch.exp(-coords[:, 1])
        return u - target

    def ic_fn(coords: torch.Tensor) -> torch.Tensor:
        # coords columns are (x, t) with t fixed at the initial slice.
        return torch.sin(torch.pi * coords[:, 0])

    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=10,
        lr=1e-2,
        per_bin=4,
        n_slice=16,
        ic_fn=ic_fn,
        seed=0,
    )
    assert len(result.windows) == 2
    assert result.windows[0].causality.n_bins == 4
    # Instrument smokes.
    assert causality_index([1.0, 2.0, 3.0]) == 0.0
    ic_probe = torch.sin(
        torch.pi * torch.linspace(0.0, 1.0, 17, dtype=torch.float64)[:-1]
    )
    v = trivial_solution_guard(
        ic_probe.numpy(), ic_probe.numpy(), ratio_threshold=1e-3
    )
    assert not v.is_trivial
    print("pinn_causal_marching: ok", result.windows[-1].final_loss)


if __name__ == "__main__":
    main()
