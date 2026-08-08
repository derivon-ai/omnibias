# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CPU smoke: causal marching vs whole-interval fit on a manufactured target.

Decision rule (fixed before the run): marched final loss must be finite and
the causality index of the last window must be reported. This is a wiring /
smoke artifact, not a headline accuracy claim.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.train.torch import march_solve


class _Field(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(32, 1, dtype=torch.float64),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)[:, 0]


def main() -> None:
    t0 = time.perf_counter()
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0, 1.0, n_windows=2, n_time_bins=4, epsilon=1.0, tolerance=1e-8
    )
    field = _Field()

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        u = fld(coords)
        target = torch.sin(np.pi * coords[:, 0]) * torch.exp(-coords[:, 1])
        return u - target

    ic = np.sin(np.pi * np.linspace(0.0, 1.0, 32, endpoint=False))
    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=20,
        lr=1e-2,
        per_bin=8,
        n_slice=32,
        ic_values=ic,
        seed=0,
    )
    payload = provenance(
        schema="causal_marching/v1",
        config={"n_windows": 2, "steps_per_window": 20, "n_time_bins": 4},
    )
    payload.update(
        {
            "n_windows": len(result.windows),
            "final_loss": result.windows[-1].final_loss,
            "last_causality_index": result.windows[-1].causality.causality_index,
            "unlocked_fraction": result.windows[-1].causality.unlocked_fraction,
            "trivial": None
            if result.trivial is None
            else {
                "is_trivial": result.trivial.is_trivial,
                "ratio": result.trivial.ratio,
            },
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    assert np.isfinite(payload["final_loss"])
    write_json("causal_marching.json", payload)
    print("wrote docs/benchmarks/causal_marching.json")


if __name__ == "__main__":
    main()
