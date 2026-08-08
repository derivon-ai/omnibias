# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CPU smoke: FBPINN vs plain one-layer on a high-frequency 1-D target.

Decision rule (fixed before the run): after a fixed Adam budget, FBPINN
MSE on ``sin(2 pi * 8 x)`` must be strictly below the plain one-layer MSE.
Also records the NTK spectral-bias index of each field.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch.fields import OneLayerVectorField, build_fbpinn_field
from omnibias.pinn.torch.losses import ntk_eigenspectrum, spectral_bias_index

DTYPE = torch.float64


def _train(field, coords, target, steps: int = 80, lr: float = 1e-2) -> float:
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = field.forward_values(coords)[:, 0]
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.mean((field.forward_values(coords)[:, 0] - target) ** 2))


def main() -> None:
    t0 = time.perf_counter()
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    x = torch.linspace(0.0, 1.0, 128, dtype=DTYPE).unsqueeze(-1)
    target = torch.sin(2 * np.pi * 8 * x[:, 0])

    plain = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=32, base="tanh", dtype=DTYPE
    )
    fbp = build_fbpinn_field(
        coordinate_spec=cs,
        components=comps,
        n_windows=4,
        hidden=8,
        frequency_scales=(1.0, 2.0, 4.0, 8.0),
        dtype=DTYPE,
    )
    mse_plain = _train(plain, x, target)
    mse_fbp = _train(fbp, x, target)

    def resid_plain():
        return plain.forward_values(x)[:, 0] - target

    def resid_fbp():
        return fbp.forward_values(x)[:, 0] - target

    ev_plain = ntk_eigenspectrum(resid_plain, list(plain.parameters()), n_eigen=8)
    ev_fbp = ntk_eigenspectrum(resid_fbp, list(fbp.parameters()), n_eigen=8)
    payload = provenance(
        schema="spectral_bias_fbpinn/v1",
        config={"steps": 80, "freq": 8, "n_windows": 4},
    )
    payload.update(
        {
            "mse_plain": mse_plain,
            "mse_fbpinn": mse_fbp,
            "spectral_bias_index_plain": spectral_bias_index(ev_plain),
            "spectral_bias_index_fbpinn": spectral_bias_index(ev_fbp),
            "fbpinn_wins": bool(mse_fbp < mse_plain),
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    assert mse_fbp < mse_plain
    write_json("spectral_bias_fbpinn.json", payload)
    print("wrote docs/benchmarks/spectral_bias_fbpinn.json")


if __name__ == "__main__":
    main()
