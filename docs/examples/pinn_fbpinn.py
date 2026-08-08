# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CI smoke: FBPINN multi-window field + spectral-bias index.

Honesty: FBPINN / NTK-spectrum tools are *numerical* mitigations and
measurements of spectral bias, not certificates that remove it.
"""

from __future__ import annotations

import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch.fields import build_fbpinn_field
from omnibias.pinn.torch.losses import ntk_eigenspectrum, spectral_bias_index


def main() -> None:
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = build_fbpinn_field(
        coordinate_spec=cs,
        components=comps,
        n_windows=3,
        overlap=0.5,
        hidden=8,
    )
    x = torch.linspace(0.05, 0.95, 48, dtype=torch.float64).unsqueeze(-1)
    w = field.window_weights(x)
    assert torch.allclose(
        w.sum(dim=-1), torch.ones(48, dtype=torch.float64), atol=1e-10
    )
    u = field.forward_values(x)
    assert u.shape == (48, 1)
    assert torch.isfinite(u).all()

    def residual_fn() -> torch.Tensor:
        return field.forward_values(x)[:, 0] - torch.sin(4.0 * torch.pi * x[:, 0])

    evals = ntk_eigenspectrum(residual_fn, list(field.parameters()), n_eigen=6)
    idx = spectral_bias_index(evals)
    assert 0.0 <= float(idx) <= 1.0
    print("pinn_fbpinn: ok", float(idx), float(evals[0]))


if __name__ == "__main__":
    main()
