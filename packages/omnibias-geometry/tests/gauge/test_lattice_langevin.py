# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(2) lattice Langevin / stochastic-quantisation updater.

The decisive correctness check is the single-link fluctuation-dissipation test:
with a *fixed* staple the geodesic Langevin chain reproduces the exact SU(2)
conditional ``exp(beta (U . Sigma))`` (the same distribution the heat bath samples)
to within the O(eps) discretisation bias. The full-lattice checks confirm the
sweep stays on SU(2) and orders the field toward the heat-bath plaquette (the
residual gap is the standard dimension-dependent unadjusted-Langevin bias, which
shrinks as ``eps -> 0`` -- the continuum stochastic-quantisation limit).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.geometry.gauge.lattice._core import kernels
from omnibias.geometry.gauge.lattice.langevin import langevin_sweep, langevin_update_links
from omnibias.geometry.gauge.lattice.observables import average_plaquette
from omnibias.geometry.gauge.lattice.su2 import random_links, sweep


def _exact_mean_q0(w: float, n: int = 200001) -> float:
    """Exact ``<q0>`` for the SU(2) marginal density ``exp(w q0) sqrt(1-q0^2)``."""
    q = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    f = torch.exp(w * q) * torch.sqrt((1.0 - q * q).clamp_min(0.0))
    return float((q * f).sum() / f.sum())


def test_langevin_link_step_reproduces_exact_conditional() -> None:
    a, beta, eps = 2.0, 1.0, 0.01
    gen = torch.Generator().manual_seed(0)
    u = torch.nn.functional.normalize(
        torch.randn(8000, 4, generator=gen, dtype=torch.float64), dim=-1
    )
    sigma = (a * torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)).expand(8000, 4).clone()
    for _ in range(400):
        xi = torch.randn(8000, 3, generator=gen, dtype=torch.float64)
        u = kernels.langevin_link_step(torch, u, sigma, beta, eps, xi)
    assert abs(float(u[:, 0].mean()) - _exact_mean_q0(beta * a)) < 0.02


def test_langevin_preserves_su2() -> None:
    gen = torch.Generator().manual_seed(1)
    links = random_links((4, 4, 4, 4), generator=gen)
    for _ in range(5):
        langevin_sweep(links, beta=1.5, eps=0.05, generator=gen, n_sub=1)
    norms = torch.linalg.vector_norm(links, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)


def test_langevin_update_accepts_explicit_noise() -> None:
    gen = torch.Generator().manual_seed(4)
    links = random_links((4, 4, 4, 4), generator=gen)
    from omnibias.geometry.gauge.lattice.su2 import checkerboard_mask

    mask = checkerboard_mask((4, 4, 4, 4), 0, device=links.device, dtype=links.dtype)
    noise = torch.zeros((int(mask.sum()), 3), dtype=links.dtype)
    before = links.clone()
    langevin_update_links(links, 0, mask, beta=1.0, eps=0.02, noise=noise)
    # pure-drift (zero noise) move must still keep SU(2) and change the masked links
    norms = torch.linalg.vector_norm(links, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)
    assert not torch.allclose(links[0][mask], before[0][mask])


@pytest.mark.slow
def test_langevin_orders_lattice_toward_heatbath() -> None:
    beta = 1.0
    gen = torch.Generator().manual_seed(2)
    links = random_links((4, 4, 4, 4), generator=gen)
    p_random = average_plaquette(links)
    for _ in range(250):
        langevin_sweep(links, beta, eps=0.02, generator=gen, n_sub=4)
    p_lang = sum(
        average_plaquette(langevin_sweep(links, beta, 0.02, generator=gen, n_sub=4) or links)
        for _ in range(40)
    ) / 40

    hgen = torch.Generator().manual_seed(3)
    hb = random_links((4, 4, 4, 4), generator=hgen)
    for _ in range(40):
        sweep(hb, beta, generator=hgen)
    p_hb = sum(average_plaquette(sweep(hb, beta, generator=hgen) or hb) for _ in range(30)) / 30

    assert p_lang > p_random + 0.15  # ordered the field away from the random start
    assert p_lang < p_hb + 0.05  # unadjusted Langevin undershoots, never overshoots
    assert (p_hb - p_lang) < 0.15  # same ballpark as the heat bath
