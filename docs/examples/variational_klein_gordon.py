# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Classical field theory: the Klein-Gordon field from least action.

Run:

    pip install omnibias-variational[torch]
    python docs/examples/variational_klein_gordon.py

For the Lorentz-invariant density ``L = 1/2((d_t phi)^2 - (d_x phi)^2) - 1/2 m^2
phi^2`` the field Euler-Lagrange equation ``sum_mu d_mu (dL/d(d_mu phi)) -
dL/dphi`` is the Klein-Gordon / wave operator ``phi_tt - phi_xx + m^2 phi``.
This script shows, on a plane wave ``phi = cos(k x - w t)``:

- *on shell* (``w^2 = k^2 + m^2``): the field EL residual vanishes -- the plane
  wave is a solution of least action;
- *off shell*: the residual is non-zero and reproduces the Klein-Gordon operator
  computed from the closed-form second derivatives;
- the stress-energy ``T^t_t`` equals the energy density
  ``1/2 phi_t^2 + 1/2 phi_x^2 + 1/2 m^2 phi^2``.

Mixed / second partials ``d_mu d_nu phi`` are closed form; the density partials
are autodiff.
"""

from __future__ import annotations

import math

import torch
from _variational_fields import PlaneWaveField
from omnibias.fields.torch.ops.basic import derivative, value
from omnibias.variational import LagrangianDensity
from omnibias.variational.torch import ops as var

K = 0.9
M = 0.7
XT = torch.tensor(
    [[0.1, 0.0], [0.5, 0.3], [-0.4, 0.8], [1.2, -0.6], [0.7, 1.1]],
    dtype=torch.float64,
)


def kg_density(m: float) -> LagrangianDensity:
    def fn(phi, dphi, x):
        phi_x = dphi[..., 0, 0]
        phi_t = dphi[..., 0, 1]
        return 0.5 * (phi_t**2 - phi_x**2) - 0.5 * m**2 * (phi[..., 0] ** 2)

    return LagrangianDensity(fn, fields=("phi",))


def main() -> None:
    torch.set_default_dtype(torch.float64)
    density = kg_density(M)

    # ---- on shell: the plane wave solves the field equation ---------------
    omega_on = math.sqrt(K**2 + M**2)
    state_on = PlaneWaveField(K, omega_on)(XT)
    res_on = var.field_euler_lagrange_residual(state_on, density)
    print(f"on-shell  (w^2 = k^2 + m^2):  max |field EL| = {res_on.abs().max().item():.2e}")
    assert res_on.abs().max().item() < 1e-10

    # ---- off shell: residual == Klein-Gordon operator ---------------------
    omega_off = 1.9
    state = PlaneWaveField(K, omega_off)(XT)
    res = var.field_euler_lagrange_residual(state, density)[:, 0]
    phi_tt = derivative(state, "phi", axis="t", order=2)
    phi_xx = derivative(state, "phi", axis="x", order=2)
    phi = value(state, "phi")
    kg = phi_tt - phi_xx + M**2 * phi
    print(f"off-shell (w = {omega_off}):  max |field EL - (phi_tt - phi_xx + m^2 phi)| "
          f"= {(res - kg).abs().max().item():.2e}")
    assert kg.abs().max().item() > 1e-3        # genuinely off-solution
    assert (res - kg).abs().max().item() < 1e-10

    # ---- stress-energy T^t_t is the energy density ------------------------
    t_tensor = var.stress_energy_tensor(state, density)  # (B, 2, 2), axes (x, t)
    phi_t = derivative(state, "phi", axis="t", order=1)
    phi_x = derivative(state, "phi", axis="x", order=1)
    energy_density = 0.5 * phi_t**2 + 0.5 * phi_x**2 + 0.5 * M**2 * phi**2
    err = (t_tensor[:, 1, 1] - energy_density).abs().max().item()
    print(f"max |T^t_t - energy density| = {err:.2e}")
    assert err < 1e-10
    print("\nOK: the Klein-Gordon plane wave is a least-action solution on shell.")


if __name__ == "__main__":
    main()
