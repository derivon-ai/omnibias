# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation smoke for the closed-form integral transforms of activations.

Run:

    pip install "omnibias-torch" "omnibias-jax" "omnibias-measure[torch]"
    JAX_PLATFORMS=cpu python docs/examples/transforms_closed_form.py

omnibias computes every *derivative* of an activation in closed form. For a
subset of the dictionary the *integral transforms* are elementary too, and this
smoke is the evidence: nine Laplace kernels, two Fourier kernels and one Mellin
kernel, each evaluated in a single elementary expression -- no quadrature, no
series truncation, no iteration -- and each checked against the integral it
claims to equal.

The gaps carry as much weight as the entries. ``sigmoid`` has a Laplace
transform but not a Fourier one (it is not L^1, and its transform is a Dirac
mass plus a principal value); ``exp`` has a Laplace transform but not a Mellin
one (omnibias's exp is e^{+z}, and the textbook Gamma(s) pair belongs to the
decaying exponential). This script prints the reason attached to each refusal,
because a kernel that silently returns the wrong object is worse than one that
is absent.

Gates exercised, in the ``omnibias-dev-empirical-validation`` sense:

* **analytic oracle** -- the two self-reciprocal profiles, Gaussian -> Gaussian
  and sech -> sech, reproduced to machine epsilon;
* **numerical oracle** -- every kernel against omnibias-measure's
  ``lebesgue_integral`` of its own defining integrand;
* **best-in-class numerics** -- the Gaussian Laplace kernel stays finite and
  accurate at s = 100, where the textbook ``exp(s^2/2) erfc(s/sqrt 2)`` form
  returns nan;
* **honest failure** -- unregistered pairs raise with the recorded mathematical
  reason, and the Fermi-Dirac integral refuses to cross the same Re(s) > 1 wall
  that ``omnibias.core.verified.dirichlet`` enforces;
* **parity** -- the torch and jax kernels agree to float64 round-off;
* **train-through** -- a ``LaplaceTransform`` layer trains, and its softplus
  reparameterisation keeps a learnable spectral variable inside the region of
  convergence under a deliberately hostile step.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.core.transforms import (
    FERMI_DIRAC_MELLIN,
    TRANSFORM_NAMES,
    find_exclusion,
    identities,
)
from omnibias.torch.activations import get_activation
from omnibias.torch.transforms import (
    LaplaceTransform,
    fermi_dirac_mellin,
    fourier_transform,
    laplace_transform,
    mellin_transform,
    region_of_convergence,
)

torch.set_default_dtype(torch.float64)

_FN = {"laplace": laplace_transform, "fourier": fourier_transform, "mellin": mellin_transform}

#: Points strictly inside each region of convergence.
SAMPLES: dict[tuple[str, str], tuple[float, ...]] = {
    ("exp", "laplace"): (1.5, 3.7),
    ("relu", "laplace"): (0.5, 2.5),
    ("sin", "laplace"): (0.4, 2.5),
    ("cos", "laplace"): (0.4, 2.5),
    ("sinh", "laplace"): (1.3, 4.0),
    ("cosh", "laplace"): (1.3, 4.0),
    ("gaussian", "laplace"): (-1.0, 0.5, 5.0),
    ("sigmoid", "laplace"): (0.3, 2.5),
    ("tanh", "laplace"): (0.3, 2.5),
    ("sech", "laplace"): (-0.5, 4.0),
    ("gaussian", "fourier"): (0.0, 1.7),
    ("sech", "fourier"): (0.0, 1.7),
    ("gaussian", "mellin"): (0.5, 4.5),
}

#: Exponential growth rate on the positive half line, used to size the
#: truncation window: the Laplace integrand decays like exp(-(s - growth) z).
GROWTH = {"exp": 1.0, "sinh": 1.0, "cosh": 1.0, "sech": -1.0}


def measure_integral(integrand, lower: float, upper: float, nodes: int = 800) -> float:
    """``int_lower^upper f`` on a Gauss-Legendre Lebesgue measure."""
    from omnibias.measure._core.integrate import lebesgue_integral
    from omnibias.measure._core.measure import lebesgue

    measure = lebesgue([(lower, upper)], nodes)
    return float(lebesgue_integral(lambda z: integrand(z[:, 0]), measure))


def numeric_transform(name: str, transform: str, point: float) -> float:
    forward = lambda z: get_activation(name).forward(torch.from_numpy(z)).numpy()  # noqa: E731
    if transform == "laplace":
        rate = point - GROWTH.get(name, 0.0)
        upper = 45.0 if name == "gaussian" else max(40.0, 40.0 / rate)
        return measure_integral(lambda z: forward(z) * np.exp(-point * z), 0.0, upper)
    if transform == "fourier":
        return measure_integral(lambda z: forward(z) * np.cos(point * z), -40.0, 40.0)
    # Mellin, substituting z = exp(u) so the z^(s-1) endpoint singularity vanishes.
    return measure_integral(
        lambda u: forward(np.exp(u)) * np.exp(point * u), -40.0 / point, 5.0
    )


def check_against_quadrature() -> None:
    print("closed form vs omnibias-measure quadrature of the defining integral")
    print(f"  {'pair':24s} {'region':12s} {'max rel err':>12s}")
    worst_overall = 0.0
    for transform in TRANSFORM_NAMES:
        for identity in identities(transform):
            name = identity.activation
            worst = 0.0
            for point in SAMPLES[(name, transform)]:
                closed = float(_FN[transform](name, torch.tensor(point)))
                numeric = numeric_transform(name, transform, point)
                worst = max(worst, abs(closed - numeric) / max(abs(numeric), 1e-300))
            worst_overall = max(worst_overall, worst)
            print(f"  {name + '/' + transform:24s} {identity.region:12s} {worst:12.2e}")
    if worst_overall > 1e-8:
        raise SystemExit(f"a kernel disagrees with its own integral (rel {worst_overall:.2e})")


def check_analytic_pairs() -> None:
    print("\nself-reciprocal profiles (the classical analytic pairs)")
    xi = torch.linspace(-4.0, 4.0, 17)
    gauss_err = float(
        (fourier_transform("gaussian", xi) - math.sqrt(2 * math.pi) * torch.exp(-0.5 * xi * xi))
        .abs()
        .max()
    )
    sech_err = float(
        (fourier_transform("sech", xi) - math.pi / torch.cosh(0.5 * math.pi * xi)).abs().max()
    )
    print(f"  F[gaussian] = sqrt(2 pi) exp(-xi^2/2)   max abs err = {gauss_err:.2e}")
    print(f"  F[sech]     = pi sech(pi xi / 2)        max abs err = {sech_err:.2e}")
    if max(gauss_err, sech_err) > 1e-14:
        raise SystemExit("a self-reciprocal identity does not hold to machine precision")


def check_numerical_conditioning() -> None:
    print("\nthe erfcx route vs the textbook exp(s^2/2) erfc(s/sqrt 2) form")
    for s in (5.0, 40.0, 100.0):
        t = torch.tensor(s)
        naive = math.sqrt(math.pi / 2) * torch.exp(0.5 * t * t) * torch.special.erfc(
            t / math.sqrt(2.0)
        )
        got = laplace_transform("gaussian", t)
        print(f"  s = {s:5.0f}   textbook = {float(naive):>12.6e}   omnibias = {float(got):.6e}")
    if not torch.isfinite(laplace_transform("gaussian", torch.tensor(100.0))):
        raise SystemExit("the erfcx route should stay finite at s = 100")


def show_honest_gaps() -> None:
    print("\nwhat is deliberately NOT registered, and why")
    for name, transform in (
        ("sigmoid", "fourier"),
        ("relu", "fourier"),
        ("exp", "mellin"),
        ("sigmoid", "mellin"),
        ("sin", "mellin"),
        ("sech", "mellin"),
    ):
        excluded = find_exclusion(name, transform)
        assert excluded is not None
        print(f"  {name + '/' + transform:20s} {excluded.reason}")
        try:
            _FN[transform](name, torch.tensor(2.0))
        except TypeError:
            pass
        else:
            raise SystemExit(f"{name}/{transform} should have refused")

    print("\nthe Fermi-Dirac companion (Mellin transform of 1 - sigmoid)")
    print(f"  {FERMI_DIRAC_MELLIN.expression}   for {FERMI_DIRAC_MELLIN.region}")
    for s in (1.2, 3.0):
        numeric = measure_integral(
            lambda u, v=s: np.exp(v * u) / (1.0 + np.exp(np.exp(u))), -40.0 / s, 5.0
        )
        closed = float(fermi_dirac_mellin(torch.tensor(s)))
        print(f"  s = {s:4.1f}   closed = {closed:.10f}   quadrature = {numeric:.10f}")
        if abs(closed - numeric) / numeric > 1e-9:
            raise SystemExit("the Fermi-Dirac integral disagrees with quadrature")
    try:
        fermi_dirac_mellin(torch.tensor(0.8))
    except ValueError:
        print("  s = 0.8   refused: the verified zeta scope stops at Re(s) > 1")
    else:
        raise SystemExit("the Re(s) > 1 scope wall was not enforced")


def check_backend_parity() -> None:
    print("\ntorch vs jax parity")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.transforms import laplace_transform as jax_laplace

    worst = 0.0
    for transform in TRANSFORM_NAMES:
        for identity in identities(transform):
            if transform != "laplace":
                continue
            points = SAMPLES[(identity.activation, transform)]
            a = laplace_transform(identity.activation, torch.tensor(points)).numpy()
            b = np.asarray(jax_laplace(identity.activation, jnp.asarray(points)))
            worst = max(worst, float(np.max(np.abs(a - b) / np.abs(b))))
    print(f"  max relative difference over every Laplace kernel = {worst:.2e}")
    if worst > 1e-13:
        raise SystemExit("the backends disagree by more than float64 round-off")


def train_a_transform_layer() -> None:
    print("\na closed-form transform as a trainable layer")
    torch.manual_seed(0)
    block = LaplaceTransform("sigmoid", features=1, init_shift=4.0)
    target = laplace_transform("sigmoid", torch.tensor(1.25))
    optimizer = torch.optim.Adam(block.parameters(), lr=0.05)
    x = torch.zeros(1)
    for _ in range(400):
        optimizer.zero_grad()
        loss = (block(x) - target).pow(2).sum()
        loss.backward()
        optimizer.step()
    recovered = float(block.argument(x).detach()[0])
    print(f"  recovered s = {recovered:.6f} (target 1.25), final loss = {float(loss):.3e}")
    if abs(recovered - 1.25) > 1e-3:
        raise SystemExit("the layer failed to recover the spectral point")

    lower, region = region_of_convergence("relu", "laplace")
    guard = LaplaceTransform("relu", features=3)
    with torch.no_grad():  # a deliberately hostile step
        guard.raw_shift.copy_(torch.tensor([-1e3, -50.0, 1e3]))
    argument = guard.argument(torch.zeros(3)).detach()
    print(f"  after a -1000 step the relu Laplace argument is {argument.tolist()}")
    print(f"  the softplus reparameterisation holds it inside '{region}'")
    if lower is None or bool((argument < lower).any()):
        raise SystemExit("a learnable spectral variable escaped its region of convergence")


def main() -> None:
    check_against_quadrature()
    check_analytic_pairs()
    check_numerical_conditioning()
    show_honest_gaps()
    check_backend_parity()
    train_a_transform_layer()
    print("\nall closed-form transform checks passed.")


if __name__ == "__main__":
    main()
