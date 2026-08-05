# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven / verified / best-in-class smoke for the eps -> 0 rank/regularization collapse.

Run:

    pip install "omnibias-curvature[torch]"
    python docs/examples/certified_rank_collapse.py

Two halves, both CPU-tiny and deterministic (a fixed seed sweep), each an instrumented
experiment under the empirical-validation gates:

1. **Certified conditioning** (rigorous register, ``omnibias.core.verified.conditioning``):
   ``certified_condition_number`` *encloses* ``numpy.linalg.cond``; ``certified_damping``
   picks an ``eps`` for which ``kappa(A + eps I) <= target`` is re-verified by numpy;
   ``certified_regularization_error`` upper-bounds the true Tikhonov error; the sealed
   ``conditioning_certificate`` round-trips (tamper-evident digest).
2. **Min-norm collapse** (differentiable register, ``omnibias.curvature.regularize`` +
   the bit-identical ``omnibias.curvature.torch`` twin): ``min_norm_solve`` matches the
   best-in-class LAPACK baselines ``numpy.linalg.pinv`` / ``numpy.linalg.lstsq`` on
   rank-deficient systems; ``(A + eps I)^{-1} b -> A^+ b`` as ``eps -> 0`` (the measured
   ``regularization_path`` homotopy); and the JAX / torch certificates are byte-identical.

This is the ``eps -> 0`` rank/regularization collapse: a **distinct** limit from the
founding multi-bias ``delta -> 0`` derivative collapse and the ``beta -> inf`` feasibility
penalty -- never conflated. Honesty labels: the solves are **numerical** (LAPACK-class);
only the conditioning enclosure is **verified**; nothing here is "closed-form".
"""

from __future__ import annotations

import numpy as np

K_SEEDS = 8


def _spd(rng: np.random.Generator, p: int, cond: float | None = None) -> np.ndarray:
    a = rng.normal(size=(p, p))
    m = a @ a.T + 0.1 * np.eye(p)
    if cond is not None:  # rescale the spectrum to a target condition number
        _, v = np.linalg.eigh(m)
        w = np.linspace(1.0, cond, p)
        m = (v * w) @ v.T
    return 0.5 * (m + m.T)


def _low_rank(rng: np.random.Generator, p: int, r: int) -> np.ndarray:
    a = rng.normal(size=(p, r))
    m = a @ a.T
    return 0.5 * (m + m.T)


def probe_certified_conditioning() -> None:
    """certified_condition_number / _damping / _regularization_error are sound vs numpy."""
    from omnibias.core.proof.certificate import verify_certificate_digest
    from omnibias.core.verified.conditioning import (
        certified_condition_number,
        certified_damping,
        certified_regularization_error,
        conditioning_certificate,
    )

    print("=== 1. certified conditioning (rigorous register) vs numpy oracle ===")
    rng = np.random.default_rng(2026)
    worst_kappa_gap = 0.0
    for seed in range(K_SEEDS):
        p = 3 + (seed % 4)
        a = _spd(rng, p, cond=10.0 ** (1 + seed % 5))  # kappa 10 .. 1e5
        rows = a.tolist()

        k = certified_condition_number(rows)
        true_k = float(np.linalg.cond(a))
        assert k.lo <= true_k <= k.hi, f"seed {seed}: certified kappa band escaped"
        worst_kappa_gap = max(worst_kappa_gap, (k.hi - k.lo) / true_k)

        target = 50.0
        eps = certified_damping(rows, target_condition=target)
        assert np.linalg.cond(a + eps * np.eye(p)) <= target + 1e-6, f"seed {seed}: damping missed target"

        b = rng.normal(size=p)
        bound = certified_regularization_error(rows, b.tolist(), 1e-3)
        xe = np.linalg.solve(a + 1e-3 * np.eye(p), b)
        x0 = np.linalg.solve(a, b)
        assert bound.hi >= float(np.linalg.norm(xe - x0)) - 1e-12, f"seed {seed}: error bound not sound"

        cert = conditioning_certificate(rows, target_condition=target, eps=1e-3)
        assert verify_certificate_digest(cert)

    print(f"  K={K_SEEDS}: kappa enclosure contains numpy.linalg.cond (max rel width {worst_kappa_gap:.1e})")
    print("  certified damping meets kappa<=50 (numpy-rechecked); reg-error bound sound; certs sealed")

    # honesty: rank-deficient A -> kappa upper endpoint is +inf, not a silent finite lie
    a = _low_rank(rng, 5, 3)
    assert certified_condition_number(a.tolist()).hi == np.inf
    print("  rank-deficient A: certified kappa upper endpoint = +inf (honest)")


def probe_min_norm_collapse() -> None:
    """min_norm_solve matches pinv/lstsq; (A+epsI)^-1 b -> A^+ b; torch/jax certs identical."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from omnibias.curvature import regularize as jreg
    from omnibias.curvature.torch import regularize as treg

    print("\n=== 2. min-norm collapse (differentiable register) vs LAPACK pinv/lstsq ===")
    rng = np.random.default_rng(4242)
    worst_pinv = worst_lstsq = worst_path = 0.0
    for seed in range(K_SEEDS):
        p = 4 + (seed % 3)
        r = p - 1
        a = _low_rank(rng, p, r)
        y = rng.normal(size=p)
        b = a @ y  # range-consistent, so the eps -> 0 limit converges

        # best-in-class baselines
        x_pinv = np.linalg.pinv(a, rcond=1e-9) @ b
        x_lstsq = np.linalg.lstsq(a, b, rcond=1e-9)[0]

        xj = np.asarray(jreg.min_norm_solve(jnp.asarray(a), jnp.asarray(b), rcond=1e-9))
        worst_pinv = max(worst_pinv, float(np.linalg.norm(xj - x_pinv)))
        worst_lstsq = max(worst_lstsq, float(np.linalg.norm(xj - x_lstsq)))
        assert jreg.numerical_rank(jnp.asarray(a), rcond=1e-9) == r

        # the collapse homotopy: x(eps) -> x_pinv as eps -> 0
        grid = jnp.asarray([1e-1, 1e-3, 1e-5, 1e-7])
        path = np.asarray(jreg.regularization_path(jnp.asarray(a), jnp.asarray(b), grid))
        worst_path = max(worst_path, float(np.linalg.norm(path[-1] - x_pinv)))

    print(f"  K={K_SEEDS}: min_norm_solve vs numpy.pinv max err {worst_pinv:.1e}; vs lstsq {worst_lstsq:.1e}")
    print(f"  regularization_path tail (eps=1e-7) vs pinv max err {worst_path:.1e} (eps -> 0 convergence)")
    assert worst_pinv < 1e-6 and worst_lstsq < 1e-6 and worst_path < 1e-4

    # torch twin: same matrix -> byte-identical certified damping + sealed certificate
    a = _spd(rng, 5, cond=1e4)
    b = rng.normal(size=5)
    rj = jreg.rank_collapse(jnp.asarray(a), jnp.asarray(b), target_condition=100.0)
    rt = treg.rank_collapse(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64),
                            target_condition=100.0)
    assert rj.eps == rt.eps
    assert rj.certificate is not None and rt.certificate is not None
    assert rj.certificate["digest"] == rt.certificate["digest"]
    print(f"  rank_collapse: jax/torch certified eps={rj.eps:.3e} and certificate digest are byte-identical")


def main() -> None:
    probe_certified_conditioning()
    probe_min_norm_collapse()
    print("\nOK: certified rank/regularization collapse (eps -> 0) passes its gates.")


if __name__ == "__main__":
    main()
