# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable Smith-Waterman local + Gotoh affine-gap alignment -- omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/local_affine_alignment.py

Both aligners are longest-path DPs on an *augmented* DAG, so they reuse the exact same
``soft_shortest_path`` substrate as global Needleman-Wunsch:

* **Smith-Waterman (local).** Free start/end 0-edges (a super-source into every cell and
  every cell into a super-sink) let the alignment begin and end anywhere; the empty
  alignment (score 0) is the floor. ``-> the local optimum`` as ``beta -> inf``.
* **Gotoh (affine gaps).** A 3-state ``M / Ix / Iy`` lattice so a gap of length ``L`` costs
  ``gap_open + L * gap_extend`` instead of ``L * gap`` -- one open edge (charged
  ``gap_open + gap_extend``) then extend edges (charged ``gap_extend``).

On both axes: ``beta -> inf`` is the relaxation (``lse_beta -> max``, certified ``log(N)/beta``
gap vs the classic DP + brute force), ``delta -> 0`` is the founding tower that differentiates
``lse_beta`` exactly -- the closed-form substitution / gap-open / gap-extend usage marginals
equal ``autograd`` / ``jax.grad``, and the torch / jax twins are bit-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    brute_force_gotoh,
    brute_force_local_align,
    build_gotoh_dag,
    build_local_dag,
    certify_soft_dp,
    hard_gotoh,
    hard_local_align,
)
from omnibias.struct.jax import align as jalign  # noqa: E402
from omnibias.struct.torch import align as talign  # noqa: E402

torch.set_default_dtype(torch.float64)


def _sub(seed: int, k: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((k, k))
    s = 0.5 * (s + s.T)
    s[np.arange(k), np.arange(k)] += 2.0  # reward matches
    return s


def local_demo() -> None:
    print("=== 1. Smith-Waterman local alignment: soft anneals to the classic DP ===")
    sub = _sub(0)
    a = np.array([0, 1, 2, 3, 1])
    b = np.array([3, 1, 2, 0])
    gap = -1.0
    hard = hard_local_align(a, b, sub, gap)
    brute = brute_force_local_align(a, b, sub, gap)
    n_paths = build_local_dag(len(a), len(b))[0].count_paths()
    assert abs(hard - brute) < 1e-9, "classic SW DP must equal brute-force enumeration"
    print(f"  hard local score V* = {hard:.4f} = brute force {brute:.4f}  (over {n_paths} sub-alignments)")
    print(f"  {'beta':>6s} {'soft (lse)':>12s} {'gap':>9s} {'log(N)/beta':>12s}  sound")
    st, gt = torch.tensor(sub), torch.tensor(gap)
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(talign.soft_local_align(a, b, st, gt, beta))
        cert = certify_soft_dp(hard, soft, n_paths, beta, brute_force_value=brute)
        print(f"  {beta:6.1f} {soft:12.4f} {cert.absolute_gap:9.4f} {cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap
    print()


def gotoh_demo() -> None:
    print("=== 2. Gotoh affine gaps (M/Ix/Iy): a length-L gap costs open + L*extend ===")
    sub = _sub(1)
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 3])
    gap_open, gap_extend = -1.5, -0.4
    hard = hard_gotoh(a, b, sub, gap_open, gap_extend)
    brute = brute_force_gotoh(a, b, sub, gap_open, gap_extend)
    n_paths = build_gotoh_dag(len(a), len(b))[0].count_paths()
    assert abs(hard - brute) < 1e-9, "classic Gotoh DP must equal brute-force enumeration"
    print(f"  hard affine score V* = {hard:.4f} = brute force {brute:.4f}  (open={gap_open}, extend={gap_extend})")
    print(f"  {'beta':>6s} {'soft (lse)':>12s} {'gap':>9s} {'log(N)/beta':>12s}  sound")
    st = torch.tensor(sub)
    ot, et = torch.tensor(gap_open), torch.tensor(gap_extend)
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(talign.soft_gotoh(a, b, st, ot, et, beta))
        cert = certify_soft_dp(hard, soft, n_paths, beta, brute_force_value=brute)
        print(f"  {beta:6.1f} {soft:12.4f} {cert.absolute_gap:9.4f} {cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap
    print()


def marginals_demo() -> None:
    print("=== 3. closed-form usage marginals == autograd + torch/jax parity ===")
    sub = _sub(2)
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 2, 3])
    beta = 4.0

    # ---- local: (d/d sub, d/d gap) ----
    st = torch.tensor(sub, requires_grad=True)
    gt = torch.tensor(-1.0, requires_grad=True)
    talign.soft_local_align(a, b, st, gt, beta).backward()
    ls, lg = talign.soft_local_align_marginals(a, b, st.detach(), gt.detach(), beta)
    err = max(float((ls - st.grad).abs().max()), abs(float(lg) - float(gt.grad)))
    print(f"  local   : max|marginal - autograd| = {err:.2e}")
    assert err < 1e-9

    # ---- gotoh: (d/d sub, d/d open, d/d extend) ----
    st2 = torch.tensor(sub, requires_grad=True)
    ot = torch.tensor(-1.5, requires_grad=True)
    et = torch.tensor(-0.4, requires_grad=True)
    talign.soft_gotoh(a, b, st2, ot, et, beta).backward()
    gs, go, ge = talign.soft_gotoh_marginals(a, b, st2.detach(), ot.detach(), et.detach(), beta)
    err2 = max(
        float((gs - st2.grad).abs().max()),
        abs(float(go) - float(ot.grad)),
        abs(float(ge) - float(et.grad)),
    )
    print(f"  gotoh   : max|marginal - autograd| = {err2:.2e}")
    assert err2 < 1e-9

    # ---- torch <-> jax parity (both families) ----
    sj = jnp.asarray(sub)
    lp = abs(
        float(talign.soft_local_align(a, b, torch.tensor(sub), torch.tensor(-1.0), beta))
        - float(jalign.soft_local_align(a, b, sj, jnp.asarray(-1.0), beta))
    )
    gp = abs(
        float(talign.soft_gotoh(a, b, torch.tensor(sub), torch.tensor(-1.5), torch.tensor(-0.4), beta))
        - float(jalign.soft_gotoh(a, b, sj, jnp.asarray(-1.5), jnp.asarray(-0.4), beta))
    )
    print(f"  parity  : local {lp:.2e}   gotoh {gp:.2e}")
    assert lp < 1e-9 and gp < 1e-9
    print()


def main() -> None:
    local_demo()
    gotoh_demo()
    marginals_demo()
    print(
        "OK: Smith-Waterman + Gotoh reuse the shortest-path substrate; soft == brute force, "
        "certified log(N)/beta gap, marginals == autograd, torch/jax parity < 1e-9."
    )


if __name__ == "__main__":
    main()
