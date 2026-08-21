# CCF self-similar singularities (CAP-ready, honesty-first)

This page documents the Córdoba–Córdoba–Fontelos (CCF) self-similar
discovery pipeline shipped in `omnibias-pinn` and validated by
`omnibias-symbolic`. It is written **honesty-first**: every claim below is
either a measured number you can reproduce from this repository, a cited
published value, or an explicit *non-claim*.

!!! info "Scope — read first"
    The repository ships **two** CCF residual domains:

    1. **Periodic torus** (`CordobaCordobaFontelos`) — tractable spectral Hilbert
       model; does **not** by itself reproduce published line-domain \(\lambda\).
    2. **Line / compactified** (`CordobaCordobaFontelosCompactified` + Hardy
       `hilbert_mode="hardy_exact"`) — lambda-tied compactification
       \(q=(1+y^2)^{-1/(2(1+\lambda))}\), Cauchy–Hardy exact Hilbert pair
       \(P/Q\) with \(\alpha=1/(1+\lambda)\), residual factorisation
       \(\mathcal{E}=F\cdot\mathcal{R}\). This is the DeepMind-style substrate
       aimed at published admissible \(\lambda\). The truncated FFT Hilbert path
       remains for periodic CCF only.

    Neither is viscous Navier–Stokes. Deliverables: bit-parity residual operators,
    discovery harnesses, CAP export, and an independent numpy validator. Honesty
    rules live in
    [Scope & guarantees § 3 / § 6](../scope-and-guarantees.md#3-the-certified-pde-stacks-navierstokes-ccf).

## The model and the contract

The CCF equation is the nonlocal transport model of Córdoba, Córdoba &
Fontelos with velocity given by the Hilbert transform \(H\):

$$
\theta_t + (H\theta)\,\theta_x = 0 .
$$

We look for a self-similar blow-up at \(t \to 1^-\) using the ansatz of
Wang et al. (arXiv:2509.14185, eq. 2):

$$
\theta(x,t) = (1-t)^{\lambda}\,\Theta(y), \qquad
y = (1-t)^{-(1+\lambda)}\,x .
$$

Substituting and cancelling the common \((1-t)^{\lambda-1}\) factor (the
algebra is \(t\)-independent — verified symbolically) gives the **stationary
profile equation** in *transport* form:

$$
\mathcal{E}(\Theta,\lambda)\;=\;
(1+\lambda)\,y\,\Theta'(y) - \lambda\,\Theta(y) + (H\Theta)(y)\,\Theta'(y) \;=\; 0 ,
$$

and the algebraically equivalent *flux* form
\((1+\lambda)\,y\,\Theta' - \lambda\,\Theta + \big(\Theta' H\Theta + \Theta\, H\Theta'\big) = 0\).
Both forms are implemented and tested for agreement. The full contract
(parity gauge, far-field condition, sign conventions, published \(\lambda\)
values) lives in the docstring of the residual module
`omnibias.pinn.jax.equations.cordoba_cordoba_fontelos`.

| Contract item | Choice |
|---|---|
| Exponent gauge | \(k(\lambda)=\lambda\) (amplitude decays as \((1-t)^\lambda\)) |
| Parity | even \(\Theta\) (odd velocity \(H\Theta\)); all residual terms even |
| Domain (numerics) | periodic torus \([-\pi,\pi)\) *or* compactified line (see below) |
| Published \(\lambda\) | stable; \(\lambda_1\approx0.6057\); \(\lambda_2\approx0.4703\) (line domain) |
| Residual norms | \(\max|\mathcal{E}|\) and RMS over the collocation grid |

## Line / compactified domain

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.pinn.jax.equations.ccf_compactified import (
    compactified_grid,
    ccf_compactified_residual_samples,
    apply_envelope,
)

q, y = compactified_grid(128, q_max=0.99)
# hat profile in q-space (even): use cos features of q, lift with decay envelope
hat = jnp.cos(jnp.pi * q)
hat_y = -jnp.pi * jnp.sin(jnp.pi * q) * (1.0 + y * y) ** (-1.5)  # chain rule via dq/dy
theta, theta_y = apply_envelope(y, hat, hat_y, power=1.0)
equation, factored, weight = ccf_compactified_residual_samples(
    y, theta, theta_y, lam=0.6057
)
assert equation.shape == y.shape
```

Hilbert on the **default line discovery path** is the **Cauchy–Hardy exact
pair** (`hilbert_mode="hardy_exact"`, \(\alpha=1/(1+\lambda)\)). The truncated
FFT path remains available for periodic / diagnostic use and is honesty-labelled
numerical — not closed-form.

## What is closed-form and what is not

`omnibias` supplies the **exact** closed-form derivative tower for the
profile network, so \(\Theta,\Theta',\Theta''\) carry no finite-difference
error. The nonlocal velocity is a different animal:

!!! note "Honest labelling of the Hilbert term"
    The local terms \(\Theta'\) use the omnibias closed-form derivative
    fast-path. The Hilbert transform `omnibias.pinn.jax.hilbert.hilbert_transform`
    is a **periodic spectral Fourier multiplier** \(-i\,\mathrm{sgn}(m)\)
    (mean and even-\(N\) Nyquist modes zeroed). It is *exact for band-limited
    periodic data* and is **not** an omnibias closed-form derivative. It is
    labelled as numerical throughout.

## Running the pipeline

Residual operator (JAX; the Torch twin is bit-identical):

```python
import jax; jax.config.update("jax_enable_x64", True)
from omnibias.pinn.jax.discovery import ccf, cap

cfg = ccf.CCFDiscoveryConfig(hidden=32, n_grid=256, parity="even", lam_init=0.6057)
result = ccf.run_ccf_discovery(cfg, steps=1500, lr=3e-3)
print(result.diagnostics["max_abs_residual"], result.diagnostics["rms_residual"])
```

Line / compactified discovery (Hardy exact-Hilbert substrate):

```python
import jax; jax.config.update("jax_enable_x64", True)
from omnibias.pinn.jax.discovery import ccf_line, cap
from omnibias.symbolic import verify_cap_bundle

cfg = ccf_line.CCFLineDiscoveryConfig(
    n_terms=4, n_grid=48, y_max=12.0, optimizer="adam", lam_init=0.6057
)
result = ccf_line.run_ccf_line_discovery(cfg, steps=40, lr=5e-3, funnel_updates=0)
bundle = cap.build_cap_bundle(result)
assert cap.cap_schema_errors(bundle) == []
assert verify_cap_bundle(bundle)["residual_samples_match"]
```

CPU smoke benchmark: `python benchmarks/ccf_line_discovery.py` writes
`docs/benchmarks/ccf_line_smoke.json` with a `gates` block.

Export an interval-friendly CAP bundle and verify it with the *independent*
numpy validator (no JAX/Torch, no shared code path):

```python
from omnibias.symbolic import verify_cap_bundle, recover_ccf_scaling_law

bundle = cap.build_cap_bundle(result, reproduces_published_lambda=None)
assert cap.cap_schema_errors(bundle) == []
cap.write_cap_bundle(bundle, "out/ccf")          # ccf_cap.json + summary.md

report = verify_cap_bundle(bundle)                # recompute residual from scratch
assert report["residual_samples_match"]
```

The CAP bundle stores the network-free description an external
interval-arithmetic checker needs: grid, \(\Theta\), \(\Theta'\), \(\lambda\),
form, the residual samples, max/RMS/far-field diagnostics, a spectral-tail
indicator, dtype/platform metadata, and a band-limited Fourier representation
with an \(L^1\) tail bound for the discarded modes.

### Method of manufactured solutions (the rigorous check)

Because the periodic toy model is not expected to reproduce the line-domain
\(\lambda\) (see below), the harness is validated by **manufacturing** a known
exact solution: pick \(\Theta^\*\), compute the forcing
\(g = \mathcal{E}(\Theta^\*,\lambda^\*)\), then recover \(\Theta^\*\) by
optimising \(\mathcal{E}[\Theta]=g\). Symbolic regression reads \(\lambda^\*\)
back off the recovered law.

```python
from omnibias.pinn.jax.discovery import ccf
from omnibias.symbolic import recover_ccf_scaling_law

cfg = ccf.CCFDiscoveryConfig(hidden=32, n_grid=256, parity="even", lam_init=0.6057)
theta_star = ccf.default_manufactured_profile()
g, th, th_y = ccf.manufactured_forcing(cfg, theta_star, 0.5)
law = recover_ccf_scaling_law(ccf.make_grid(cfg), th, th_y, forcing=g)
print(law["lambda_recovered"])   # -> 0.5 to ~1e-11
```

## Measured results (reproducible in this repo)

All numbers below are from JAX `float64` runs on CPU; the cross-backend rows
compare against the Torch `float64` twin.

| Quantity | Measured | Meaning |
|---|---|---|
| Hilbert \(H[\cos],H[\sin]\) error | \(<10^{-10}\) | spectral convention correct |
| Hilbert JAX↔Torch parity | \(8.9\times10^{-16}\) | bit-parity (round-off) |
| CCF residual op, exact substitution | \(0.0\) | operator matches hand algebra |
| CCF residual JAX↔Torch parity | \(\sim4\times10^{-16}\) | bit-parity (round-off) |
| **MMS** loss reduction | \(2.0 \to 1.6\times10^{-4}\) | harness recovers a known profile |
| **MMS** \(\Theta\) RMSE vs \(\Theta^\*\) | \(1.4\times10^{-2}\) | profile recovered (finite run) |
| **MMS** \(\lambda\) recovery error | \(2.6\times10^{-11}\) | symbolic law reads \(\lambda^\*\) back |
| CAP independent recompute diff | \(0.0\) | numpy validator agrees with JAX |
| CAP Fourier tail bound | \(\sim2\times10^{-12}\) | discarded-mode \(L^1\) bound |
| Conjugate \(N=0\) dense \(\max|\mathcal{E}|\) (full) | \(8.96\times10^{-2}\) | Q-only Hardy-Ω floor, 10 atoms |
| Conjugate best \(N=3\) (full) | \(7.48\times10^{-2}\) | \(1.20\times\) vs \(N=0\), not \(10\times\) |
| Conjugate matched-width \(N=4\) | \(8.99\times10^{-2}\) | same 10 atoms as \(N=0\); G2 ratio \(0.996\) |
| Conjugate empirical \(p\) (\(N=0\ldots4\)) | \(0.05\) | flat; dictionary order is not the floor |
| Conjugate Gram \(\kappa\) | \(10^{14}\)–\(10^{29}\) | conditioning caps usable \(N\) |
| Conjugate whole-line CAP | BLOCKED | residual \(\sim 10^{-1}\); both NK fail |
| Conjugate `orders_to_stretch` (raw) | \(11.87\) | stretch \(10^{-13}\) unearned |

## Honest comparison to the published baseline

Wang et al. (arXiv:2509.14185) report CCF self-similar profiles on the line
at **near machine precision** (residuals \(\sim10^{-13}\)) with
\(\lambda_1\approx0.6057\), \(\lambda_2\approx0.4703\), etc.

| Axis | Published (arXiv:2509.14185) | This repo |
|---|---|---|
| Domain | line \(\mathbb{R}\) | periodic torus (model) |
| Best residual | \(\sim10^{-13}\) | MMS forced \(\sim10^{-4}\) (short run); operator exact |
| \(\lambda\) source | line eigenvalue problem | recovered exactly **only** for manufactured solutions |
| Validation | their CAP | independent numpy recompute + symbolic law |

!!! danger "Non-claims"
    - The unforced periodic run does **not** reproduce the published
      line-domain \(\lambda\); its residual stays \(O(1)\) on the torus
      (max \(\approx1.3\), RMS \(\approx0.55\) for the example config). This
      is an expected domain mismatch, reported plainly rather than tuned away.
    - No exact symbolic CCF solution is claimed. `assess_ccf_candidate`
      returns `exact_solution_claim=False`.
    - No Navier–Stokes result is claimed. The CAP bundle carries
      `navier_stokes_proof_claim=False`.
    - Enlarging the Cauchy–Hardy dictionary to derivative order \(N>0\) is a
      CCF residual experiment. It does not earn stretch \(10^{-13}\) or Rung-1
      \(10^{-11}\) unless the measured dense residual and both Newton–Kantorovich
      closures actually pass. `whole_line_certified` is not a 3-D NS claim.

## DeepMind recipe vs omnibias (architecture and optimizer)

Wang et al., [Discovery of Unstable Singularities](https://arxiv.org/abs/2509.14185)
and the follow-up [gradient-normalized residual](https://arxiv.org/abs/2511.22819)
are the source, not a popular summary. The Medium write-up is a paraphrase.

| Ingredient | DeepMind paper | Omnibias |
|---|---|---|
| Compactified even coords \(q=(1+y^2)^{-\alpha/2}\) | yes | `compactify_y_lambda` |
| Envelope / odd lift \(\Omega=y\cdot E\cdot\mathrm{hat}\) | yes | `apply_envelope` / `omega_from_net` |
| Small \(\tanh\) MLP (thousands–tens of thousands of weights) | yes | `CompactifiedOmegaOMBU` / JetMLP |
| Exp-adjacent last layer (dynamic range) | yes | `exp_core=True` (paper); `deepmind_signed_hat_config` turns it off |
| Gradient-normalized residual | follow-up | `use_grad_norm=True` |
| Two-stage MSNN, \(\Phi_0+\varepsilon\Phi_1\) | yes (5 orders on CCF) | `multistage` + `stage2_even_hat` (even \(q\)); `deepmind_multistage_config` |
| Full-matrix / exact Gauss–Newton | kfac-jax rank-1 EMA + Martens–Grosse LR | **exact JVP** Martens–Grosse (`solver="qr"` / `"dense"`) plus cubic GN and linearized L∞. Measured CCF L∞ earn winner: epigraph L∞, not L2 GN |
| Adaptive collocation | yes | `adaptive_power` / `resample_every` |

Helpers: `deepmind_paper_architecture_config` (paper stack + `wholeline_hp`),
`deepmind_signed_hat_config` (same, signed hat),
`deepmind_multistage_config` (small Fourier stage-2 + Martens–Grosse).
PirateNet is a *different* PINN backbone (jaxpi), not the singularity paper.
The current official-path champ is a signed-hat pad stack plus a
zero-readout PirateNet correction plus even Fourier stage-2 plus
identity-init even-`q` MSNN plus a second new-seed MSNN at
\(7.938\times 10^{-3}\); stretch is
unearned. Paper L2 / gradient-normalized residual / exp-adjacent
multiplicative correction were rerun on that MSNN family and all
raised 1601-pt L∞ (and pushed `HΩ(0)` away from `+1.303`). Additive
epigraph L∞ earned \(4.8\times 10^{-5}\). Keep L∞ for this stretch
metric until a paper loss actually beats it. `martens_grosse` /
`wang_linearized_gn` / grad-norm stay the paper trainers for a
future \(10^{-8}\) stage-1; Adam stays the smoke heuristic. The
missing *use* of the paper stack is still a stage-1 that already
sits near \(10^{-8}\), which the official softplus net never reached
(ghost at raw \(\Omega\sim 10^{-6}\)). The follow-up paper states
MSNN is ineffective above that basin.

## Where reproduction / improvement / proof-readiness stand

- **Reproduction:** the *operator* and *self-similar algebra* are reproduced
  exactly (bit-parity across backends, zero substitution error). Line-domain
  discovery uses Hardy exact-Hilbert (`ccf_line` / `ccf_vorticity`) plus torch
  compactified neural vorticity (`ccf_vorticity_neural`) with **Hardy-aligned
  train Hilbert** (default) and CubicGaussNewton / Martens–Grosse earn path
  (Adam forbidden). Absolute Rung-1 gates are reported via
  `benchmarks/ccf_hardy_rung_acceptance.py` / `docs/benchmarks/` **only when
  earned**. Torch stage-2 `optimizer="gauss_newton"` is the labeled
  corr-matching proxy (`gauss_newton_corr_proxy`). The paper eq. 19
  residual-vector path is JAX `optimizer="martens_grosse"` /
  torch `optimizer="wang_linearized_gn"` (torch-graph residual required).
  Stretch \(10^{-13}\) never forges Rung-1. Measured dense Wang
  floors under nontrivial gauge remain \(O(10^{-2})\)–\(O(10^{-1})\).
  Periodic truncated-line FFT and finite-interval PV are **numerical
  diagnostics** and err at \(O(10^{-1})\) vs exact \(H[Q]=-P\); they are
  not the reproduce train operator. `train_hilbert="pv_mapped_tail"` is
  the older single-panel GL + raw-`u` tail (planted core `~1e-3` at 96
  nodes). Reproduce / paper free-Ω Hilbert is
  `train_hilbert="wholeline_hp"`
  (`omnibias.pinn.{jax,torch}.hilbert_line`): split-core GL plus a
  power-mapped algebraic tail. Planted `H[Q]=-P` can sit near `1e-14`;
  that is a numerical whole-line quadrature, **not** a closed-form Hardy
  transform and **not** a certificate. Stretch `10^{-13}` is still
  unearned on a trained Wang net. Score a
  free `Ω` with `free_omega_vorticity_residual` and an analytic `Ω_y`
  (`integrate_velocity_from_hilbert` builds `U`, not `OperatorBlock`
  `integral`). `omnibias.pinn.jax.discovery.ccf_hat_homotopy` is a
  signed-hat prototype (nodal init + frozen-velocity Picard + coupling
  homotopy) with hard gauge inside the Jacobian. A clustered-origin run
  measured official-path `max|r| = 7.840e-3` (11-node cubic
  Hermite plus frozen dual even Chebyshev-arctan, Gaussian /
  sech / compact-C¹ / rat4 / Laplace pads, then linearized L∞
  Newton on `s=0.8`, `s=0.15`, `s=3.5`, `s=0.08`, `s=0.30`,
  `s=1.8`, `s=2.8`, tanh-chart Chebyshev `s=1.0`, Boyd-map
  Chebyshev `s=0.9`, tanh-sinh Chebyshev `s=1.0`, erf-chart
  Chebyshev `s=1.1`, p=4 algebraic Chebyshev `s=1.0`,
  asinh-arctan Chebyshev `s=0.85`, softsign Chebyshev `s=1.2`,
  even Legendre on a p=3 algebraic chart, even Chebyshev-U on
  a p=2 algebraic chart, even Gegenbauer \(C^{(3/2)}\) on
  a log1p-arctan chart, even Jacobi \(P^{(2,2)}\) on a
  p=6 algebraic chart, mapped Laguerre on
  \(\xi=y^2/(s^2+y^2)\), p=5 algebraic Chebyshev,
  half-stereo Chebyshev, even Gegenbauer \(C^{(5/2)}\)
  on a p=1.5 algebraic chart, associated Laguerre
  \(L_k^{(1)}\) on \(\xi=1-e^{-y^2/s^2}\), p=7 algebraic
  Chebyshev + compact C¹² at `|y|=1.60` + multiplicative
  Tukey \(\alpha=0.25\) at `|y|=1.60`, circular-chart
  Chebyshev \(x=2sy/(s^2+y^2)\) + Planck taper at
  `|y|=0.85`, and erf-sinh Chebyshev
  \(x=\mathrm{erf}(\sinh(y/s))\) + flat-top at `|y|=0.10`,
  then a signed PirateNet residual correction
  (`omnibias.pinn.jax.discovery.pirate_hat`), then even
  Fourier stage-2 on compactified `q`, then identity-init
  even-`q` MSNN (`stage2_even_hat`), then a second
  identity-init even-`q` MSNN (new seed), then even
  Chebyshev \(T_k(2q-1)\) on compactified `q`, then even
  Legendre \(P_k(2q-1)\) on compactified `q`, then even
  Padé `[4/4]` on \(x=2q-1\), then even Fourier
  \(k=6..12\) on compactified `q`, then an identity-init
  tanh MLP readout on even `(q, ξ, bump)`, then even
  compact Gaussians at the origin and `|y|≈1.50`
  (identity-init SiLU readout, tanh-hidden `ΔW`,
  multiplicative Fourier-on-`ξ`, GELU-on-`ρ`,
  arctan-chart MSNN, Mish-on-log1p, even sinc, and
  even \(J_0\), a residual-shaped even lift, and
  PirateNet last-layer `ΔWout`, and even
  Hermite–Gauss, stage-3 MSNN `ΔB`, PirateNet
  skip-gate `Δα`, asinh-chart tanh, even
  Dawson \(y\,\mathrm{dawsn}(y/s)\), and PirateNet
  embedding `Δbe` and even
  \(\tanh(\sinh(y/s))^2\) (Hilbert-quadratic ray
  `s=0` on both) did not
  promote; sinc realized `~7.0e-7`),
  scored on
  1601-pt L∞; `HΩ(0)≈+1.007`; tanh-even parent was
  `7.847e-3`; Fourier-\(k=6..12\) parent was
  `7.876e-3`; Padé-on-`q` parent was
  `7.884e-3`; Legendre-on-`q` parent was
  `7.885e-3`; Chebyshev-on-`q` parent was
  `7.889e-3`; stage-3 MSNN parent was
  `7.938e-3`; first-MSNN parent was
  `7.943e-3`; even-Fourier parent was
  `7.991e-3`; PirateNet parent was
  `8.063e-3`; erf-sinh parent was
  `8.068e-3`; circular-Planck parent was
  `8.227e-3`; alg7 parent was
  `8.232e-3`; explag1 parent was
  `8.234e-3`; alg15-Gegenbauer parent was
  `8.278e-3`; half-stereo parent was
  `8.279e-3`; alg5 parent was
  `8.294e-3`; Laguerre parent was
  `8.297e-3`; alg6-Jacobi parent was
  `8.299e-3`; log1p-Gegenbauer parent was
  `8.309e-3`; alg2-U parent was `8.356e-3`;
  alg3-Legendre parent was `8.365e-3`; softsign parent was `8.37e-3`;
  asinh was `8.38e-3`; alg4 was `8.39e-3`; erf was `8.43e-3`;
  tanh-sinh was `8.44e-3`; Boyd was `8.46e-3`; tanh was
  `8.48e-3`; `s=2.8` was `8.50e-3`; `s=1.8` was `8.53e-3`;
  `s=0.30` was `8.56e-3`; L∞ parent was `1.008e-2`; Laplace was
  `1.24e-2`; rat4 was `1.25e-2`; C¹ was `1.28e-2`; peak-weighted
  `p=12` was `1.80e-2`; unweighted Hermite was `1.90e-2`; parent
  dual-scale plus Gaussian pads was `4.44e-2`);
  it does not move stretch /
  Rung-1. Neither mode is a certificate, and neither moves
  the stretch / Rung-1 thresholds. Dictionary enrichment on the same
  `{P,Q}` family was
  tried; 03-10 jet–Padé locates a singularity and is not a residual.
  **Rung-1 (\(10^{-11}\)) and Rung-2 remain unearned** until dense residual +
  CAP close; absolute thresholds were not moved.
- **Conjugate-tower dictionary (orders \(N>0\)):**
  `hardy_conjugate_dictionary` / `spatial_odd_omega_atoms` enlarges the odd
  Hardy-Ω span by \(Q^{(\mathrm{even})}\) and \(P^{(\mathrm{odd})}\), with
  closed-form \(U\) from \(U'=H\Omega\), \(U(0)=0\). Measure with
  `python benchmarks/reproduce_deepmind_ccf.py --dictionary conjugate --max-order N`
  and `python benchmarks/ccf_conjugate_sweep.py --write-docs`. Compare arms on
  raw `dense_max_abs` (not the anti-ghost gate floor). The smoke table in
  [`ccf_conjugate_sweep_smoke.json`](../benchmarks/ccf_conjugate_sweep_smoke.json)
  records \(N=0\) vs \(N=1\), Gram condition, empirical \(p\), and a whole-line
  CAP attempt. A drop of even \(10\times\) still leaves
  `orders_to_stretch` large; that is a measured floor, **not** progress toward
  stretch. Rung-1 / Rung-2 stay unearned unless the numbers say otherwise.
- **Improvement:** autonomy tick `benchmarks/deepmind_campaign_tick.py`,
  `CCFHardyAdapter` (Martens–Grosse), IPM/Boussinesq adapters, Phase 5
  beyond-DeepMind helpers (`phase5_beyond`, gated on Rung-2).
- **Proof readiness:** Hardy whole-line CAP
  (`certified_ccf_hardy_wholeline_blowup_attempt`, `form="vorticity"`).
  `whole_line_certified` flips only on genuine residual + \(\ell^1_\nu\)
  closure. Clay NS remains external (`navier_stokes_proof_claim=False`).

## See also

- API: [`omnibias-pinn`](../api/pinn.md), [`omnibias-symbolic`](../api/symbolic.md)
- [PINN Navier–Stokes (2D / 3D)](pinn-navier-stokes.md) for the field-state ops
- Paper: [Discovery of Unstable Singularities](https://arxiv.org/abs/2509.14185)
