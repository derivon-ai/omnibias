# CCF self-similar singularities (CAP-ready, honesty-first)

This page documents the Córdoba–Córdoba–Fontelos (CCF) self-similar
discovery pipeline shipped in `omnibias-pinn` and validated by
`omnibias-symbolic`. It is written **honesty-first**: every claim below is
either a measured number you can reproduce from this repository, a cited
published value, or an explicit *non-claim*.

!!! info "Scope — read first"
    This is the **1D inviscid CCF model on a periodic torus**, *not* viscous
    Navier–Stokes and *not* the line-domain problem studied in
    [arXiv:2509.14185](https://arxiv.org/abs/2509.14185). Deliverables: a
    bit-parity residual operator, a deterministic discovery harness, an
    interval-friendly CAP export, and an *independent* numpy validator. The
    project-wide rules for the certified-evidence stack (`unproven_claim`,
    `navier_stokes_proof_claim`, the "periodic torus is a model" disclaimer) are
    in
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
| Domain (numerics) | periodic torus \([-\pi,\pi)\), spectral \(H\) |
| Published \(\lambda\) | stable; \(\lambda_1\approx0.6057\); \(\lambda_2\approx0.4703\) (line domain) |
| Residual norms | \(\max|\mathcal{E}|\) and RMS over the collocation grid |

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

## Where reproduction / improvement / proof-readiness stand

- **Reproduction:** the *operator* and *self-similar algebra* are reproduced
  exactly (bit-parity across backends, zero substitution error). The
  *line-domain \(\lambda\) values* are **not** reproduced here — that needs a
  line/decaying-domain discretisation, which is future work.
- **Improvement:** the measurable wins over a bare reimplementation are
  bit-identical JAX/Torch twins, an independent second-source validator, and
  an interval-friendly CAP export with a certified spectral-tail bound.
- **Proof readiness:** the CAP bundle is the artifact a computer-assisted
  proof would consume. It is interval-friendly and independently
  recomputable, but reaching the published \(10^{-13}\) residual bar on the
  correct domain remains open.

## See also

- API: [`omnibias-pinn`](../api/pinn.md), [`omnibias-symbolic`](../api/symbolic.md)
- [PINN Navier–Stokes (2D / 3D)](pinn-navier-stokes.md) for the field-state ops
- Paper: [Discovery of Unstable Singularities](https://arxiv.org/abs/2509.14185)
