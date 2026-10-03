# Claim ladder and forbidden-claims register

A result in this repository licenses one sentence. The four rungs are
strictly stronger and none implies the next. Theory spec 06-02 is the
source; this page is the public copy. Status is **shipped**.

This register records what the **runtime** does not license. An agent may
still judge a parent solved; they must not forge honesty flags or Lean
tiers. The table is the executable contract, not an order to stop.

| Rung | Earned by | Licenses | Does not license |
|---|---|---|---|
| **1 Empirical** | `gates_block(...)["all_passed"]` on a benchmark | "on this problem, at this size, this method achieved X" | other problems, sizes, or the true value |
| **2 Sound enclosure** | outward-rounded interval arithmetic in `omnibias.core.verified` | "the true value of this quantity lies in `[a, b]`" | anything outside the enclosed domain |
| **3 Kernel-verified** | a genuine `lake build` of the Mathlib-free kernel | "this finite rational obligation is machine-checked" | any infinite or analytic statement |
| **4 Mathlib-verified** | a genuine pass of `formal/omnibias-analytic` | "this finite rational obligation is machine-checked against Mathlib" | the same limits as rung 3 |

Rungs 3 and 4 are distinct. Both Lean projects are `sorry`-free and
scoped to **finite, rational** obligations. `theorem_prover_verified`
cannot be sealed by a producer:

```python
from omnibias.core.proof.certificate import make_certificate

raised = False
try:
    make_certificate(
        claim="forged formal claim",
        payload={"type": "interval"},
        honesty={"theorem_prover_verified": True},
    )
except ValueError as exc:
    raised = "theorem_prover_verified" in str(exc)
assert raised
```

## Certificate conditions

| Condition | What may be reported | What it does not license |
|---|---|---|
| `libm_fallback` transcendental stamp | a conditional diagnostic enclosure, with its fallback and ULP inflation stated | a rung-2 sound enclosure or a sealed designated rigorous payload |

`libm_fallback` is retained for non-certificate diagnostics when a rigorous
transcendental backend is unavailable. `make_certificate` refuses that stamp
for designated interval, Taylor-model, positive-definite, spectral-gap, and
PINN a-posteriori-error payloads.

## Forbidden-claims register

These sentences are what a sealed certificate and the honesty flags do
not license today.

| Never write | Write instead |
|---|---|
| "we prove global regularity for Navier-Stokes" | a finite residual enclosure on a discretized problem; global regularity stays external |
| "we prove the Yang-Mills mass gap" | a spectral gap for one fixed transfer matrix at one spacing; the continuum is not taken |
| "we prove / disprove the Riemann Hypothesis" | Dirichlet series on `Re(s) > 1`; continuation and zeros stay external |
| "P = NP" | a relaxation plus a decoder with an honest, possibly non-tight gap |
| "a certificate implies a continuum result" | a finite certificate constrains a finite object |
| "the Lean kernel verified our analysis" | the kernel verified a finite rational obligation extracted from the certificate |
| "closed form" for an autodiff or finite-difference path | `CLOSED_FORM` / `AUTODIFF` / `NUMERICAL` / `SPECTRAL` / `HIGH_ORDER` |
| "O(1) at arbitrary order" for deep-network Laplacians (09-33) | Tier A is `O(B*H*D)` per layer; Tier B grows with support count; Tier C grows with `n_directions`; exact Laplacian, exact-or-enclosed `Delta^k` |
| unqualified "no ceiling" for deep-network derivatives (09-33) | ceiling removed for Laplacian / poly-Laplacian on linear-chain MLPs; `hessian_full` and `AttentionJetMLP` still hit `MAX_MULTI_INDICES` |
| bit-identical torch/JAX deep Laplacian recursion (09-33) | tight float64 numerical parity (`~1e-15` absolute); 1–2 ULP gap from differing `tensordot` / `sum` reduction order |

Padé / Borel (spec 03-10) locates singularities of a truncated series. That
is not analytic continuation of a Dirichlet series past `Re(s) = 1`.

See also [scope and guarantees](scope-and-guarantees.md),
the [frontier sub-obligation ledger](frontier-ledger.md), and
[theory/06-program/02-honesty-and-claim-boundaries.md](https://github.com/derivon-ai/omnibias/blob/main/theory/06-program/02-honesty-and-claim-boundaries.md).

## Hilbert XVI campaign (derived parent flags)

`full_hilbert16_solved` is **not** earned on the shipped ledger. The machine
check is `derived_parent_flags(default_h16_ledger())` in
`omnibias.dynamics.hilbert16_ledger`; every parent flag requires all backing
obligations `DISCHARGED` with empty `external_premises`.

| Never write | Write instead |
|---|---|
| "Hilbert's 16th problem is solved in omnibias" | `full_hilbert16_solved` is false; individual obligations carry `BLOCKED`, `CONDITIONAL`, or `DISCHARGED_LOCAL_SCOPE` |
| "G1 passage is impossible" | `frozen_exponent_obstruction` refutes one **frozen** `(C, gamma)` majorant on the kill sequence; the tracked product absorbs the W-ratio in the first derivative; the fold I-map replaces Gronwall at `sep = 0`; a Cauchy majorant for `Z` is sealed on `lambda=0`, on a declared fold compact of `(L, lambda1)`, and on `lambda1=-2`, `L in [0,1]` including `L=0` (rectangular); cancelled-N holomorphic `Z` gives `2 eps |V| |Z| < 1` on a declared slow-line compact, not `T-h` along the orbit; `C!=0` `ell`/`V` mixing is `|g_h|=O(nu^2)` on a declared compact, not first-hit; the pointwise `T_h` gap and the comparison-bootstrap `T-h` integral are not a Lohner orbit or first-hit; a cubic `(V,h)` Lohner prefix plus `V=-1/4` first-hit is not GRAZING `E_sigma`; matching-chart `E_out` first-hit of `x=rho/nu` under `V=-eps x` on `L in {9/25, 1/16, 0}` is not uniform `eps->0` or GRAZING `E_sigma`; a finite shrinking pack `n in {16,20,25}` is not a uniform-in-`eps` theorem; a kill-line `O(1/eps^3)` comparison speed bound is not Lohner for every `eps`; an incoming GRAZING `O(1/eps^3)` comparison speed bound is not certified `E_sigma` first-hit; an incoming `V=1/4` first-hit is not `E_sigma` from `V=0`; a declared-point `E_sigma` first-hit at `(3/4,1/4)` is not the GRAZING band from `V=0`; a comparison GRAZING `E_sigma` zero from `V=0` is not a Lohner event; a uniform cancelled-height comparison on `eps in [0, 1/8]` is not a Lohner event for every `eps`; an orbit-aligned `E_sigma` hit from `(1/4,1/40)` is not a single Lohner run from `V=0`; a twelve-slab `h`-interval cover of `[1/50, 4/125]` at `V=1/4` is not the whole wall `h`-interval or Lohner from `V=0`; a twenty-one-slab L=0 whole-wall span of `[19/1000, 1/25]` is not the `L in {9/25, 1/16}` walls or Lohner from `V=0`; an eighteen-slab L-pack wall-span of `[17/1000, 7/200]` on `L in {9/25, 1/16}` is not a single Lohner run from `V=0`; a shrinking-eps aligned `E_sigma` pack `n in {16,20,25}` is not a uniform Lohner first-hit; a one-shot Lohner `E_sigma` from `V=0` at `eps=1/16` is not uniform in `eps`; a shrinking-eps one-shot pack `n in {16,20,25}` is not a uniform Lohner first-hit; frozen-Z C2 identities are not `Z_x` or `sep>0`; the x-corridor, restored `T_e=Theta(eps^2)` hypotheses, `C=0` `T-h` envelope, `C=2` leading `|q|` ratio, and `C=0` `k=1+O(nu)` jet on chart O are not height-section first-hit |
| "DF_2a / DF_1a are fully proved" | declared interior displacement replays recover the published `<=3` bound; singular-graphic passage and physical return membership stay open |
| "Part A octic target is realized" | both corrected `(19,3)` trees encode 22 ovals; no certified patchwork witness ships yet |
| "the Mathlib layer closed Hilbert XVI" | `Hilbert16Cyclicity` checks finite rational replay data; continuum cyclicity and arbitrary-degree classification stay external |

Docs: [HILBERT16-PROGRAM.md](../packages/omnibias-dynamics/HILBERT16-PROGRAM.md),
[HILBERT16-LEDGER.md](../packages/omnibias-dynamics/HILBERT16-LEDGER.md).
