# 07-04 Yang-Mills adjacent: holonomy bands and finite transfer gaps

## 1. Thesis and status

omnibias already certifies a spectral gap for a fixed lattice transfer matrix;
the Wilson-line holonomy band supplies a **new class of trial states** for the
Lehmann-Maehly variational bounds, which is the one lever that makes a certified
lower bound tighter without enlarging the matrix. The continuum limit is not
taken, here or anywhere.

- **Status**: designed
- **Depends on**: 01-01, 02-14, 07-01, 07-05
- **Blocks**: none

## 2. Where it lands

`omnibias.geometry.gauge` — where the transfer machinery already lives and where
the folded-in `gauge` package was deliberately placed. Re-creating
`omnibias-gauge` would repeat a mistake `AGENTS.md` records.

## 3. Prior art in omnibias

Extensive, and it already contains the honesty text this spec must respect.

- `omnibias.geometry.gauge.transfer` — `TransferMatrix`,
  `certified_transfer_matrix_gap`, `TransferGapResult`, `GapCandidate`,
  `certified_multistep_gap_refinement`, `certified_effective_mass_curve`,
  `heat_kernel_gap_scaling_report` with `ScalingReport.continuum_claim = False`
  as a dataclass default, `seal_transfer_gap_certificate`,
  `replay_transfer_matrix_gap`, `certified_gap_versus_monte_carlo`,
  `sample_transfer_path_ensemble`, and the SU(2) constructors
  `su2_wilson_transfer`, `su2_heat_kernel_transfer`,
  `su2_class_angle_transfer`. Two gap methods, `SYMMETRIC_METHOD` and
  `BIRKHOFF_METHOD`.
- `omnibias.geometry.gauge.proofmachine` — the proof driver for gauge
  certificates.
- `omnibias.core.verified.eig_operator` — `lehmann_maehly_lower_bounds`,
  `LehmannCertificate`, `certified_spectral_gap`, `SpectralGapCertificate`,
  `temple_lower_bound`, `interval_ldlt_inertia`, `ritz_upper_bound`,
  `generalized_eigenvalue_enclosure`, `count_eigenvalues_below`.
- `packages/omnibias-geometry/tests/gauge/test_transfer_gap.py` and
  `test_transfer_certificates.py`.

**Confirmed gap.** The trial vectors feeding the variational lower bounds are
generic. There is no gauge-covariant trial-state construction, and holonomy —
the natural gauge-invariant object — is not represented as an operator at all.

## 4. Mathematics

### Why trial states are the lever

Lehmann-Maehly-Goerisch bounds produce a **lower** bound on an eigenvalue from a
trial subspace `V`. The bound's quality is governed by how well `V` captures the
relevant eigenvectors: a trial space aligned with the true low-lying states
gives a sharp bound, a generic one gives a loose bound, and enlarging `V` with
badly chosen vectors mostly adds conditioning problems.

So there are exactly two ways to tighten a certified gap for a *fixed* physical
setup: enlarge the matrix (expensive, and changes the object being certified) or
**improve the trial space** (cheap, and certifies the same object better). This
spec does the second.

### The holonomy band as a trial-state generator

Spec 02-14 defines the holonomy band: the path-ordered transport across the slab
between two parallel hyperplanes,

```
W[C] = P exp( int_C A . dx )
```

with three regimes — abelian (closed form), transverse-constant non-abelian
(closed form), and general non-abelian (an explicitly finite Magnus truncation
with a stated remainder).

Gauge-invariant observables are traces of holonomies around closed loops. A
trial state built from `tr W[C]` for a family of loops `C` is gauge invariant by
construction, which means the trial space lies inside the physical sector rather
than spanning gauge copies. For a variational bound, wasting trial dimensions on
gauge directions is pure loss, so this is exactly the right structural
restriction.

The band's role is that a slab between hyperplanes is precisely the region a
transport crosses, so the loop family is parameterized by the same bias
coordinates the rest of the library uses, and the derivatives of `tr W[C]` with
respect to those coordinates come from the tower.

### What a tighter gap does and does not mean

A certified lower bound `g_lo > 0` for a fixed matrix `T` at spacing `a` in
dimension `d` says: **this matrix has a gap of at least `g_lo`**. Tightening
`g_lo` from `0.01` to `0.40` when the numerical gap is `0.4137` means the
certificate is now informative rather than vacuous.

It says nothing about `a -> 0`, nothing about `d -> infinity`, and nothing about
the continuum theory. `ScalingReport.continuum_claim` is `False` by dataclass
default precisely so that a scaling study — which is a legitimate and useful
thing to compute — cannot be mistaken for a continuum statement.

### The scaling study, and why it is not evidence

`heat_kernel_gap_scaling_report` computes gaps across spacings. It is tempting
to read a sequence of positive gaps at decreasing `a` as evidence for a
continuum gap. It is not, for a reason worth stating in one line: **a sequence
of positive numbers can converge to zero.** Without a *uniform* lower bound
proven over the family — which would be a substantial part of the parent problem
(see the test-3 analysis in spec 07-01) — the sequence constrains nothing about
its limit.

## 5. Worked example

**A vacuous certificate made informative.**

Take an SU(2) heat-kernel transfer matrix at spacing `a = 0.1`, truncated to
`d = 64`. Suppose:

- the floating-point eigensolver reports eigenvalues `1.0000` and `0.6613`, so a
  numerical gap of `0.3387` in the transfer eigenvalue, corresponding to an
  effective mass `-log(0.6613) / a = 4.135`;
- `certified_transfer_matrix_gap` with a generic trial space returns a certified
  lower bound of `0.02` on the gap.

The certificate is **sound and nearly useless**: it proves a gap exists but at
`6%` of its actual size, so any downstream statement conditioned on the gap
being large fails.

Now supply a holonomy trial space: `k = 16` gauge-invariant vectors built from
`tr W[C]` over a family of loops spanning the slab. Lehmann-Maehly with a
well-aligned `16`-dimensional trial space typically recovers a large fraction of
the true gap. Suppose the bound improves to `0.31`, that is `92%` of the
numerical value.

**The three sentences this licenses, in order:**

1. Empirical: "the numerically computed gap of this `64`-dimensional SU(2)
   heat-kernel truncation at `a = 0.1` is `0.3387`."
2. Sound: "the gap of this matrix is at least `0.31`."
3. Kernel-verified, once the finite rational obligation "the enclosed gap is
   strictly positive" passes `lake build`: "positivity of this matrix's gap is
   machine-checked."

**And the sentence it does not license**, in any form: anything about the
Yang-Mills mass gap. The distance is not `9%`; it is the entire continuum limit
plus the infinite-volume limit plus the construction of the quantum field
theory, none of which is approached by any of the three statements.

**A scaling table, read correctly.** Suppose the results across three spacings
are

| `a` | certified lower bound | numerical gap | effective mass `-log(lambda_1)/a` |
|---|---|---|---|
| `0.20` | `0.52` | `0.55` | `3.99` |
| `0.10` | `0.31` | `0.3387` | `4.14` |
| `0.05` | `0.17` | `0.19` | `4.21` |

The effective mass is roughly stable while the transfer gap shrinks — which is
what one expects when the gap is `1 - exp(-m a)` with `m` roughly constant, so
`gap ~ m a` for small `a`. Reading the shrinking gap column as "the gap is
vanishing" would be wrong, and reading the stable mass column as "the continuum
mass is `4.2`" would also be wrong, because three points with no uniform error
control over the family do not determine a limit. The correct reading is: three
independent certified statements about three different matrices.

## 6. Proposed API

```python
# omnibias/geometry/gauge/holonomy.py           -- from spec 02-14
def holonomy_band(connection, slab, *, regime: Literal["abelian",
    "transverse_constant", "magnus"], order: int = 4) -> HolonomyOperator: ...

# omnibias/geometry/gauge/transfer/trial.py     -- new
def holonomy_trial_space(
    transfer: TransferMatrix, loops: Sequence[Loop], *, dim: int,
) -> TrialSpace:
    """Gauge-invariant trial vectors from tr W[C]. Reports the Gram condition
    number, since a badly conditioned trial space loosens rather than tightens
    the certified bound."""

def certified_transfer_matrix_gap(          # existing, extended
    transfer, *, trial: TrialSpace | None = None, ...
) -> TransferGapResult: ...
```

The extension is a keyword argument to an existing function, so every existing
caller, certificate schema and replay path is unchanged.

## 7. Practical use cases

1. **Making existing certificates informative.** A bound at `6%` of the true
   value is technically sound and practically useless; the trial space is the
   cheapest fix.
2. **Certified effective-mass curves** with error bars that are enclosures
   rather than statistics, comparable against Monte Carlo through the existing
   `certified_gap_versus_monte_carlo`.
3. **Gauge-invariant feature construction** for any learned model on a lattice
   configuration, independent of the certification use.
4. **A cross-check on Monte Carlo** in a regime where its error bars are
   statistical and this one's are not.

## 8. Acceptance gates

Baselines: the current generic-trial-space `certified_transfer_matrix_gap`
output, and the Monte Carlo comparison already implemented.

- **G1 tightness.** On a suite of at least `20` transfer matrices spanning
  spacings and truncation dimensions, the holonomy trial space yields a
  certified lower bound at least `5x` larger than the generic trial space, and
  at least `80%` of the numerically computed gap, at matched cost.
- **G2 soundness.** The certified bound never exceeds the true gap, verified
  against high-precision eigenvalues on `1000` synthetic matrices. **A single
  violation is a bug.**
- **G3 gauge invariance.** Trial vectors are invariant under a random gauge
  transformation to `1e-14` relative. Asserted by test, because a trial space
  that silently leaks gauge directions would loosen bounds without any visible
  symptom.
- **G4 conditioning reported.** The trial-space Gram condition number is in
  every certificate, and a run whose condition number exceeds a stated threshold
  is flagged rather than silently trusted.
- **G5 honesty flags.** `continuum_claim` is `False` in every emitted
  certificate and `ScalingReport`; asserted against the existing schema
  validators.
- **G6 kernel obligation.** The positivity obligation extracted from a tightened
  gap passes `lake build`, earning `theorem_prover_verified` **from the kernel**,
  and a test asserts the flag cannot be set without the pass.

## 9. Benchmark plan

- `benchmarks/gauge_holonomy_gap.py`: tightness across the matrix suite,
  soundness over synthetic instances, gauge-invariance check, conditioning
  sweep, Monte Carlo cross-check.
- Smoke JSON in `docs/benchmarks/`; full under
  `$OMNIBIAS_SCRATCH/gauge_gap/`.

## 10. Honesty and scope

- **Every certificate is one fixed matrix at one spacing in finite dimension**,
  which is the existing module's own stated scope and is not widened here.
- `continuum_claim = False` stays a dataclass default; a certificate that set it
  `True` would be rejected by the existing validators, and this spec does not
  add a path that could.
- A scaling table across spacings is **not** evidence about the continuum limit.
  A sequence of positive gaps can converge to zero, and nothing here bounds the
  family uniformly.
- The Magnus truncation in the general non-abelian regime is a **finite
  truncation with a stated remainder**, not an exact holonomy. Certificates must
  carry the remainder, and it enters the enclosure width.
- The founding `delta -> 0` bias collapse supplies derivatives of the holonomy
  with respect to slab coordinates. No temperature collapse appears.
- Certificate tier: **sound enclosure**, escalating to `theorem_prover_verified`
  only on a genuine kernel pass of the finite positivity obligation.

## 11. Open questions and risks

- **Trial-space conditioning is the main technical risk.** Gauge-invariant loop
  traces over a large loop family are highly correlated; the Gram matrix will be
  ill-conditioned, and an ill-conditioned trial space can make the certified
  bound *worse* while looking richer. G4 exists for this and may cap the usable
  dimension well below what G1 assumes.
- **The `5x` in G1 is a guess** based on how variational bounds usually respond
  to aligned trial spaces. It should be treated as a target to be measured, and
  revised in the artifact if the mechanism works but the factor is smaller.
- **Magnus remainder may dominate.** In the general non-abelian regime the
  truncation remainder could exceed the tightening gained, making the abelian
  and transverse-constant regimes the only ones where this pays.
- **SU(2) only.** The existing constructors are SU(2); SU(3) is the physically
  interesting case and is not implemented. Claiming generality would be wrong.
- **Falsifier.** If G1 fails — the holonomy trial space does not tighten the
  bound materially — then gauge invariance of the trial space is not the binding
  constraint, and the effort should move to spec 07-05's general spectral
  machinery instead.

## 12. Implementation checklist

- [ ] `packages/omnibias-geometry/src/omnibias/geometry/gauge/holonomy.py`
      (spec 02-14's operator)
- [ ] `packages/omnibias-geometry/src/omnibias/geometry/gauge/transfer/trial.py`
- [ ] `trial=` keyword on `certified_transfer_matrix_gap`, existing callers
      unchanged
- [ ] Gauge-invariance test to `1e-14` under random gauge transformation
- [ ] Gram condition number in every certificate, with a flagging threshold
- [ ] `1000`-instance soundness test against high-precision eigenvalues
- [ ] Magnus remainder carried into the enclosure width, never dropped
- [ ] Existing schema validators run against the extended certificate
- [ ] Kernel obligation wired through `omnibias.core.proof.lean_check`, with a
      test that the flag cannot be forged
- [ ] `benchmarks/gauge_holonomy_gap.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Yang-Mills existence and mass gap (Clay Millennium Problem).**

The parent asks for the construction of a quantum Yang-Mills theory on
four-dimensional Minkowski space satisfying the Wightman axioms, together with a
proof that its Hamiltonian has a strictly positive mass gap. Everything this
spec produces is a certified lower bound on the spectral gap of **one fixed
finite-dimensional matrix at one lattice spacing in one gauge group
truncation**.

Three limits separate the two, none of which is taken here: the continuum limit
`a -> 0`, the infinite-volume limit, and the removal of the finite-dimensional
truncation. A certified positive gap at every spacing in a finite ladder is
fully consistent with a vanishing continuum gap, so such a ladder is not
evidence for the parent in any usable sense — and the existence part of the
parent, the construction of the theory itself, is not addressed at all.

`omnibias.geometry.gauge.transfer` already encodes this by fixing
`continuum_claim = False`, including as a dataclass default on `ScalingReport`.
This spec does not claim, imply, or provide evidence for the Yang-Mills mass
gap.
