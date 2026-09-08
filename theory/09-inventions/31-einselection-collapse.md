# 09-31 Einselection collapse

## 1. Thesis and status

A sixth named collapse in `omnibias.core.collapse`: a sound, finite
adjudication of whether a pure-dephasing density matrix has decohered
below a declared measurement sensitivity. The surviving object is an
**einselected distribution** over pointer-basis populations, not a
wave-function-collapse claim and not a single-outcome claim.

- **Status**: shipped (G1–G5 CI; export, not a new package; not a
  measurement-problem resolution)
- **Depends on**: 01-14, 09-30
- **Blocks**: none

### Operator card

- **Benefit.** A certificate answering "at sensitivity `eps`, can any
  interference experiment still distinguish `rho(t)` from the
  classical mixture `{|c_i|^2}`?" instead of an unfalsifiable
  "it collapsed."
- **How it works.** A pure-dephasing model's off-diagonal coherence
  is enclosed with `ComplexInterval` / `exp_iv`; the max off-diagonal
  magnitude is compared against a caller-declared budget `eps`.
- **Strength.** Exact enclosure of an analytically solved dephasing
  channel. Honest about what it does not touch.
- **When to use.** Certifying a decoherence-timescale claim on a
  named pure-dephasing model with a stated sensitivity.
- **When not.** As a measurement-problem resolution, a Born-rule
  derivation, a single-outcome claim, or a relaxation (`T1`) model.
- **Accuracy floor.** Pure dephasing only; no energy relaxation; the
  candidate pointer basis is supplied, not discovered by proof.

## 2. Where it lands

`omnibias.core.collapse.einselection`, beside `verdict`, `identity`,
`winding`, `pairing`, and `rank` in `omnibias-core`. No new package
and no new submodule directory: this is one more module in the
existing `omnibias.core.collapse` catalogue.

## 3. Prior art in omnibias

- `omnibias.core.collapse.schema` — `CollapseSpec` / registry gate;
  a name earns a slot only by minting a distinct `(parameter,
  surviving_object)` pair.
- `omnibias.core.collapse.verdict` — `{0}` proves, exclusion
  disproves, a fat zero is `BLOCKED`. Einselection collapse reuses
  this three-way shape on a coherence enclosure instead of a generic
  residual.
- `omnibias.core.collapse.pairing` — the precedent for an honest
  "not a strong solution" disclaimer sitting beside a `PROVED`
  outcome; einselection collapse's "not a single-outcome claim"
  plays the same role.
- `omnibias.core.verified.complex_interval.ComplexInterval` — sound
  rectangular complex enclosures; `.modulus()` for `|rho_ij|`.
- `omnibias.core.verified.transcend.exp_iv` — sound enclosure of
  `exp(x)`, used for the dephasing exponential; never `math.exp`.
- `omnibias.core.proof.engine` — the `verdict` / `identity` /
  `winding` / `pairing` / `rank` kinds route through
  `omnibias.core.proof.engine.prove`; this spec adds `einselection`
  to that same router.

**Confirmed gap.** No decoherence, einselection, density-matrix,
Lindblad, or pointer-state vocabulary exists anywhere in the repo
today (verified by full-text search). `omnibias.qpinn` ships only
unitary evolution (TDSE, NLS, Dirac); it has no open-system residual.

**Asymmetry worth recording.** `verdict`, `identity`, `winding`,
`pairing`, and `rank` collapse shipped as code plus docs with no
`theory/09-inventions` spec and no invention-ledger row — they
predate and bypass the ledger entirely, cataloged instead directly in
`docs/api/collapse.md` and the AGENTS.md Wave-1 primitives list. This
spec exists as an explicit design record because it introduces a new
class of honesty flag (a fixed set of permanently-false physics
claims) and a new verified-numerics composition (`ComplexInterval`
plus `exp_iv` plus a max-over-intervals reduction) that is worth
writing down before landing. Landing it under Group 09's `export`
taxonomy is the closest fit — a compiled `Verdict` beside a claim,
the same shape as 09-30 — not a retroactive claim that the five
prior collapses need specs too.

## 4. Mathematics

Fix a candidate pointer basis of dimension `d` and initial amplitudes
`c_i` with `sum_i |c_i|^2 = 1`. The exact solution of the pure-dephasing
Lindblad equation (no energy relaxation, `T1 = inf`) in this basis is

```
rho_ij(t) = c_i * conj(c_j) * exp(-Gamma_ij * t)   (i != j)
rho_ii(t) = |c_i|^2                                 (i == j)
```

with a symmetric nonnegative rate matrix `Gamma_ij = Gamma_ji >= 0`,
`Gamma_ii = 0`. Define the coherence

```
C(t) = max_{i != j} |rho_ij(t)|
```

Given a caller-declared sensitivity `eps > 0`, the adjudicated claim
is:

> "No interference measurement with resolution coarser than `eps`
> can distinguish `rho(t)` from the classical mixture `{|c_i|^2}`."

On a *sound* enclosure `C_hat` of `C(t)` (built from `ComplexInterval`
arithmetic and `exp_iv`, never a float `exp`):

- `C_hat.hi < eps` -> `PROVED` (`collapsed`); surviving object is the
  einselected distribution `{|c_i|^2}`
- `C_hat.lo > eps` -> `DISPROVED` (`excluded`); coherence is provably
  still detectable at this sensitivity
- otherwise -> `BLOCKED` (`inconclusive`); not falsity

This is **temperature collapse's** shape (a threshold decision) laid
over einselection's own moving parameter (`Gamma_ij * t -> inf`,
i.e. `decoherence_rate -> inf`), which is why it earns a *new*
registry slot rather than reusing `temperature`: the surviving object
is a probability distribution over pointer populations, not a 0/1
indicator, and the parameter is a physical decoherence rate, not an
annealed sharpness `beta`. It is not founding bias collapse
(`delta -> 0`) and it is not Enclosure Collapse of a scalar: the
surviving object on `PROVED` is a `d`-vector distribution, and
`DISPROVED` here is a positive physical statement (interference
remains detectable), which Enclosure Collapse's point-plus-proof
shape has no analogue for.

**What this does not claim.** The global state `|Psi(t)> = sum_i c_i
|s_i> |E_i(t)>` stays pure, unitary, and entangled throughout. `rho(t)`
is an *improper* mixture: it has the numerical form of "system is in
state `i` with probability `|c_i|^2`" but the ignorance interpretation
is not licensed, because the correlations still exist in the
(untraced) environment. Einselection explains the preferred basis and
the disappearance of interference. It does not explain, and this
module does not claim to explain, why any single outcome is realized.

## 5. Worked example

`d = 2`, `c = (1/sqrt(2), 1/sqrt(2))`, `Gamma_01 = 1`, `eps = 1e-6`.

At `t = 5`: `|rho_01(5)| = 0.5 * exp(-5) ≈ 3.37e-3 > eps` ->
`DISPROVED`; coherence is still certainly detectable.

At `t = 20`: `|rho_01(20)| = 0.5 * exp(-20) ≈ 1.03e-9 < eps` ->
`PROVED`; surviving einselected distribution is `(0.5, 0.5)`.

## 6. Proposed API

```python
# omnibias.core.collapse.einselection — implemented
EINSELECTION_SPEC = CollapseSpec(
    name="einselection",
    parameter="decoherence_rate",
    limit="inf",
    surviving_object="einselected_distribution",
    failure="residual_coherence_above_budget",
    home="omnibias.core.collapse.einselection",
    register="measure",
)

@dataclass(frozen=True)
class DephasingModel:
    amplitudes: tuple[ComplexInterval, ...]
    rates: tuple[tuple[Interval, ...], ...]

def reduced_density_matrix(
    model: DephasingModel, time: IntervalLike
) -> tuple[tuple[ComplexInterval, ...], ...]: ...
def coherence_enclosure(
    rho: Sequence[Sequence[ComplexInterval]],
) -> Interval: ...
def einselected_distribution(model: DephasingModel) -> tuple[Interval, ...]: ...
def einselection_collapse(
    model: DephasingModel, *, time: IntervalLike, coherence_budget: float
) -> ObligationVerdict: ...
def commutator_enclosure(
    a: Sequence[Sequence[ComplexInterval]],
    h: Sequence[Sequence[ComplexInterval]],
) -> tuple[tuple[ComplexInterval, ...], ...]: ...
def pointer_basis_verdict(
    a: Sequence[Sequence[ComplexInterval]],
    h: Sequence[Sequence[ComplexInterval]],
) -> ObligationVerdict: ...
def propose_pointer_basis(
    model: DephasingModel, *, time: IntervalLike
) -> tuple[int, ...]: ...
```

Pure Python in `omnibias-core`. No torch, no jax. Default dtype is not
applicable (no tensors); all arithmetic is exact-rational-seeded
outward-rounded `Interval` / `ComplexInterval`.

## 7. Practical use cases

1. **Timescale certificate.** "At sensitivity `1e-6`, this two-level
   dephasing model is indistinguishable from a classical mixture
   after `t = 20`" (worked example).
2. **Sensitivity sweep.** Certify the same model at several `eps` to
   report the smallest sensitivity that still proves.
3. **Pointer-basis sanity check.** Certify that a candidate system
   observable commutes with the declared interaction Hamiltonian
   before trusting it as a pointer basis.
4. **Not** a claim about which outcome is realized, a Born-rule
   derivation, or a T1/T2 relaxation model.

## 8. Acceptance gates

- **G1 registry.** `einselection` registers, is distinct from all
  eight active specs (`bias`, `temperature`, `enclosure`, `verdict`,
  `identity`, `winding`, `pairing`, `rank`), and a same-axis rebrand
  attempt is refused by `register_collapse`.
- **G2 soundness.** Over a dense deterministic grid of `(Gamma, t)`
  plus a random sample, every enclosed `rho_ij(t)` contains the float
  truth computed independently with `math.exp`.
- **G3 monotone decision.** With fixed `Gamma > 0` and `eps`,
  increasing `t` moves the verdict `DISPROVED -> BLOCKED -> PROVED`
  and never regresses; at `Gamma = 0` the verdict never proves for
  `eps` below the initial coherence.
- **G4 pointer basis.** A constructed commuting `(A, H)` pair proves
  `pointer_basis_verdict`; a non-commuting pair disproves it; the
  float `propose_pointer_basis` sieve's ranking never changes either
  verdict.
- **G5 honesty non-vacuity.** Every `einselection_collapse` and
  `pointer_basis_verdict` outcome carries
  `wave_function_collapse_claim`, `measurement_problem_resolved`,
  `single_outcome_claim`, and `born_rule_derived` all `False`; a
  static scan confirms the module source never sets any of the four
  to `True`.

## 9. Benchmark plan

No dedicated `benchmarks/*.py` artifact: this is a finite-obligation
certificate primitive, adjudicated per call like `verdict` /
`identity` / `winding` / `pairing` / `rank`, not a trained model with
a wall-clock benchmark. `packages/omnibias-core/tests/test_collapse_einselection.py`
is the G1–G5 record, run in default CI.

## 10. Honesty and scope

- Not a wave-function-collapse claim, not a resolution of the
  measurement problem, not a Born-rule derivation, and not a
  single-outcome claim. The global state stays pure and entangled;
  `rho(t)` is an improper mixture.
- Do not conflate this with founding bias collapse (`delta -> 0`),
  temperature collapse (`beta -> inf`), or Enclosure Collapse
  (`width -> 0` of a sound enclosure, yielding a point plus a proof).
- Pure dephasing only (`T1 = inf`); no relaxation channel.
- The pointer basis is a caller-supplied candidate; `pointer_basis_verdict`
  checks one candidate's commutator, it does not search for one, and
  `propose_pointer_basis` is a float heuristic that never gates the
  verdict.
- `theorem_prover_verified` and `mathlib_verified` stay false; this is
  a sound-enclosure-tier certificate, not a Lean obligation.
- Navier-Stokes, Yang-Mills mass gap, RH, and P vs NP stay external
  and unrelated.

## 11. Open questions and risks

- **Rebrand challenge.** A reviewer may argue this is Enclosure
  Collapse of the coherence scalar. Section 4's distinction (a
  `d`-vector surviving object; `DISPROVED` as a positive physical
  statement) is the defense, not a settled fact.
- **Budget silently defaulting.** `coherence_budget` must always be
  an explicit caller argument. A hidden default would turn the
  certificate into an unstated tolerance fudge.
- **Falsifier.** A `PROVED` verdict whose `C_hat.hi` does not
  actually bound the float-computed `|rho_ij(t)|` on the G2 grid, or
  any honesty key in G5 observed `True`.

## 12. Implementation checklist

- [x] `omnibias.core.collapse.einselection` (`EINSELECTION_SPEC`,
      `DephasingModel`, enclosures, `einselection_collapse`,
      `commutator_enclosure`, `pointer_basis_verdict`,
      `propose_pointer_basis`)
- [x] `packages/omnibias-core/tests/test_collapse_einselection.py`
      (G1-G5)
- [x] `omnibias/core/collapse/__init__.py` re-export + `__all__`
- [x] `omnibias.core.proof.engine` `einselection` kind
- [x] `docs/cookbook/einselection-collapse.md`
- [x] `docs/api/collapse.md`, `docs/theory.md`, `AGENTS.md`,
      `mkdocs.yml` nav
- [x] Index row in `theory/README.md`

---

## Repo invariants this spec must respect

Check these before submitting an implementation.

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`. Pure-Python math lives there and every backend imports it.
- **Bit-identical twins**: torch and jax implementations must agree exactly.
  Polynomial coefficients come from `omnibias.core.polynomials`; never fork them
  per backend.
- **Default dtype**: use the framework default
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a
  hardcoded `float32`.
- **Vendor-neutral language**: no scheduler commands, vendor names, internal
  hostnames, usernames, or absolute local paths in tracked files. Artifacts go
  to `$OMNIBIAS_SCRATCH`, defaulting to a repo-relative `artifacts/`.
  `packages/omnibias-core/tests/test_no_leakage.py` enforces this across the
  whole readable surface.
- **Terminology**: distinguish the founding bias collapse (`delta -> 0`, yields a
  derivative) from temperature collapse (`beta -> inf`, yields a 0/1 step). The
  older penalty phrasings are retired and guarded by `tests/test_terminology.py`.
  Every new `**/relaxation.py` must carry the cross-reference note and be listed
  in `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- **Executable docs**: if any part of this lands in `docs/` or a package README,
  every fenced Python block is executed by `tests/test_docs_snippets.py`. Verify
  calls against real signatures; opting out needs a directive with a stated
  reason.
- **Earned flags**: `theorem_prover_verified` is set only by a genuine
  `lake build` pass of the Mathlib-free kernel, and `mathlib_verified` only by a
  genuine pass of the Mathlib-backed project. Asserting either without a pass
  blocks the verdict.
- **Typing tier**: the strict CI gate covers `core`, `torch`, `jax` and
  `ferminet`. Newly authored modules should be written strict-clean regardless
  of tier; curated beta modules can be added to
  `scripts/mypy_strict_allowlist.txt` once
  `mypy --strict --follow-imports=silent <file>` is clean.
