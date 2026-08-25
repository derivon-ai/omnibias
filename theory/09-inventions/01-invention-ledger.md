# 09-01 The invention ledger

## 1. Thesis and status

The closed-form tower, the unused `integral` role, and the same algebra in
float / interval / Lean make a family of **architectures, learning rules, and
exports** worth trying that are not trainers-of-`theta` (Group 08) and not
already first-class specs in Groups 02–05. This file is the index: what is
already specified, what is a new Group 09 spec, what is rejected, and which
cluster to implement first.

- **Status**: gated
- **Depends on**: 01-01, 01-10, 06-01, 06-02, 08-01
- **Blocks**: 09-02, 09-03, 09-04, 09-05, 09-06, 09-07, 09-08, 09-09,
  09-10, 09-11, 09-12, 09-13, 09-14, 09-15, 09-16, 09-17, 09-18, 09-19,
  09-20, 09-21, 09-22, 09-23, 09-24, 09-25, 09-26, 09-27, 09-28

### Operator card

- **Benefit.** One page answers "is this idea new, already a spec, or not
  worth a file" without re-deriving the four knobs or the CCF floor.
- **How it works.** Three taxonomies (architecture / learning rule / export),
  an already-specified map, a reject list, and a first-bet ranking. Every
  09-02…09-28 names a parent row here.
- **Strength.** Stops a jet-token net from being written as Jet-KAN, an
  integral-kernel operator from being written as BEM-Net, and any invention
  from being aimed at `1e-13` CCF stretch.
- **When to use.** Before implementing any 09-* file, or when proposing a
  twenty-ninth invention.
- **When not.** This file does not train a network and does not ship a layer.
- **Accuracy floor.** None of its own. CCF stretch remains an operator floor
  (`~1e-1` Hilbert / dictionary), not an architecture floor.

## 2. Where it lands

A document. No module and no package. The same pattern as spec 08-01: the
ledger is the index; the other Group 09 files are the entries. Implementation
homes are declared per entry in section 2 of those files and summarized in
spec 06-03. **Zero new packages.**

## 3. Prior art in omnibias

Group 09 must not re-derive shipped primitives or existing specs.

- `docs/operator-surface.md` — six `OperatorBlock` roles, including the
  closed-form `integral` window `S(z+b_hi)-S(z+b_lo)` with `S'=sigma`. Most
  of Group 02 spent the derivative roles; the unused creative space is
  antiderivative-first.
- `omnibias.{torch,jax}.jet` — `compose_jet`, `layer_jet`, `mlp_jet`,
  `affine_jet`. Exact directional jets; Faà di Bruno is the chain rule.
- `omnibias.{torch,jax}.jet_mv` — `compose_jet_mv`, `layer_jet_mv`.
- `omnibias.core.polynomials` — shared `sigma^(n)` coefficients.
- `omnibias.core.verified` — `TaylorModel`, `TaylorModelMV`, `Interval`,
  `radii_polynomial_certificate`, `lohner_flow`.
- `omnibias.hopfield` — `attention`, `modern_hopfield_retrieve`,
  `logsumexp_hessian` (vector-valued).
- `omnibias.score.flow` — `cnf_dynamics`, `integrate_cnf` (continuous
  flow; exact field `div`).
- `omnibias.verify` — `lipschitz_bound`, `taylor_output_bounds`.
- `omnibias.geometry.atlas` — `AtlasSpec`, `blended_metric` (metrics, no
  cocycle).
- `omnibias.holonomic._core` — `DFinite`, `OrePolynomial`, `_fit_annihilator`.
- `omnibias.qcalculus` — `q_derivative`, `q_derivative_limit`.
- `omnibias.timescale` — `delta_derivative`, `delta_derivative_limit`,
  `TimeScale`.
- Group 02 — Scan-Net, Jet-KAN, VPINN, BEM-Net, LadderNet, holonomy band,
  linearizing transforms, arrangement GNN, …
- Group 03 — jet–Padé, pack refinement, neural quadrature, jet line search, …
- Group 08 — trainers that step `theta` given `L`. 08 still owns that
  surface; 09 owns architectures and non-`theta` learning rules.

**No gap.** This ledger is the index. `test_theory_invention_ledger.py`
checks G1–G5. The entries are already gated; this file records the
taxonomy, already-specified map, rejects, and first-bet order they must
keep.

## 4. Mathematics

### Three taxonomies

An **architecture** is a hypothesis class: what a hidden state *is* and
how a layer maps it. Specs 09-02…09-15 and 09-27…09-28 sit here. The founding **bias
collapse** (`delta -> 0`) supplies `sigma^(n)` when a pack coalesces; the
**finite-gap** knob supplies `band` / `integral` when it does not.

A **learning rule** (in this group) is a loss, a meta-loop, or an invert
that is not "step `theta` given a scalar `L`" — those remain Group 08.
Specs 09-16…09-23 sit here. 08-03 invert-and-match is *not* re-specified.

An **export or app** is a forward contract or a compiled object: a
certificate beside `y`, a jet world-model, an Ore annihilator. Specs
09-24…09-26 sit here.

Faà di Bruno **is** the chain rule. No 09 spec may claim to avoid
compositional differentiation. None of 09-02…09-28 is temperature
collapse (`beta -> inf`) unless it delegates to 04-02 / 05-02 / 09-07's
router honesty or to 03-05 / 09-28 named `soft_top_k` / Hopfield `beta`.

### Already specified (no duplicate file)

| Idea | Spec | Why not a new file |
|---|---|---|
| Previous-layer / invert-and-match / EM-invert | 08-03 | Local target + `sigma^{-1}` is already the invert variant |
| Depth-causal PDE residual | 08-05 | PINN residual marched in depth |
| Neural quadrature as the model | 03-06 | Certified cubature is the model |
| Jet–Padé pole diagnostic | 03-10 | Locates the nearest complex singularity |
| Pack birth / death | 03-13 | Residual-driven refinement |
| BEM / single-layer potential | 02-06 | Surface densities; off-surface exact |
| One-step Kantorovich accept | 08-04 | Unique-zero ball on *one* Newton step |
| Sharpness as *step size* | 08-06 | `lambda_max` sets cubic `sigma` / lr |
| Certified *step* filter | 08-09 | Accept/reject `theta'` |
| Arrangement LP / face walk | 03-02 | Learned facets into a certified argmin |
| Mechanism cells / causal slabs | 05-02 G5 | Failed vs S4D; sequence submodule retired; do not re-open as 09 |
| Slab-mass probabilities | 04-02 | Calibrated conformal slabs |
| Learnable morphology / SoftMaxPool / index-SE | 03-05 | Flat dilation + `soft_top_k`; not a seventh role |
| Semiring neighborhood combiner | shipped `cmbConv*` + 01-08 + 03-05 | Sum-product vs max-plus homotopy |
| Parametric DeepONet slab (no `μ` jets) | shipped `pinn.operator` | `ParametricOperatorSlab` / `pde_params` encoder |

09-14 is **not** 02-06: volumetric / DeepONet kernel versus surface BEM.
09-20 is **not** 08-04: a path of problems versus one accept/reject.
09-23 is **not** 08-06: `lambda_max` in the *loss* versus the schedule.
09-24 is **not** 08-09: a certificate on the *forward* versus on the step.
09-27 is **not** 09-16 / 09-20 / 09-22: mixed `x`–`μ` jets versus MAML / homotopy / Newton-on-`x`.
09-28 is **not** 09-02 / 05-01: scan-jet tokens + named energy versus `compose_jet` stream / interface inverse.

### Rejected (no spec)

1. **"Skip the chain rule and solve for a global min."** Same as 08-01
   reject 1. Exact `sigma^(n)` does not convexify a deep nest.
2. **Full `d h / d theta` as a "flow."** Same as 08-01 reject 2. Jet-token
   state is `N`-jets in `k` directions or `d / d x`, never `width x P`.
3. **Any invention aimed at CCF `1e-13`.** Stretch is an operator gate
   (Hilbert × dictionary, spec 07-03). Architectures do not clear it.
4. **Generic ImageNet transformer / MoE / Adam.** If it is implementable
   with a ReLU MLP and vanilla autodiff, it is not an omnibias invention.

```python
# documentation only — reject keys, not a module
REJECTED = (
    "skip_chain_rule_global_min",
    "full_parameter_jacobian_flow",
    "ccf_stretch_by_architecture",
    "generic_imagenet_reimplementation",
)
```

### First-bet ranking (implement later, not this pass)

The first-bet cluster is already gated as code. This ranking stays the
recorded order, not a claim that later entries are unearned:

1. **09-03 FTC-Net + 09-17 dual-FTC** — unused `integral` cell + FTC
   consistency residual.
2. **09-02 jet-token + 09-19 jet distillation** — jet-valued state and
   the loss that matches it.
3. **09-16 exact MAML** — inner exact Hessian; few-shot PINN adaptation.
4. **09-05 Taylor-model neuron + 09-24 proof-carrying forward** — three
   registers as the hidden state.
5. **09-06 coupling Jet-Flow** — exact `sum log sigma'` det.
6. **09-07 Pack-MoE** — four knobs as experts; slab-mass router.
7. **09-18 remainder training** (with existing 03-13 birth).

The remaining 09-04, 09-08…09-15, 09-20…09-23, 09-25…09-28 wait on a
first-bet gate passing or failing.

### CCF / stretch floor

`CCF_STRETCH_RESIDUAL_GATE = 1e-13` is an **operator** gate. The recorded
Hilbert × dictionary catch-22 floors near `1e-1`. No 09 architecture,
learning rule, or export clears that floor. Group 09 files must not
weaken the stretch gate and must not infer Navier–Stokes regularity,
Yang–Mills mass gap, RH, or P vs NP.

### Comparison (concept, not measured)

| Entry | Taxonomy | What is new | Typical floor |
|---|---|---|---|
| 09-02 jet-token | architecture | Hidden state is an `N`-jet | jet truncation `R_N`; not ImageNet |
| 09-03 FTC-Net | architecture | Cell is `integral` | FTC residual on 1-D; not CCF |
| 09-04 Frame-UNet | architecture | Order encoder + integral decoder | 01-06 frame honesty (`sigma'` not admissible) |
| 09-05 Taylor-model neuron | architecture | Unit is a `TaylorModel` | enclosure explosion past shallow depth |
| 09-06 Jet-Flow | architecture | Finite coupling, exact det | small generative models |
| 09-07 Pack-MoE | architecture | Slab-mass router | 04-02 calibration; LightGBM still a baseline |
| 09-08 Characteristic-Net | architecture | Transport along learned `v` | 1-D conservation; characteristics cross |
| 09-09 sheaf-atlas | architecture | Cocycle residual on jets | chart overlap quality |
| 09-10 Riccati flow | architecture | Depth is Riccati time | distinct from DEQ / CNF |
| 09-11 Collapse-Net | architecture | Train discrete, infer `delta -> 0` | stencil exactness (01-04) |
| 09-12 holonomic layer | architecture | Ore annihilator as the block | D-finite class only |
| 09-13 jet-Hopfield | architecture | Memories are germs | contact match, not vector Hopfield |
| 09-14 integral-kernel op | architecture | OMBU `integral` as DeepONet kernel | not surface BEM |
| 09-15 q-OMBU | architecture | Named `q -> 1` / `mu -> 0` | residual of the limit |
| 09-16 exact MAML | learning rule | Exact inner HVP / IFT | same operator floor as the task |
| 09-17 dual-FTC | learning rule | Derivative vs integral residual | FTC identity |
| 09-18 remainder train | learning rule | Loss is `R_N` | 03-10 diagnostic + 03-13 birth |
| 09-19 jet distillation | learning rule | Match teacher `N`-jet | jet order, not logits |
| 09-20 homotopy | learning rule | Path of problems + 08-04 | empty ball => halt |
| 09-21 exact score match | learning rule | Hyvärinen on tower score | CNF already has exact `div` |
| 09-22 inverse-design | learning rule | Newton-on-`x` | `sigma' -> 0` saturation |
| 09-23 sharpness loss | learning rule | `lambda_max` in `L` | not 08-06 schedule |
| 09-24 PCI forward | export | `(y, certificate)` | vacuous box on deep nets |
| 09-25 world-model jet | export | Next `N`-jet + Lohner | not global regularity |
| 09-26 annihilator export | export | Ore + finite Lean obligation | finite rational only |
| 09-27 parameter-space jets | architecture | Mixed `∂^{α,β} u / ∂x^α ∂μ^β` | `μ` must enter the tower; not a ParamPINN package |
| 09-28 sliced-jet encoder | architecture | Tokens are scan jets + named energy | not a ViT; not `R^D` |

## 5. Worked example

**Triaging a proposed twenty-ninth invention.**

*Proposal:* "A transformer whose tokens are activation vectors and whose
attention is `omnibias.hopfield.attention`."

- Taxonomy: architecture.
- Test against the already-specified map: hopfield attention is shipped;
  Group 02 has no jet-token residual stream. If tokens stay vectors, this
  is **not** 09-02 — it is a consumer of hopfield. **No new spec.**
- If tokens become `N`-jets mixed by `compose_jet`, it **is** 09-02.

*Proposal:* "Minimize the Hilbert residual of CCF with a Frame-UNet."

- CCF test: **fails** reject 3. Point at 07-03. 09-04 may still be
  specified for local PDEs; its honesty section forbids stretch inference.

*Proposal:* "Target-propagate by inverting `sigma` at each layer."

- Already 08-03 invert-and-match. **Pointer only.**

*Proposal:* "Learnable max-pool / learn kernel indices."

- Already 03-05 (flat dilation + `soft_top_k`). **Pointer only.**

*Proposal:* "ViT without patches on ImageNet."

- Reject 4. 09-28 is a sliced-jet encoder on a named small image, not that claim.

## 6. Proposed API

Does not exist as code. The ledger is a table and a ranking.

```python
# documentation only — not a module
LEDGER_ARCHITECTURES = (
    "09-02", "09-03", "09-04", "09-05", "09-06", "09-07",
    "09-08", "09-09", "09-10", "09-11", "09-12", "09-13",
    "09-14", "09-15", "09-27", "09-28",
)
LEDGER_LEARNING_RULES = (
    "09-16", "09-17", "09-18", "09-19", "09-20", "09-21",
    "09-22", "09-23",
)
LEDGER_EXPORTS = ("09-24", "09-25", "09-26")
ALREADY_SPECIFIED = (
    "08-03", "08-05", "03-06", "03-10", "03-13", "02-06",
    "08-04", "08-06", "08-09", "03-02", "05-02", "04-02",
    "03-05",
)
REJECTED = (
    "skip_chain_rule_global_min",
    "full_parameter_jacobian_flow",
    "ccf_stretch_by_architecture",
    "generic_imagenet_reimplementation",
)
FIRST_BET = (
    "09-03+09-17", "09-02+09-19", "09-16", "09-05+09-24",
    "09-06", "09-07", "09-18",
)
```

## 7. Practical use cases

1. **Choosing an implementation order.** Follow first-bet; do not start
   at 09-26.
2. **Writing a design note** without implying Clay, stretch, or ImageNet.
3. **Refusing a duplicate** of 08-03, 02-06, or 03-06.
4. **CCF campaign hygiene.** Any PR that claims a Group 09 object moved
   stretch must fail this ledger's CCF paragraph.

## 8. Acceptance gates

- **G1 completeness.** Every `theory/09-inventions/*.md` except this file
  appears in the section-4 comparison table with taxonomy and floor.
- **G2 already-specified named.** The pointer table in section 4 lists
  every plan-filter row and has no implementation file of its own.
- **G3 rejects named.** The four rejected proposals are written in
  section 4 and do not have implementation files.
- **G4 no stretch inference.** No Group 09 spec sets or weakens
  `CCF_STRETCH_RESIDUAL_GATE`.
- **G5 no new package.** Every implementable spec's home fails the
  "earn independent existence" test (submodule of an existing package).

These are document gates. Numerical gates live in 09-02…09-28.

## 9. Benchmark plan

None of this file's own. Each entry names `$OMNIBIAS_SCRATCH/inventions/<name>/`
and a committed smoke JSON under `docs/benchmarks/` when implemented.
No CI job for the ledger.

## 10. Honesty and scope

- The tower is the founding **bias collapse** (`delta -> 0`). Finite-gap
  `integral` / `band` is the window knob, not a third collapse.
  Temperature collapse appears only where a router or partition hardens
  (`beta -> inf`) and is named in that spec.
- Group 08 still owns trainers that step `theta` given `L`. This ledger
  owns architectures, invert-on-`x`, meta-loops, and exports.
- Certificate tiers used by entries: empirical (most), sound enclosure
  (09-05, 09-20, 09-24, 09-25). `theorem_prover_verified` and
  `mathlib_verified` appear only as *future earned* flags on 09-24 /
  09-26 and are never asserted by the spec text.
- Navier–Stokes, Yang–Mills mass gap, RH, and P vs NP stay external.

## 11. Open questions and risks

- **Operator family is 01-13.** The scan-of-six-roles catalog
  (generator, role × scan, rejects) lives in spec 01-13. This ledger
  still owns named architectures.
- **Citation path is 06-05.** Inventions are not the publish-and-use
  order. Spec 06-05 owns the frozen public surface, the first paper,
  the jet-vs-AD bench, and the external-user obligation. This ledger
  must not be used as that path. 09-27 and 09-28 do not enlarge
  `PUBLIC_SURFACE` and 09-28 is not a ViT.
- **First-bet may fail.** If 09-03 + 09-17 cannot beat a named local
  baseline on a 1-D conservation identity, the integral-first story is
  recorded as unearned, not deleted.
- **Enclosure explosion.** 09-05 / 09-24 are allowed to die on depth
  greater than two. That is an accuracy floor, not a surprise.
- **Overlap with Group 02.** A reviewer may read 09-14 as 02-06. The
  surface-versus-volume sentence in the pointer table is the guard.
- **Falsifier for this ledger.** If a Group 09 implementation ships a
  rejected claim (global min, stretch cleared, new package, ImageNet
  win), G3/G4/G5 have failed.

## 12. Implementation checklist

- [x] `theory/09-inventions/01-invention-ledger.md` (this file)
- [x] `packages/omnibias-core/tests/test_theory_invention_ledger.py`
      asserting G1–G5 by reading the markdown
- [x] Index row in `theory/README.md` (Group 09; wired in the index pass)
- [x] No Python invention modules in this ledger pass

---

## Repo invariants this spec must respect

Check these before submitting an implementation.

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`. Pure-Python math lives there and every backend imports it.
- **Bit-identical twins**: torch and jax implementations must agree exactly.
  Polynomial coefficients come from `omnibias.core.polynomials`; never fork them
  per backend.
- **Default dtype**: use the framework default
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded
  `float32`.
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
