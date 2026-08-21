# 08-01 The training-idea ledger

## 1. Thesis and status

The closed-form derivative tower makes a small family of **tower-native
trainers** implementable — exact jets, exact composed Hessians, certified
accept/reject — and this file is the index that says which of those ideas are
already specified, which are new Group 08 specs, which are rejected, and which
stack is the default.

- **Status**: designed
- **Depends on**: 01-01, 01-10, 03-12, 06-01, 06-02
- **Blocks**: 08-02, 08-03, 08-04, 08-05, 08-06, 08-07, 08-08, 08-09

### Operator card

- **Benefit.** One page answers "what trainer should we implement next" without
  re-deriving the honesty constraints or the CCF floor.
- **How it works.** Two taxonomies (optimizer vs learning rule), a recommended
  stack, a comparison table, and a reject list. Every Group 08 spec names a
  parent row here.
- **Strength.** Stops a better local step from being written as a global
  solver, and stops a new optimizer from being aimed at `1e-13` CCF stretch.
- **When to use.** Before implementing any 08-* file, or when proposing a
  ninth trainer.
- **When not.** This file does not train a network.
- **Accuracy floor.** None of its own. CCF stretch remains an operator floor
  (`~1e-1` Hilbert / dictionary), not a trainer floor.

## 2. Where it lands

A document. No module and no package. The same pattern as spec 07-01: the
ledger is the index; the other Group 08 files are the entries. Implementation
homes are declared per entry in section 2 of those files and summarized in
spec 06-03.

## 3. Prior art in omnibias

Shipped trainers and jets — Group 08 must not re-derive them.

- `omnibias.{torch,jax}.optim` — `GaussNewton`, `CubicNewton`,
  `CubicRegularizedNewton`, `CubicRegularizedGaussNewton`, `NaturalGradient`,
  `JetSubspaceTensor`, `TrustRegionNewtonCG`, `JetLBFGS`, `hvp`,
  `taylor_line_min`, `taylor_subspace_model`, `martens_grosse_gauss_newton_minimize`.
- `omnibias.{torch,jax}.jet` — `compose_jet`, `layer_jet`, `mlp_jet`,
  `affine_jet`. Exact directional jets through depth; Faà di Bruno is the
  compositional rule.
- `omnibias.core.polynomials` — shared `sigma^(n)` coefficients. Every 08
  trainer that evaluates an activation derivative imports these, never a
  per-backend fork.
- `omnibias.core.verified.kantorovich` — `radii_polynomial_certificate`,
  `newton_kantorovich_bounds`, `krawczyk_certificate`. Unique-zero balls.
- `omnibias.pinn.train` — `causality_index`, Wang–Perdikaris causal weights,
  `march` (time slabs). Depth-causal residual training (08-05) is the
  analogue in *network* depth, not a rewrite of time marching.
- `omnibias.score.flow` — `cnf_dynamics`, `integrate_cnf`. Depth-as-fluid
  already exists as a *model*, not as a new Group 08 trainer.
- `omnibias.verify` — `lipschitz_bound`, `taylor_output_bounds`. Input-output
  certificates for 08-09.
- Spec 03-12 — exact jet line search (designed, unbuilt as a module; `taylor_line_min`
  is a related but not certified-radius implementation).
- Spec 03-10 — jet–Padé singularity tracking. Diagnostic, not a step rule.
- Spec 03-11 — Lie symmetry. Constrains the hypothesis class.
- Spec 03-13 — adaptive pack refinement. Architecture as the learner.
- Spec 04-01 — closed-form Fisher / natural gradient (G2 earned).
- Spec 05-02 — arrangement / temperature collapse. A different hypothesis
  class, not a generic-MLP trainer.

**Confirmed gap.** There is no index that separates "how to step `theta` given
`L`" from "what signal a layer may use before `L` is known," and no written
reject list for "skip the chain rule" or full `d h / d theta` transport.

## 4. Mathematics

### Two taxonomies

An **optimizer** is a map `(theta, L) -> theta'`. It may use exact jets of
`L` along a direction, exact Hessians, or a radii polynomial. It still
evaluates a *global* (or end-to-end residual) objective. Specs 03-12, 08-04,
08-06, 08-07, 08-08 (as a step on the implicit residual) sit here.

A **learning rule** is a map from *layer-local* information to a weight
update that does not wait for the final `L`. Depth is causal: layer `ell`
may use only jets computed from layers `1 .. ell`. Specs 08-02 (joint two-layer
exception), 08-03, 08-05 sit here. 08-09 is an accept/reject *filter* on any
of the above.

Faà di Bruno **is** the chain rule at every order. No 08 spec may claim to
avoid compositional differentiation. The founding **bias collapse**
(`delta -> 0`) supplies `sigma^(n)` and the jets. None of 08-02…08-08 is
temperature collapse (`beta -> inf`) unless it delegates to 05-02.

### Recommended stack

```
objective (end-to-end L, or PDE residual)
    -> 08-02 if a layer-wise slice has stalled (joint negative mode)
    -> GaussNewton / CubicNewton          (direction)
    -> 03-12 jet line search              (length)
    -> 08-04 Kantorovich accept / reject  (unique-zero ball)
    -> 08-06 sharpness                    (cubic lambda / lr)
```

08-03 / 08-05 are an **alternative** information path (local residual), used
as a warm start or as a PINN-depth march, then handed to the stack above.
08-07 is a structured special case of 03-12. 08-08 replaces unrolled depth
with a fixed point. 08-09 is optional and is not the CCF path.

### Rejected (no spec)

1. **"Skip the chain rule and solve for a global min."** A deep net is a
   composition. Expanding `L(theta)` as a polynomial in all weights is
   exponential in depth. Exact `sigma^(n)` does not convexify `L`.
2. **Full `d h / d theta` as a "flow."** That Jacobian is `width x n_params`.
   Every 08 learning rule carries `k << P` directions or `d / d x` (the PINN
   jet). An API that materializes the full parameter Jacobian must raise.

### CCF / stretch floor

`CCF_STRETCH_RESIDUAL_GATE = 1e-13` is an **operator** gate. The recorded
Hilbert × dictionary catch-22 floors near `1e-1`. No 08 trainer, and no
amount of Martens–Grosse, clears that floor. Group 08 files must not weaken
the stretch gate and must not infer Navier–Stokes regularity from a better
step.

### Comparison (designed, not measured)

Costs are versus one forward+backward. `L` = depth, `N` = jet order,
`k` = stored directions, `P` = parameters.

| Entry | Extra cost / step | Local rate | Typical floor |
|---|---|---|---|
| Adam / backprop (shipped) | `1x` | empirical | residual operator |
| CubicNewton / GN (shipped) | `O(k)` HVP or one GN solve | superlinear near a root | residual operator / singular `J` |
| 03-12 jet line search | `O(L N^2)` once | same as the direction method | jet truncation `R_N` |
| 08-02 joint Hessian | `O(k^2)` in a 2-layer subspace | leaves a *slice* critical point | model class; not global |
| 08-03 / 08-05 local | `O(L k)` | no end-to-end guarantee | greedy bias |
| 08-04 Kantorovich | GN + interval residual | quadratic **and** a uniqueness ball | empty ball => reject |
| 08-06 sharpness | 1–few HVPs | still first/second-order | HVP budget |
| 08-07 block search | cheap per block | exact on that line | block structure required |
| 08-08 DEQ + IFT | one linear solve | implicit-depth Newton | fixed-point quality |
| 08-09 certified step | interval bound | may refuse steps | enclosure explosion |

Nothing in this table is a global solver for a deep nest. The only guarantees
stronger than Adam are **local quadratic** (Newton family) and **local
uniqueness** (Kantorovich). 08-02 is the only entry whose purpose is "leave
this layer-wise min."

## 5. Worked example

**Triaging a proposed ninth trainer.**

*Proposal:* "During the forward pass, store `d h_i / d W` for every weight
and flow those derivatives like a fluid into the next layer."

- Taxonomy: learning rule (causal in depth).
- Test against reject 2: the stored object is full `d h / d theta`. **Fails.**
  Rewrite as 08-03: store `k` directional jets or `d h / d x`.
- CCF test: if the claim is "this hits `1e-13`," **fails.** Point at 07-03.
- Stack: after the rewrite, 08-03 is a warm start; direction still comes from
  GN.

*Proposal:* "Use previous-layer curvature so the current layer is not stuck."

- Taxonomy: learning rule with a two-layer exception.
- Identity: `(f circ g)'' = f''(g) (g')^2 + f'(g) g''`. Already 08-02.
- Escape criterion: `lambda_min(slice) >= 0` and `lambda_min(joint) < 0`.
  If the joint block is also PD, this is ordinary Newton, not escape.

## 6. Proposed API

Does not exist as code. The ledger is a table and a recommended call order.
A later structural test (optional, same spirit as 07-01) may assert that
every `theory/08-training/*.md` except this file names a row in section 4
and that the two rejected sentences do not appear as claims in Group 08.

```python
# documentation only — not a module
LEDGER_OPTIMIZERS = (
    "03-12", "08-04", "08-06", "08-07", "08-08",
)
LEDGER_LEARNING_RULES = (
    "08-02", "08-03", "08-05",
)
LEDGER_FILTERS = ("08-09",)
REJECTED = (
    "skip_chain_rule_global_min",
    "full_parameter_jacobian_flow",
)
```

## 7. Practical use cases

1. **Choosing an implementation order.** Build 03-12 first (already specified),
   then 08-02's 2-layer PINN ablation (the only new falsifier that can kill
   the curvature-escape story in a day), then 08-04 on a toy root.
2. **Writing a release note** for a trainer without implying Clay, stretch,
   or a global min.
3. **Refusing a ninth idea** that is reject 1 or 2, or that duplicates 03-12 /
   `CubicNewton` / CNF.
4. **CCF campaign hygiene.** Any PR that claims a trainer moved stretch must
   fail this ledger's CCF paragraph.

## 8. Acceptance gates

- **G1 completeness.** Every `theory/08-training/*.md` except this file
  appears in the section-4 table with taxonomy, home, and floor.
- **G2 rejects named.** The two rejected proposals are written in section 4
  and do not have implementation files.
- **G3 no stretch inference.** No Group 08 spec sets or weakens
  `CCF_STRETCH_RESIDUAL_GATE`, and each implementable spec's honesty section
  restates the Hilbert floor.
- **G4 no global-min claim.** Each implementable spec's honesty section
  forbids "global minimum of a deep nest" and "we do not use the chain rule."
- **G5 recommended stack is implementable.** Every name in the stack exists
  as a shipped symbol or as a Group 08 / 03-12 spec.

These are document gates (checked by review / a later structural test), not
numerical gates. Numerical gates live in 08-02…08-09.

## 9. Benchmark plan

None of this file's own. Each entry names `$OMNIBIAS_SCRATCH/training/<name>/`
and a committed smoke JSON under `docs/benchmarks/` when implemented.
No CI job for the ledger.

## 10. Honesty and scope

- The tower is the founding **bias collapse** (`delta -> 0`). Group 08 does
  not introduce a third sense of "collapse."
- A Mathlib- or kernel-verified finite obligation is still a finite
  obligation (spec 06-02). 08-04's uniqueness ball is not a continuum PDE
  theorem.
- Certificate tiers used by entries: empirical gates (most), sound enclosure
  (08-04, 08-09, 03-12 radius). `theorem_prover_verified` is earned only by
  a genuine `lake build` of the Mathlib-free kernel and is not a trainer flag.
- Shipped CNF / natural gradient / arrangement are **not** re-specified here.

## 11. Open questions and risks

- **Group 09.** New architectures and non-trainer ideas (jet-valued state,
  integral-first cells, three-register forwards, invert-on-`x`, meta-loops,
  exports) live in Group 09. This ledger still owns trainers that step
  `theta` given `L`.
- **`taylor_line_min` vs 03-12.** A univariate Taylor minimizer already
  exists in `optim.py`. 03-12's delta is a certified truncation radius,
  Wolfe-as-interval, and `verify=True` never-worse. If G4 of 03-12 fails,
  the stack uses `taylor_line_min` and 03-12 is recorded as unearned, not
  deleted from this ledger.
- **08-02 may fail.** If the joint Hessian never has `lambda_min < 0` when
  the slice is PD, the escape story is false and 08-02 becomes a more
  expensive Newton. That is an allowed outcome.
- **Local rules (08-03, 08-05) can hurt.** Greedy bias is the predicted
  failure. Gates require a warm-start or last-layer comparison, not a claim
  that locality beats backprop on ImageNet.
- **Falsifier for this ledger.** If a Group 08 implementation ships a
  rejected claim (global min, full `d h / d theta`, stretch cleared by
  Adam/GN), G3/G4 of this file have failed and the implementation is wrong
  even if the numbers look good.

## 12. Implementation checklist

- [x] `theory/08-training/01-training-idea-ledger.md` (this file)
- [ ] Optional `packages/omnibias-core/tests/test_theory_training_ledger.py`
      asserting G1–G4 by reading the markdown (same spirit as 07-01's proposed
      guard)
- [x] Index row in `theory/README.md` (Group 08)
- [ ] No Python trainer modules in the spec-only pass

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
