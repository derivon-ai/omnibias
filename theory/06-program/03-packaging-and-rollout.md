# 06-03 Packaging and rollout

## 1. Thesis and status

Fifty-four specs would be thirty new packages if each were shipped the obvious
way, and the repository has already learned that lesson the expensive way — so
this file assigns every spec a home under the "earn independent existence" rule
and sequences them so that the cheap falsifiers run before the expensive builds.

- **Status**: designed
- **Depends on**: 06-01, 06-02
- **Blocks**: none

## 2. Where it lands

This file, plus `AGENTS.md`'s package inventory when a spec actually ships.
Nothing to build.

## 3. Prior art in omnibias

- `AGENTS.md` — the "earn independent existence" rule: a new distribution needs
  a distinct **domain**, a distinct **dependency or maturity tier**, or a
  distinct **audience**. Failing that, ship a submodule.
- The cautionary record, stated in `AGENTS.md` itself: `omnibias-score` and
  `omnibias-qpinn` are named as stubs that inflated the tree, and `pde`, `gauge`
  and `flow` were **folded back** into `pinn`, `geometry` and `score`. Folding
  out later is cheap; un-shipping a premature distribution is not.
- `packages/omnibias-core/tests/test_package_registry.py` — enforces
  workspace / folded-name / Python-floor consistency, so a fold is mechanically
  checkable.
- The `omnibias-dev-new-package` skill — the scaffolding procedure when a
  package is genuinely warranted: pyproject, SPDX headers, workspace exclude,
  sorted `__all__`, CI job, `docs/api` page plus mkdocs nav, `llms.txt` and
  `CHANGELOG.md` entries.
- 42 existing packages, most of which are the right home for one of these specs.

**No gap.** The rule and the enforcement exist. This spec applies them.

## 4. Mathematics

None. The content is an assignment and an ordering.

### Homes

Almost everything lands in an existing package.

| Home | Specs |
|---|---|
| `omnibias-core` | 01-04 stencils, 01-10 jet bundle (docs), 01-11 Lean obligations, 03-06 quadrature |
| `omnibias-torch` / `omnibias-jax` (twins) | 01-01 multi-pack, 01-02 scan, 01-07 order-as-frequency, 02-01 Scan-Net, 02-03 Jet-KAN, 02-08 equivariant scan, 03-12 line search, 03-13 refinement |
| `omnibias-fields` | 01-05 mollifier calculus, 02-07 pack tree |
| `omnibias-pinn` | 02-04 weak-form VPINN, 02-05 transmission PINN, 02-06 BEM-Net, 02-12 equality-intersection layers, 02-13 linearizing transforms, 05-01 inverse problems |
| `omnibias-geometry` | 01-03 arrangements (geometry side), 02-14 Wilson-line band (`geometry.gauge`) |
| `omnibias-verify` | 03-08 certified localization, 04-02 uncertainty |
| `omnibias-curvature` | 04-01 information geometry |
| `omnibias-discrete` / `-qubo` | 03-01 evolution, 03-03 CSP |
| `omnibias-convex` | 03-02 arrangement LP |
| `omnibias-shape` | 03-05 morphology, 03-09 topology, 05-02 shape part |
| `omnibias-tab` | 05-02 tabular part |
| `omnibias-symbolic` | 03-10 Padé tracking, 03-11 Lie symmetry |
| `omnibias-measure` | 03-04 sliced OT |
| `omnibias-difference` | 01-08 tropical homotopy (algebraic side) |
| `omnibias-dynamics` | 07-06 validated orbits |
| Docs only | 01-09 equality locus (math shared by 02-12), 01-06 wavelet frames, 01-12 conjugate tower (theory; the code lands in `core.verified.hardy_line`), 06-01…06-04, 07-01 |

### The one package that might be earned

**`omnibias-arrangement`** — specs 01-03, 02-02, 03-02, 03-09, 05-02's tabular
part. Test it against the rule:

- Distinct domain? Yes: combinatorial geometry of hyperplane arrangements —
  cells, flats, face lattices, tope graphs — which is not any existing package's
  subject.
- Distinct dependency tier? No: it needs only `core` plus torch/jax, same as
  most of the tree.
- Distinct audience? Partly: computational-geometry users are not the PINN
  audience.

That is one clear yes and one partial. **The rule's verdict is: start as
`omnibias.geometry.arrangement`, and promote only if two independent consumers
outside `geometry` depend on it and the submodule exceeds roughly 2 000 lines.**
Recording the promotion criterion now is the point; deciding by feel later is
how the stub packages happened.

### Rollout waves, ordered by information per unit of effort

**Wave 0 — falsifiers first.** Cheap experiments that can kill an expensive
line of work before it is built.

1. 05-02's G1 (arrangement versus LightGBM on the constructed oblique dataset) —
   kills or licenses the tabular application in a day.
2. 05-02's G5 (transverse filter versus a structured state-space model) —
   expected to fail; run it and delete the sequence submodule from the plan if
   it does.
3. 04-01's G2 (the `delta^2` Fisher degeneracy exponent — pack spread
   `delta`, **not** the scan tempering scale `alpha`; `alpha` belongs to
   05-01 G7) — validates or refutes the architectural argument that the
   whole tree leans on.
4. 05-01's G7 (the `alpha^(n - 5/2)` localization scaling) — validates the scan
   design rule before anything is built on it.

Wave 0 is the highest-value work in the program and costs the least.

**Wave 1 — the primitives everything else imports.** 01-01 multi-pack, 01-02
scan, 01-04 stencils. Nothing in Groups 02, 03 or 05 can be built honestly
before these, and they are also the specs most likely to reveal that a downstream
design is unimplementable.

**Wave 2 — guards.** 06-01's gate extensions and 06-02's claim guards. These
must precede Group 07 rather than follow it: a frontier result produced before
the guards exist will be described in whatever language its author reached for.

**Wave 3 — architectures with the strongest independent motivation.** 02-04
weak-form VPINN and 02-05 transmission PINN (both plug into an existing,
benchmarked PINN surface), 02-03 Jet-KAN (a well-defined comparison against
spline KANs), 03-12 line search and 03-13 refinement (small, self-contained,
immediately useful to existing optimizers).

**Wave 4 — the frontier program.** Group 07, in the order 07-01 (the ledger,
which is a document), 07-03 (CCF, which has a live campaign and a live gate),
then 07-02, 07-04, 07-05, 07-06, 07-07.

**Wave 5 — speculative.** 02-06 BEM-Net, 02-09 solitons, 02-10 Hermite ladder,
02-11 transfer matrices, 02-13 linearizing transforms, 03-04 sliced OT, 03-07
scale flow. Each is interesting; none blocks anything; all should wait for
evidence that the primitives hold up.

### What to do when a wave-0 falsifier fails

Delete the dependent specs from the plan and record why in the index. A spec
whose falsifier failed and which remains listed as "future work" is how a
research program accumulates dead weight that nobody feels authorized to remove.

## 5. Worked example

**Applying the rule to a spec that looks like a package and is not.**

Spec 02-01 Scan-Net is a network architecture with layers, a training loop, and
its own benchmarks. The instinct is `omnibias-scan`. Test it:

- Distinct domain? No. It is a composition of the bias scan (01-02), which is a
  torch/jax activation-level primitive, into a layer stack. `omnibias.torch`
  already hosts `unit`, `blocks` and `growable`.
- Distinct dependency tier? No. `core` plus one backend, identical to
  `omnibias-torch`.
- Distinct audience? No. Anyone using Scan-Net is already using OMBU layers.

Three noes. It ships as `omnibias.torch.scan` and `omnibias.jax.scan`, bit-identical
twins, and the fact that it has its own benchmarks is irrelevant — `omnibias.torch`
already has benchmarks.

**Now the contrast.** Spec 02-14, the Wilson-line holonomy band, could be
`omnibias-gauge`. But `gauge` was *already folded* into `omnibias.geometry` for
exactly this reason, and `AGENTS.md` records it. Re-creating it would repeat a
documented mistake, so it ships as `omnibias.geometry.gauge.holonomy`.

**And the honest maybe.** `omnibias.geometry.arrangement` starts as a submodule
with a written promotion criterion, which is the only case in the tree where a
new distribution is even plausible.

**Net effect: 54 specs, 0 new packages at the start, at most 1 later.** If that
number grows during implementation, the rule is being bypassed, and the tree
inventory in `AGENTS.md` is the place it will show.

## 6. Proposed API

None. This spec produces an assignment table and an ordering.

The one mechanical deliverable is a test:

```python
# packages/omnibias-core/tests/test_theory_homes.py
def test_every_spec_declares_a_home():
    """Each theory spec's section 2 names an existing package or an explicitly
    justified new one; the set of new packages proposed across the tree is
    checked against the allowlist in this file."""

NEW_PACKAGES_ALLOWED: frozenset[str] = frozenset()   # empty, by design
```

Starting the allowlist empty means adding a package requires editing a test with
a justification, which is the right amount of friction.

## 7. Practical use cases

1. **Deciding where a spec's code goes** without relitigating the packaging
   question each time.
2. **Sequencing a quarter of work** so that falsifiers precede builds.
3. **Resisting package proliferation** with a written criterion rather than
   taste.
4. **Knowing what to delete** when a falsifier fails.

## 8. Acceptance gates

- **G1 zero new packages at start.** After Waves 0 through 3, the package count
  in `AGENTS.md` is unchanged. Checked by `test_package_registry` plus review.
- **G2 homes declared.** Every spec's section 2 names a real, existing module
  path, or a new submodule inside one. Checked by
  `test_theory_homes.py`.
- **G3 falsifiers first.** No Wave 1+ implementation lands before its Wave 0
  falsifier has run and its result is recorded in the index, pass or fail.
- **G4 deletion discipline.** Any spec whose falsifier failed is marked
  `retired` in the index within the same change that records the failure, with
  a one-line reason.
- **G5 promotion criterion.** If `omnibias.geometry.arrangement` is promoted to
  a package, the two-external-consumer and size criteria are demonstrated in the
  promoting change, not asserted.

## 9. Benchmark plan

Not a benchmark. Deliverables: the assignment table above kept current, the
`test_theory_homes.py` guard, and index status transitions recorded as work
proceeds.

## 10. Honesty and scope

- The wave ordering is a **judgement about information per unit of effort**, not
  a dependency graph. The real dependencies are in each spec's "Depends on".
- "Zero new packages" is a target, not a virtue in itself. If a spec genuinely
  earns a distribution, it should get one — the allowlist is a speed bump, not a
  wall.
- The promotion criterion for `arrangement` (two external consumers, ~2 000
  lines) is a heuristic chosen to be checkable, not a principle. Its only real
  job is to be decided in advance.
- No claims of any kind are made here; no collapse limit appears.

## 11. Open questions and risks

- **The submodule that should have been a package.** The rule is biased toward
  submodules, and the failure mode is a package (say `omnibias-pinn`) that grows
  until nobody can navigate it. Watch total lines per package, and fold *out*
  when a submodule has a genuinely separate audience.
- **Wave 0 results may be ambiguous.** A falsifier that neither clearly passes
  nor clearly fails is the worst outcome, because it licenses continuing without
  evidence. Each falsifier should state its ambiguous-outcome policy, and the
  default should be to treat ambiguity as failure.
- **Effort estimates are absent.** The waves are ordered by information value,
  not cost, and a Wave 1 primitive could be a month of work. Ordering by
  information value is still right, but the schedule cannot be read off it.
- **Falsifier for this spec.** If, after Wave 3, the package count has grown,
  the rule was not actually being applied and the assignment table needs to be
  rebuilt from what happened rather than from what was planned.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/tests/test_theory_homes.py` with an empty
      `NEW_PACKAGES_ALLOWED`
- [ ] Keep the assignment table current as specs move
- [x] Run Wave-0 falsifiers A6 (04-01 G2), A7 (05-01 G7), and A4 (05-02 G1/G2)
      before Wave 1+; A5 still pending
- [x] Record A6, A7, and A4 falsifier outcomes in `theory/README.md` (all passed)
- [ ] Mark failed-falsifier specs `retired` with a reason
- [ ] Written promotion criterion for `omnibias.geometry.arrangement` in its
      module docstring
- [x] Index row in `theory/README.md`
