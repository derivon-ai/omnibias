# 09-13 Jet-Hopfield memory

## 1. Thesis and status

Store and retrieve **germs** (value plus derivatives), not vectors, so
associative memory matches contact.

- **Status**: gated
- **Depends on**: 01-10, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Patterns are jets; retrieval uses closed-form
  log-sum-exp curvature on a jet metric.
- **How it works.** Memories `J_mu` are `N`-jets. Query jet `J` is
  scored by a contact mismatch (value + scaled 1-jet). Retrieve with
  existing `modern_hopfield_retrieve` / `attention` on those scores,
  optionally using `logsumexp_hessian`.
- **Strength.** 1-D profile dictionaries (several known shapes).
- **When to use.** After jet-token; when hopfield users have jets.
- **When not.** As a rewrite of vector hopfield. Not ImageNet
  retrieval SOTA.
- **Accuracy floor.** Contact metric choice; value-only hopfield is
  the baseline that must be beaten on jet error.

## 2. Where it lands

`omnibias.hopfield`. No new package.

## 3. Prior art in omnibias

- `omnibias.hopfield.{torch,jax}.ops.hopfield.modern_hopfield_retrieve`
- `attention`, `logsumexp_value`, `logsumexp_jacobian`,
  `logsumexp_hessian`, `hopfield_energy`
- Spec 09-02 — jet residual stream (different object: a transformer,
  not a memory)

**Confirmed gap.** Memories are vectors. Nothing stores an `N`-jet as
a pattern.

## 4. Mathematics

Contact mismatch (order 1):

```
d(J, J_mu)^2 = |u - u_mu|^2 + lam |u' - u_mu'|^2
```

Hopfield logits are `-beta d^2`. Retrieval is the existing softmax
memory. `beta -> inf` is **temperature collapse** (hard nearest
germ) and must be labelled; default finite `beta`. Jets of stored
profiles may come from **bias collapse** of an OMBU dictionary.

## 5. Worked example

Two memories: `J1=(1.0, 0.0)`, `J2=(0.0, 1.0)`. Query
`J=(1.0, 0.01)`, `lam=1`, `beta=10`. Distance^2 to J1 is `1e-4`, to
J2 is `1+0.99^2=1.9801`. Softmax mass on J1 is
`e^{-0.001} / (e^{-0.001}+e^{-19.801}) ≈ 1`. Retrieved value must be
within `1e-6` of `1.0`. A value-only memory (`lam=0`) still
retrieves J1 here; G2 uses a query `(0.6, 1.0)` that is closer in
value to a third memory `(0.5, 0.0)` but closer in contact to
`(0.0, 1.0)`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/hopfield/torch/ops/jet_hopfield.py  (and jax twin) — proposed
@dataclass(frozen=True)
class JetHopfieldConfig:
    jet_order: int = 1
    lam: float = 1.0
    beta: float = 1.0

def jet_hopfield_retrieve(query_jet, memory_jets, *, config: JetHopfieldConfig):
    """Retrieve a germ. Uses logsumexp_* on contact scores."""
```

## 7. Practical use cases

1. **Dictionary of 1-D kinks** (02-09 adjacent) retrieved by slope.
2. **PINN profile library** (several analytic solutions).
3. **Not** embedding retrieval for language.

## 8. Acceptance gates

- **G1.** Worked near-J1 query retrieves value `1.0` within `1e-6`.
- **G2 skill.** Named three-memory contact probe: jet-Hopfield
  retrieves the contact-nearest memory; value-only hopfield retrieves
  the other. Five seeds of noise `1e-3` must keep this split.
- **G3 honesty.** `temperature_collapse_used` recorded.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/jet_hopfield.py`
- Smoke: `docs/benchmarks/jet_hopfield_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/jet_hopfield/`

## 10. Honesty and scope

- Temperature collapse only if `beta -> inf`. Not CCF. Not ImageNet.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Scale of `lam`.** G2 must fix `lam` in the artifact.
- **Falsifier.** Contact and value nearest coincide on all probes.

## 12. Implementation checklist

- [x] `omnibias.hopfield` jet retrieve twins
- [x] Contact-vs-value split test
- [x] `benchmarks/jet_hopfield.py` plus smoke JSON
- [x] `__all__` update
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
