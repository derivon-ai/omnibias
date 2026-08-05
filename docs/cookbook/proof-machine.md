# The prove/disprove machine

omnibias produces many independently-checkable **certificates** — the
Constantin–Lax–Majda (CLM) and generalized-CLM blow-up certificates, the
multi-zero *earliest* blow-up theorem, the Birkhoff–Hopf spectral-gap
certificate, and more. Each has its own schema validator and, often, an
independent numpy/mpmath replay twin.

`omnibias.core.proof.ProofMachine` gives them **one front door**: you state a
`Conjecture`, the machine dispatches it to the right prover, runs the schema
gate, the independent replay, and the honesty gate, and returns a `Verdict`.

!!! info "What the verdicts mean"
    `PROVED` / `DISPROVED` / `BLOCKED` are defined precisely — including the
    three gates that can downgrade a result to `BLOCKED` — in
    [Scope & guarantees § 3.1](../scope-and-guarantees.md#31-the-provedisprove-machine-verdict-semantics).
    `PROVED` always means a **model** statement is certified, never that an open-problem
    problem is solved.

## Quick start

```python
from omnibias.pinn.certified import build_default_machine
from omnibias.core.proof import Conjecture

machine = build_default_machine()
sorted(machine.kinds())
# ['ccf_selfsimilar_blowup', 'clm_blowup', 'clm_multizero_blowup',
#  'gclm_gradient_amplification', 'gclm_selfsimilar_blowup', 'perron_spectral_gap']
```

### A certified blow-up (PROVED, with independent replay)

> **Not a global-regularity result.** This certifies finite-time blow-up in the 1D
> Constantin–Lax–Majda *model*; it says nothing about 3D Navier–Stokes
> global regularity (the global-regularity problem).

```python
v = machine.evaluate(
    Conjecture("CLM earliest blow-up", "clm_multizero_blowup",
               {"coeffs": [-1.0], "scales": [1.0]})
)
assert v.status == "PROVED"
assert v.replay_ok is True        # the numpy twin agrees
assert v.honesty_ok is True
print(v.certificate["first_blowup_time"])   # two-sided certified interval ~ [2, 2]
```

### A certified non-event (DISPROVED)

For `ω₀ = +q_1` the line-Hilbert transform is negative at the only zero, so the
CLM solution provably never blows up — and because the multi-zero prover
enumerates **all** zeros (exact Sturm), it can say so rigorously:

```python
v = machine.evaluate(
    Conjecture("no CLM blow-up", "clm_multizero_blowup",
               {"coeffs": [1.0], "scales": [1.0]})
)
assert v.status == "DISPROVED"
```

### The honesty gate (forged global-regularity claim → BLOCKED)

Asserting a claim the certificate does not back is always downgraded:

```python
v = machine.evaluate(
    Conjecture("overclaim", "clm_multizero_blowup",
               {"coeffs": [-1.0], "scales": [1.0]},
               claims={"unproven_claim": True})
)
assert v.status == "BLOCKED"
assert v.honesty_ok is False
```

A claim the certificate *does* back (e.g. `interval_verified`) passes:

```python
v = machine.evaluate(
    Conjecture("honest", "clm_multizero_blowup",
               {"coeffs": [-1.0], "scales": [1.0]},
               claims={"interval_verified": True})
)
assert v.status == "PROVED"
```

## The built-in provers

| `kind` | Certificate | Can DISPROVE? |
|---|---|---|
| `clm_blowup` | `certified_clm_blowup` (origin only) | No — origin-only; reports an upper bound on the first singularity time |
| `clm_multizero_blowup` | `certified_clm_multizero_first_blowup` | **Yes** — all zeros enumerated, so `max H ω₀ < 0` disproves blow-up |
| `ccf_selfsimilar_blowup` | `certified_ccf_selfsimilar_blowup_attempt` | No — radii-polynomial closure → `PROVED` (collocation profile) else `BLOCKED` with the quantified gap (see [CCF-on-the-line calibration](ccf-line-calibration.md)) |
| `gclm_selfsimilar_blowup` | `certified_gclm_selfsimilar_blowup` | No |
| `gclm_gradient_amplification` | `certified_gclm_gradient_amplification` | No |
| `perron_spectral_gap` | `certified_perron_spectral_gap` (`omnibias.core.verified.eig`) | No — a non-positive matrix or no certifiable gap → `BLOCKED` |

```python
# Birkhoff-Hopf spectral gap of a fixed positive transfer matrix:
v = machine.evaluate(
    Conjecture("gap", "perron_spectral_gap", {"matrix": [[2.0, 1.0], [1.0, 2.0]]})
)
assert v.status == "PROVED"
print(v.certificate["spectral_gap_lower"])   # > 0 (lattice-unit gap lower bound)
```

## The Lean formal loop (kernel-checked verdicts)

omnibias runs one derivative tower in three registers — **differentiable**,
**rigorous** (`omnibias.core.verified`), and **formal** (`formal/`). The proof
machine connects the last two: a certificate's *finite, rational* obligation can
be re-checked by a **Mathlib-free Lean 4 kernel** (`formal/omnibias-verified-kernel`),
and only a genuine `lake` pass sets `Verdict.theorem_prover_verified`.

```python
from omnibias.core.proof import (
    check_certificate, generate_obligation, lean_check_available,
)

# A Birkhoff-Hopf certificate carries a rational subdominant-ratio upper bound,
# i.e. a finite obligation the kernel can discharge:
cert = {"subdominant_ratio_upper": 0.625, "honesty": {"unproven_claim": False}}
print(generate_obligation(cert))          # emits Lean: `spectral_gap_pos ...`
if lean_check_available():                # True iff `lake` + the kernel are present
    print(check_certificate(cert).verified)   # runs `lake build` -> True
```

Wire it into a verdict with the `theorem_prover_verified` honesty claim — it
behaves exactly like the other honesty gates, but the backing evidence is a
**kernel proof**, not a certificate flag (so it can never be forged):

```python
from omnibias.pinn.certified import build_default_machine
from omnibias.core.proof import Conjecture

machine = build_default_machine()
v = machine.evaluate(
    Conjecture("gap", "perron_spectral_gap", {"matrix": [[2.0, 1.0], [1.0, 2.0]]},
               claims={"theorem_prover_verified": True})
)
# With a Lean toolchain: PROVED and v.theorem_prover_verified is True.
# Without one: BLOCKED (the asserted formal claim has no kernel backing).
```

The kernel proves the obligation by chaining its own *proven* `ZInterval`
soundness lemmas (no `sorry`); **infinite** analytic obligations are out of
scope and are not expressed in the Lean projects at all.
A mutated certificate fails its digest and is rejected before any Lean is
emitted, and when no toolchain is present the bridge degrades gracefully (the
flag stays `False`; an unclaimed verdict is unaffected).

Beyond the scalar sign / gap facts, the kernel now also discharges a
**matrix positive-definiteness** obligation: a `positive_definite` certificate
(the interval `LDLᵀ` pivot vector sealed by `certify_trained_min`) emits
`allPivotsPos [⟨lo,hi⟩, …] = true`, proven by the sorry-free
`Omnibias.matrix_positive_definite_certified` lemma — every pivot of the
factorisation is certified strictly positive (zero negative inertia), so the
strict-local-minimum Hessian claim is kernel-verified as a *matrix* fact, not a
single scalar. This is backed by a new, sorry-free proof of interval
**multiplication** soundness (`ZInterval.mem_mul`) in the Mathlib-free kernel.

## Adding your own prover

A prover is anything satisfying the `Prover` protocol; the easiest route is the
`FunctionProver` adapter, which wraps plain callables:

```python
from omnibias.core.proof import Conjecture, FunctionProver, ProofAttempt, ProofMachine

def prove(conj):
    cert = {"value": conj.data["x"], "honesty": {"unproven_claim": False}}
    status = "PROVED" if conj.data["x"] > 0 else "DISPROVED"
    return ProofAttempt(status=status, certificate=cert)

prover = FunctionProver(
    name="sign", kinds=frozenset({"sign"}),
    prove_fn=prove,
    schema_fn=lambda cert: [] if "value" in cert else ["missing value"],
    replay_fn=lambda cert: cert["value"] != 99,   # an independent recheck
)

machine = ProofMachine().register(prover)
machine.evaluate(Conjecture("pos", "sign", {"x": 3})).status   # 'PROVED'
```

The machine still applies the schema, replay, and honesty gates on top of your
prover, so a buggy or over-claiming prover is caught rather than trusted.

## Where this sits

- Engine (pure-Python, dependency-free): `omnibias.core.proof`.
- Default registry (wires the concrete certificates): `omnibias.pinn.certified.build_default_machine`.
- The certificates themselves: [Navier–Stokes certified validation](navier-stokes-certified.md),
  [CCF singularities](ccf-singularity.md).
