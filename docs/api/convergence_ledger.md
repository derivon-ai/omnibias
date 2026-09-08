# Convergence ledgers (07-08)

A staged iterative construction (a wave-packet ladder, a cluster
expansion) closes if and only if a finite list of affine margin
inequalities holds on a rational, possibly unbounded, feasible
interval. That collapse is the whole trick, and it is decidable:
each residual is affine, so `< 0` is decided by endpoint values plus
slope sign. No LP solver, no floats.

This is the finite spine shared by OpenAI's Navier–Stokes
`ExponentLedger` (Clay alternatives (C) and (D)) and the
Kotecký–Preiss polymer-coordination criterion behind strong-coupling
Yang–Mills. The checker certifies the **algebra**. It does not
construct a correction, define an analytic class, or prove a PDE.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/convergence_ledger.py`). Parent-level honesty flags
(`navier_stokes_proof_claim`, `yang_mills_mass_gap_claim`) are
**derived**: they become true only when every margin discharges **and**
`external_premises` is empty. The three curated ledgers (NS exponent,
polymer, NS scale) ship with a printed premise list, so the honest
outcome today is `CONDITIONAL`. See theory spec
[07-08](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/08-convergence-ledgers.md).

Home: `omnibias.core.proof.obligations.convergence_ledger`.

```python
from fractions import Fraction

from omnibias.core.proof.obligations.convergence_ledger import (
    check_ledger,
    navier_stokes_exponent_ledger,
    seal_ledger_certificate,
    strong_coupling_polymer_ledger,
)

ns = navier_stokes_exponent_ledger()
report = check_ledger(ns)
assert report.holds
assert report.strength == "CONDITIONAL"
assert report.binding_threshold is not None
assert report.binding_threshold[0] == "kappa"
assert report.binding_threshold[2] == Fraction(1, 200)
assert ns.external_premises

sealed = seal_ledger_certificate(ns, run_lean=False)
assert sealed.mathlib_verified is False
assert sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False
assert sealed.certificate["payload"]["type"] == "convergence_ledger"

polymer = check_ledger(strong_coupling_polymer_ledger())
assert polymer.holds
assert polymer.strength == "CONDITIONAL"
assert polymer.failing == ()
```

A deliberately failing margin is named, not swallowed.

```python
from omnibias.core.proof.obligations.convergence_ledger import (
    check_ledger,
    failing_margin_ledger,
)

failed = check_ledger(failing_margin_ledger())
assert failed.holds is False
assert failed.strength == "BLOCKED"
assert "particular" in failed.failing
```

An unbounded stage interval is decided by slope sign, not by sampling.

```python
from omnibias.core.proof.obligations.convergence_ledger import (
    check_ledger,
    unbounded_slope_ledger,
)

assert check_ledger(unbounded_slope_ledger(holds=True)).holds
assert check_ledger(unbounded_slope_ledger(holds=False)).holds is False
```

Kernel emission is `allRatLt` over integer cross-multiplications
(`Omnibias.RationalStencil`). The Mathlib-backed checker applies
`ns_manuscript_margins` / `polymer_ledger_margins` in
`OmnibiasAnalytic.Check.ConvergenceLedger`. `theorem_prover_verified`
is earned only by a genuine kernel `lake build`. `mathlib_verified`
is a distinct tier.

## API

::: omnibias.core.proof.obligations.convergence_ledger
    options:
      show_root_heading: false
      heading_level: 3
