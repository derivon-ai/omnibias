# omnibias-ferminet

FermiNet bridge for the omnibias closed-form n-th derivative framework.

## Install

```bash
pip install omnibias-ferminet
# Optional: pull in folx for the bit-stable Laplacian comparison test
pip install omnibias-ferminet[folx]
```

`omnibias-ferminet` depends on `omnibias-core`, `omnibias-jax`, and
`jax>=0.4.30`. It does **not** require a FermiNet installation to import
or to be unit-tested (mock log|psi| factories are provided for the bridge
contracts); a real FermiNet checkout is needed only at production VMC time.

## What is in here

- ``omnibias.ferminet.folx_compat``: ``forward_laplacian``,
  ``closed_form_forward_laplacian``, ``laplacian_factory`` -- folx-API
  drop-ins so any FermiNet / DeepQMC code that imports from ``folx`` can
  switch to omnibias's closed-form Laplacian by changing one import.
- ``omnibias.ferminet.integration``: the production bridge --
  envelope value / gradient / Hessian, optional one-body backflow, and
  the ``make_omnibias_envelope_local_kinetic_energy`` and
  ``make_omnibias_tier2_local_kinetic_energy`` factories that the
  upstream FermiNet ``laplacian_method`` switch consumes.
- ``omnibias.ferminet.restricted``: Tier-2 / Tier-2-full restricted
  FermiNet ansatz with end-to-end closed-form Laplacian (used to bring
  the bridge up before plugging into the upstream FermiNet checkpoint).
  ``tier2_grad_laplacian_log_psi`` exposes the gradient and Laplacian of
  ``log|det M|`` separately (a bit-identical refactor of the kinetic path)
  so an additive correlation factor composes correctly.
- ``omnibias.ferminet.jastrow``: closed-form symmetric **Padé-Jastrow**
  correlation factor ``exp(J)`` -- analytic value / gradient / Laplacian
  (``jastrow_value_grad_laplacian``) with physical e-e (``1/2``, ``1/4``)
  and e-n (``-Z``) cusp slopes, plus ``jastrow_slater_local_kinetic_energy``
  combining it with a Slater determinant (correct cross term in
  ``|grad log|psi||^2``).
- ``omnibias.ferminet.multiblock`` and
  ``omnibias.ferminet.multiblock_integration``: multi-block FermiNet
  primitives for Born-Oppenheimer / nuclear-Hessian work.

## Public API (lazy)

```python
from omnibias.ferminet.folx_compat import (
    forward_laplacian, closed_form_forward_laplacian, laplacian_factory,
    OmnibiasFwdLaplResult,
)
from omnibias.ferminet.integration import (
    envelope_value_grad_hessian,
    apply_optional_backflow,
    make_omnibias_envelope_local_kinetic_energy,
    make_omnibias_tier2_local_kinetic_energy,
)
from omnibias.ferminet.restricted import (
    Tier2Params, Tier2SymParams,
    tier2_log_abs_psi, tier2sym_log_abs_psi,
    tier2_local_kinetic_energy, tier2sym_local_kinetic_energy,
)
```

The top-level ``omnibias.ferminet`` namespace deliberately does **not**
re-export these: importing the bridge is opt-in so the package stays
cheap to import in scripts that do not need FermiNet.

## Contract

`omnibias-ferminet` produces a Laplacian that is bit-identical to
FermiNet's `default` autograd path on the closed-shell systems we have
validated (Be 4-electron 4-walker test in
the private release-validation archive
row R1: `rel_err = 0.0 to ULP`, `vs folx <= 5.07e-15`). This is a
**bit-stable** contract for the CPU deterministic forward path; on
stochastic GPU runs we hold to a 1-sigma reproducibility budget.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
