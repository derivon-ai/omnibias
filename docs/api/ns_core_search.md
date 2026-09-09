# NS-core profile search (07-14)

A finite family of axis-regular jets ``F = c(1 + a X + b X^2)`` whose
score is the similarity residual ``T_b(F) - 1``, not the independent
plant ``R = r^2``. Accept/reject includes the 07-10 cone. The locked
07-09 origin still discharges; ``(c, a, b) = (1, 3, -1)`` is a second
witness.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/ns_core_search.py`). This is a fragment of a constructed
forced blowup. It does not re-prove Clay (C)/(D) and does not touch
unforced (A)/(B). See theory spec
[07-14](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/14-similarity-profile-search.md).

Home: `omnibias.pinn.certified.anisotropic` plus
`omnibias.pinn.jax.discovery.ns_core`.

```python
from omnibias.pinn.certified.anisotropic import (
    NS_CORE_OPPOSITE,
    NS_CORE_ORIGIN,
    NS_CORE_SECOND_WITNESS,
    check_ns_core_candidate,
    honesty_payload,
    locked_axis_regular_profile,
    profile_similarity_residual,
)
from omnibias.pinn.jax.discovery.ns_core import run_ns_core_discovery

assert profile_similarity_residual(locked_axis_regular_profile()) == 0
origin = check_ns_core_candidate(NS_CORE_ORIGIN)
assert origin is not None and origin.ok
second = check_ns_core_candidate(NS_CORE_SECOND_WITNESS)
assert second is not None and second.ok
opposite = check_ns_core_candidate(NS_CORE_OPPOSITE)
assert opposite is not None and opposite.ok is False
assert opposite.payload["cone_reason"] == "opposite_cone"
empty = run_ns_core_discovery(budget=0)
assert empty.search_incomplete
assert honesty_payload()["navier_stokes_proof_claim"] is False
assert honesty_payload()["forced_blowup_reproof_claim"] is False
```
