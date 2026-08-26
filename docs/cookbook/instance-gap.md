# Instance-gap tightening

`tighten_gap` runs increasing Lasserre levels and keeps the **best
sound** lower bound. The sandwich is `lower <= OPT <= energy`. The gap
is reported. It is never claimed tight, and it is never P = NP.

## Named one-variable MaxSAT

```python
from omnibias.discrete import brute_force_min, decode, tighten_gap
from omnibias.discrete.maxsat import max_sat

prob = max_sat([[1], [-1]], weights=[1.0, 1.0])
_, opt = brute_force_min(prob)
assignment, _ = decode(prob, n_starts=4)
tightened = tighten_gap(prob, assignment, levels=(1, 2), bisection_steps=8)
assert tightened.tight is False
assert tightened.honesty()["p_vs_np_claim"] is False
assert tightened.certificate.lower_bound <= opt + 1e-6
assert tightened.certificate.energy >= opt - 1e-9
assert tightened.level_used in tightened.levels_tried
```

## Honesty

The smoke
[`instance_gap_tightening_smoke.json`](../benchmarks/instance_gap_tightening_smoke.json)
records median gap ratio on a named `n<=8` family. Combinatorics
LP-dual tightness is a different object and is not a P = NP story.

## See also

- API: `omnibias.discrete.certify`
- Ledger: [`frontier-ledger.md`](../frontier-ledger.md)
