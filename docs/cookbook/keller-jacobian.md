# Keller Jacobian replay and tangent-sweep search

Alpöge's 2026 map is a finite exact-`Q` identity: a polynomial
`F: Q^3 -> Q^3` with `det J(F) ≡ -2` that sends three distinct points to
one image. This page replays that identity and runs a **blind** deg-2
tangent-sweep search (Gao arXiv:2608.00222). Finding a map is a stress-test
of the engine. It is **not** a claim that omnibias refuted the Jacobian
conjecture. Dimension `n = 2` is a separate finite box:
[Jacobian n=2 finite box](jacobian-n2-box.md).

The closed-form `σ^(n)` tower is not used here. The work is exact rational
algebra in `omnibias-holonomic`.

```python
from omnibias.holonomic.keller import verify_alpoge_map, verify_gallagher_map

alpoge = verify_alpoge_map()
assert alpoge.replay_ok
assert alpoge.jacobian_constant == -2
assert alpoge.honesty["jacobian_conjecture_proof_claim"] is False
assert alpoge.honesty["jacobian_n2_claim"] is False
assert alpoge.honesty["discovered_by_omnibias"] is False

gallagher = verify_gallagher_map()
assert gallagher.replay_ok
assert gallagher.jacobian_constant == 2
```

## Blind sweep

`search_tangent_sweep` enumerates deg-2 curves and affine parameters. It
does not seed Alpöge's `p` or the published witness triple. A hit that
equals those parameters is a rediscovery; any other `p` is a new map in
the same family.

```python
from omnibias.holonomic.keller import matches_known_replay, search_hit_certificate
from omnibias.holonomic.keller_search import search_tangent_sweep

hits = search_tangent_sweep(deg_p=2, height=6, max_hits=1)
assert hits
payload = search_hit_certificate(hits[0])
assert payload["honesty"]["jacobian_conjecture_proof_claim"] is False
if matches_known_replay(hits[0]):
    assert payload["rediscovered_known_witness"] is True
    assert payload["discovered_by_omnibias"] is False
```

## Fiber degree (no Groebner)

The tangency polynomial of the underlying plane curve has constant leading
coefficient. For Alpöge's cubic that degree is 3.

```python
from omnibias.holonomic.keller import fiber_report

report = fiber_report()
assert report["degree"] == 3
assert report["leading_coeff_constant"] is True
```

`PROVED` on the holonomic machine means those finite identities hold. It
does not mean the Jacobian conjecture is solved.
