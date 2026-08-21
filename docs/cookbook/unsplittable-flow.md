# Unsplittable-flow cost separation (DGG)

The 1999 Dinitz–Garg–Goemans **congestion** theorem
(`y_a ≤ x_a + d_max`) stays true. The Goemans **cost** conjecture is
already false. This page replays the Isabelle AFP / Rybin instance
(fractional cost 58, every congestion-legal unsplittable at least 60) and
runs a blind parameter-box search on the same H* 7-vertex topology.

The engine can find a cost-separating instance in the H* family. That is
**not** a claim that omnibias disproved Goemans.

```python
from omnibias.combinatorics.unsplittable import verify_rybin_instance

replay = verify_rybin_instance()
assert replay["replay_ok"] is True
assert replay["fractional_cost"] == "58"
assert replay["min_legal_unsplittable"] == "60"
assert replay["honesty"]["dgg_congestion_theorem_refuted"] is False
assert replay["honesty"]["discovered_by_omnibias"] is False
```

## Blind H* search

Search does not import the published `(58, 60)` pair. The topology (two
Paid/Free routes per terminal) is public; the winning demands, paid-arc
costs, and splits are what the box must find.

```python
from omnibias.combinatorics.unsplittable import (
    matches_known_replay,
    search_dgg,
    search_hit_certificate,
)

hits = search_dgg(family="hstar_parameter_box", max_hits=1)
assert hits
payload = search_hit_certificate(hits[0])
assert payload["honesty"]["dgg_congestion_theorem_refuted"] is False
if matches_known_replay(hits[0]):
    assert payload["rediscovered_known_witness"] is True
else:
    assert payload["discovered_by_omnibias"] is True
```
