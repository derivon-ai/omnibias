# Extremal graph templates (not Erdős 146 / 180)

`C4`, `C6`, `jTemplate`, `kTemplate`, and `pairGraph(4, 2)` are the
finite graphs appearing in the OpenAI compactness / 2-degenerate
formalization. This page checks structural predicates: connected,
bipartite, has a cycle, 2-degenerate. It does **not** claim
`¬IsCompactFamily` or an `atTop` extremal inequality.

```python
from omnibias.combinatorics.extremal import (
    c4,
    is_bipartite,
    is_connected,
    verify_forbidden_family,
    verify_pair_graph,
)

assert is_connected(c4()) and is_bipartite(c4())
family = verify_forbidden_family()
assert family["replay_ok"] is True
assert family["honesty"]["erdos_146_claim"] is False
assert family["honesty"]["erdos_180_claim"] is False

pair = verify_pair_graph()
assert pair["replay_ok"] is True
assert pair["two_degenerate"] is True
assert pair["max_degree"] > 2
assert pair["honesty"]["extremal_graph_replay"] is True
```
