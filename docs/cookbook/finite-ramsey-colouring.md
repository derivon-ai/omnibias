# Finite Ramsey colouring (not Erdős 183)

A triangle-free `k`-edge-colouring of `K_n` is a finite witness that
`R_k(3) > n`. The 5-cycle versus its complement on `K_5` is the classical
`R(3,3) > 5` colouring. That does **not** prove
`R_k(3) = k^{Θ(k)}` (Erdős 183 / the OpenAI ten-proofs theorem).

```python
from omnibias.combinatorics.ramsey import (
    is_triangle_free_edge_colouring,
    pentagon_two_colouring,
    verify_pentagon_colouring,
)

colours = pentagon_two_colouring()
assert is_triangle_free_edge_colouring(5, 2, colours)
cert = verify_pentagon_colouring()
assert cert["replay_ok"] is True
assert cert["finite_statement"] == "R_2(3) > 5"
assert cert["honesty"]["erdos_183_claim"] is False
assert cert["honesty"]["ramsey_colouring_replay"] is True
```

`IsSaturated` is a matrix predicate, replayed at tiny `(H, m)`. It is not
a construction of `R_k(3)`.

```python
from omnibias.combinatorics.ramsey import is_saturated, verify_saturated_smoke

assert is_saturated(2, 2, 1, ((0, 1),))
assert is_saturated(3, 2, 1, ((0, 1, 2),))
sat = verify_saturated_smoke()
assert sat["replay_ok"] is True
assert sat["honesty"]["erdos_183_claim"] is False
```
