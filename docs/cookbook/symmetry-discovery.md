# Lie symmetry discovery

The heat equation's scaling generator is a linear combination
in the affine ansatz. Exact jets put it in the nullspace;
finite-difference prolongation does not recover the same rank.

```python
from omnibias.symbolic.symmetry import (
    DISCLAIMER,
    affine_basis,
    designed_samples,
    discover_symmetries,
    heat_known_coeffs,
    determining_matrix,
    pr_heat,
    pr_heat_fd,
    suite,
)

heat = next(p for p in suite() if p.name == "heat")
basis = affine_basis()
samples = designed_samples(32)
exact = discover_symmetries(pr_heat, heat.restrict, basis, samples)
assert exact.algebra_dim == 5
assert exact.separation > 1e6
mat = determining_matrix(pr_heat, heat.restrict, basis, samples)
for vec in heat_known_coeffs():
    assert abs(mat @ vec).max() < 1e-12
fd = discover_symmetries(pr_heat_fd, heat.restrict, basis, samples)
assert fd.algebra_dim != exact.algebra_dim
assert "ansatz" in DISCLAIMER
```

A generic forced heat equation has no point symmetry in this
basis.

```python
from omnibias.symbolic.symmetry import (
    affine_basis,
    designed_samples,
    discover_symmetries,
    negative_control,
    pr_for,
)

spec = negative_control()
res = discover_symmetries(
    pr_for(spec.name), spec.restrict, affine_basis(), designed_samples(32)
)
assert res.algebra_dim == 0
```
