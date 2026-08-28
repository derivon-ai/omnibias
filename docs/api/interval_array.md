# Interval arrays and verified linear solves

`IntervalArray` is the dense, vectorized companion to scalar
`omnibias.core.verified.Interval`. It stores one binary64 lower-endpoint array
and one upper-endpoint array. Arithmetic uses `numpy.nextafter` to round each
computed endpoint outward, under the same IEEE-754 assumptions as the scalar
interval implementation.

Use it for elementwise enclosure arithmetic, interval dot products, and dense
or COO sparse matrix-vector products. It is a small rigorous substrate, not a
general sparse-array package.

```python
import numpy as np

from omnibias.core.verified import IntervalArray, dot, sparse_matvec

matrix = IntervalArray([[1.0, 2.0], [-1.0, 3.0]], [[1.0, 2.0], [-1.0, 3.0]])
vector = IntervalArray.point([2.0, 1.0])
assert np.all(matrix.matvec(vector).contains([4.0, 1.0]))
assert dot(vector, vector).contains(5.0)

# COO entries represent the same matrix.
sparse = sparse_matvec(
    rows=[0, 0, 1, 1],
    columns=[0, 1, 0, 1],
    data=[1.0, 2.0, -1.0, 3.0],
    vector=vector,
    shape=(2, 2),
)
assert np.all(sparse.contains([4.0, 1.0]))
```

`interval_solve` returns an enclosure for each component of `A⁻¹ f`. The
general path uses a Krawczyk contraction; it raises `ValueError` when the
contraction test cannot certify the solve. For symmetric systems,
`symmetric=True` selects a sign-definite interval LDLᵀ path.

```python
from omnibias.core.verified import interval_solve

solution = interval_solve(((2.0, 0.0), (0.0, 3.0)), (4.0, 9.0))
assert solution[0].contains(2.0)
assert solution[1].contains(3.0)
```

These are finite-dimensional numerical enclosures. They do not, by
themselves, establish a continuum spectral or PDE statement.

## Certified inertia of a symmetric matrix box

`interval_ldlt_inertia_array` is the `IntervalArray` twin of the scalar
`interval_ldlt_inertia` reference path in
`omnibias.core.verified.eig_operator`. It runs the same left-looking interval
LDLᵀ factorization one column at a time on endpoint arrays, applying the same
outward-rounded primitives in the same order, so its pivot enclosures agree
with the scalar path endpoint for endpoint.

Once every pivot interval is sign-definite, `S = L D Lᵀ` is a congruence and
Sylvester's law of inertia fixes the signature of every symmetric point matrix
in the box. When a pivot straddles zero the sign — and hence the inertia — is
uncertified, and the call returns `None` rather than a guess. Only the lower
triangle and the diagonal of the input are read.

```python
import numpy as np

from omnibias.core.verified import (
    IntervalArray,
    interval_ldlt_inertia_array,
    interval_ldlt_pivots_array,
    is_positive_definite_array,
)

midpoint = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
box = IntervalArray(midpoint - 1e-6, midpoint + 1e-6)

inertia = interval_ldlt_inertia_array(box)
assert inertia is not None
assert (inertia.negative, inertia.positive) == (0, 3)
assert is_positive_definite_array(box)

pivots = interval_ldlt_pivots_array(box)
assert pivots is not None
assert np.all(pivots.lo > 0.0)

# A box wide enough to contain a singular matrix is reported as uncertified.
assert interval_ldlt_inertia_array(IntervalArray(midpoint - 4.0, midpoint + 4.0)) is None
```

This is a statement about the supplied finite matrix box. It is not a
continuum, operator-limit, or spectral-asymptotics result.

## Non-self-adjoint finite-matrix eigenvalue counts

`count_eigenvalues_in_contour` certifies the algebraic eigenvalue count of a
finite real matrix inside a circular or rectangular contour. It combines the
winding number of `det(z I - A)` with an interval solve of the real block form
of `z I - A` on every contour segment. If the boundary resolvent or winding is
inconclusive, it returns an uncertified result with `count=None`.

```python
from omnibias.core.verified import count_eigenvalues_in_contour

# A non-symmetric triangular matrix with eigenvalues 1/4 and 3/2.
certificate = count_eigenvalues_in_contour(((0.25, 0.2), (0.0, 1.5)), radius=1.0)
assert certificate.certified
assert certificate.count == 1
```

This is an argument-principle count for the supplied finite matrix, including
algebraic multiplicity. It is not a continuum-spectrum, operator-limit, or
pseudospectral result.

## API

::: omnibias.core.verified.interval_array
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.linalg
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.linalg_array
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.nonselfadjoint
    options:
      show_root_heading: false
      heading_level: 3
