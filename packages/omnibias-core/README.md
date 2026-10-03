# omnibias-core

Shared numerical substrate for closed-form activation derivatives and PINNs.
Every backend imports the same polynomial coefficients and jet combinatorics.
The package has no tensor-framework dependency; vectorized interval routines
use NumPy, and directed transcendental bounds optionally use mpmath.

```bash
pip install 'omnibias-core[verified]'
```

```python
from omnibias.core import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs

print(sigmoid_polynomial_coeffs(3))
print(tanh_polynomial_coeffs(4))
```

| Module | Purpose |
| --- | --- |
| `polynomials`, `bell`, `multi_index` | Shared derivative coefficients and jet composition |
| `spec` | Backend-independent activation specification |
| `multipack`, `scan`, `refine` | Adaptive activation banks |
| `verified` | Outward-rounded intervals, Taylor models, quadrature and numerical enclosures |
| `proof.certificate` | Versioned, hash-sealed numerical certificates |
| `proof.lean_check` | Optional finite rational checks through the Lean kernel |

A finite certificate establishes only its stated numerical obligation. Network
accuracy and PDE solution guarantees depend on the assumptions and domain in
that certificate.

See the [project overview](../../README.md).

## License

Apache-2.0; see [LICENSE](LICENSE).
